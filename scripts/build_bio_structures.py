"""ChEBI + LIPID MAPS LMSD candidate structures built from the raw public releases (both CC BY 4.0) -- our own
replacement for sergiomalv/casmi-chebi-lmsd-structures (docs/reproduce.md G1: that set has no build code and was
filtered with a CC-BY-NC-SA key list).

python scripts/build_bio_structures.py <chebi.sdf.gz> <LMSD.sdf | LMSD.sdf.zip> <coconut_2026_10.parquet> <out.parquet> [procs]
Sources: ChEBI release 255 https://ftp.ebi.ac.uk/pub/databases/chebi/archive/rel255/SDF/chebi.sdf.gz (2- and 3-star
entries); LMSD https://www.lipidmaps.org/files/?file=LMSD&ext=sdf.zip (not versioned upstream: record the date / md5).
Standardisation (RDKit 2026.03.3, as the original set): largest organic fragment, uncharged; entries with dummy atoms /
R groups, isotope labels or no carbon dropped; key = standard InChIKey first block. Structures whose key is in our
COCONUT 2026-10 table are removed (COCONUT is already in the pool), so in_bio marks ChEBI / LMSD chemistry outside
COCONUT. Output columns: inchikey14, smiles, source (ChEBI / LIPID MAPS LMSD), source_id, chebi_star,
monoisotopic_mass; one row per key (ChEBI before LMSD, then by source id).
"""
import gzip, io, os, sys, zipfile
from multiprocessing import Pool
import numpy as np, pandas as pd


def records(path, src):
    """(source, id, star, molblock) per SDF record."""
    if path.endswith('.zip'):
        z = zipfile.ZipFile(path)
        name = [n for n in z.namelist() if n.lower().endswith('.sdf')][0]
        fh = io.TextIOWrapper(z.open(name), encoding='utf-8', errors='replace')
    elif path.endswith('.gz'):
        fh = gzip.open(path, 'rt', encoding='utf-8', errors='replace')
    else:
        fh = open(path, encoding='utf-8', errors='replace')
    buf = []
    for line in fh:
        if line.startswith('$$$$'):
            block = ''.join(buf); buf = []
            mol, _, props = block.partition('M  END')
            tags = {}
            cur = None
            for pl in props.splitlines():
                if pl.startswith('> '):
                    cur = pl[pl.find('<') + 1:pl.rfind('>')]
                elif cur and pl.strip():
                    tags.setdefault(cur, pl.strip())
            if src == 'ChEBI':
                sid = tags.get('ChEBI ID') or tags.get('chebi_id') or ''
                star = tags.get('STAR') or tags.get('Star') or ''
            else:
                sid = tags.get('LM_ID') or ''
                star = ''
            yield src, sid, star, mol + 'M  END\n'
        else:
            buf.append(line)


def standardise(rec):
    src, sid, star, mb = rec
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.Descriptors import ExactMolWt
    RDLogger.DisableLog('rdApp.*')
    try:
        m = Chem.MolFromMolBlock(mb, sanitize=True)
    except Exception:
        m = None
    if m is None:
        return ('parse', None)
    if any(a.GetAtomicNum() == 0 for a in m.GetAtoms()) or 'R#' in mb or ' R ' in mb:
        return ('dummy/R-group', None)
    if any(a.GetIsotope() for a in m.GetAtoms()):
        return ('isotope', None)
    try:
        m = rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(m)
        m = rdMolStandardize.Uncharger().uncharge(m)
        Chem.SanitizeMol(m)
    except Exception:
        return ('standardise', None)
    if not any(a.GetAtomicNum() == 6 for a in m.GetAtoms()):
        return ('no carbon', None)
    smi = Chem.MolToSmiles(m)
    m2 = Chem.MolFromSmiles(smi)
    if m2 is None:
        return ('reparse', None)
    ik = Chem.MolToInchiKey(m2)
    if not ik:
        return ('no inchikey', None)
    return ('ok', (ik[:14], smi, src, sid, star, ExactMolWt(m2)))


if __name__ == '__main__':
    chebi, lmsd, coco, out = sys.argv[1:5]
    procs = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    recs = [r for r in records(chebi, 'ChEBI') if r[2] in ('2', '3')] + list(records(lmsd, 'LIPID MAPS LMSD'))
    print('records', len(recs), {s: sum(r[0] == s for r in recs) for s in ('ChEBI', 'LIPID MAPS LMSD')}, flush=True)
    with Pool(procs) as p:
        res = p.map(standardise, recs, chunksize=500)
    drops = {}
    rows = []
    for st, v in res:
        if st == 'ok':
            rows.append(v)
        else:
            drops[st] = drops.get(st, 0) + 1
    B = pd.DataFrame(rows, columns=['inchikey14', 'smiles', 'source', 'source_id', 'chebi_star', 'monoisotopic_mass'])
    B['o'] = (B.source != 'ChEBI').astype(int)
    B = B.sort_values(['o', 'source_id'], kind='stable').drop_duplicates('inchikey14').drop(columns='o')
    n_unique = len(B)
    ck = set(pd.read_parquet(coco, columns=['ik14']).ik14)
    B = B[~B.inchikey14.isin(ck)].reset_index(drop=True)
    B['chebi_star'] = pd.to_numeric(B.chebi_star, errors='coerce').fillna(0).astype(int)
    B.to_parquet(out)
    print('drops', drops, '| standardised', len(rows), 'unique', n_unique, '| outside COCONUT', len(B),
          B.source.value_counts().to_dict(), flush=True)
