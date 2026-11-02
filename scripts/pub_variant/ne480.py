"""NE480=1: the neural ensemble of bgunshui/casmi26-480-neural-ensemble (public LB 0.480 on bgunshui's and on alexchilton's
account, the latter a verbatim rerun; our verbatim fork is kernels/casmi26-ne480) on a build of make_pub_variant.py.

That notebook is casmi26-pubm (OURS False, PF_MID_W 0) plus, in the driver cell (id 67514abd), two blocks copied here as
text (the code runs only on Kaggle):
  SWAP_BLOCK after materialise_sources(): finds a scorer bundle other than the public kit (final_model.pt next to
             config_used.json: the kernel output of bgunshui/casmi26-scorer-train-public, the published saner kit's
             train.py run with 80 epochs on train.parquet) and stages it as OURS_BUNDLE (public model.py / features.py /
             config.json, its final_model.pt); without one it keeps the public bundle.
  ENS_BLOCK  around score_final_lists(): the final lists are scored with the public model and with OURS_BUNDLE, then
             neural_score := 0.5 z(public) + 0.5 z(ours) (within-molecule z), neural_cos / neural_fp / neural_std := means,
             neural_ok := both; the outside candidates are then scored with OURS_BUNDLE alone (BUNDLE stays OURS_BUNDLE).
  FIND_NEW   in materialise_sources() (cell id ac4866c0): bgunshui's fallback when find_bundle sees two bundles with
             the public pins (the kernel output also carries kit_staged/, a copy of the public kit): the shortest
             path containing casmi-spectrum-model-v1.
Kernel source: bgunshui/casmi26-scorer-train-public (kernel-metadata kernel_sources). With ELIG=1 the eligibility cell 8
counts it as a listed mount (elig.cell8_kernel_source_edits). Not combined with SENS / ENG3_W / OURS (one neural-term
change at a time).
"""
import os

SOURCE_KERNEL = 'bgunshui/casmi26-scorer-train-public'
DRIVER_ID = '67514abd'
ANCHOR_SWAP = 'BUNDLE = materialise_sources()\n'
ANCHOR_ENS = 'NEURAL_OK = score_final_lists()\n'
SWAP_BLOCK = '# ---- private-model swap: use our own longer-trained spectrum model ----\ndef _swap_to_ours(base_bundle):\n    """Find our private scorer bundle and use it as the neural scorer.\n\n    Accepts any attached data source that contains final_model.pt + config_used.json\n    (works for a Kaggle dataset or for a kernel output), while ignoring the public\n    kit bundle and the staged copy of it.\n    """\n    import shutil as _s\n    from pathlib import Path as _P\n    _hits = []\n    for _p in _P(\'/kaggle/input\').rglob(\'final_model.pt\'):\n        _par = _p.parent\n        if \'casmi-spectrum-model-v1\' in str(_p) or \'kit_staged\' in str(_p):\n            continue\n        if (_par / \'config_used.json\').is_file() or \'casmi_spectrum_model_ours\' in str(_p) or \'scorer-v2-model\' in str(_p):\n            _hits.append(_p)\n    if not _hits:\n        print(\'OURS: private model not found under /kaggle/input, keeping the public bundle\')\n        return base_bundle\n    _src = _hits[0].parent\n    _dst = _P(\'/kaggle/working/ours_bundle\'); _dst.mkdir(parents=True, exist_ok=True)\n    _s.copy2(_P(base_bundle) / \'model.py\', _dst / \'model.py\')\n    _s.copy2(_P(base_bundle) / \'features.py\', _dst / \'features.py\')\n    _cfg = _src / \'config_used.json\'\n    if _cfg.is_file():\n        _s.copy2(_cfg, _dst / \'config_used.json\')\n    _s.copy2(_P(base_bundle) / \'config.json\', _dst / \'config.json\')\n    _s.copy2(_src / \'final_model.pt\', _dst / \'final_model.pt\')\n    print(\'OURS BUNDLE READY:\', sorted(p.name for p in _dst.iterdir()), \'from\', _src)\n    return str(_dst)\n\n\nPUB_BUNDLE = BUNDLE\nOURS_BUNDLE = _swap_to_ours(PUB_BUNDLE)\nBUNDLE = OURS_BUNDLE\n'
ENS_BLOCK = "BUNDLE = PUB_BUNDLE\nNEURAL_OK = score_final_lists()\nshutil.copyfile(ROOT / 'neural_candidate_features.parquet', ROOT / 'neural_public.parquet')\nBUNDLE = OURS_BUNDLE\nif score_final_lists():\n    shutil.copyfile(ROOT / 'neural_candidate_features.parquet', ROOT / 'neural_ours.parquet')\n    import pandas as _pd\n    _a = _pd.read_parquet(ROOT / 'neural_public.parquet').set_index(['molecule_id', 'smiles'])\n    _b = _pd.read_parquet(ROOT / 'neural_ours.parquet').set_index(['molecule_id', 'smiles']).reindex(_a.index)\n    def _z(_s):\n        _g = _s.groupby(level=0)\n        return (_s - _g.transform('mean')) / _g.transform(lambda x: x.std(ddof=0)).clip(lower=1e-6)\n    _a['neural_score'] = 0.5 * _z(_a['neural_score'].fillna(0.0)) + 0.5 * _z(_b['neural_score'].fillna(0.0))\n    _a['neural_cos'] = 0.5 * _a['neural_cos'] + 0.5 * _b['neural_cos'].fillna(0.0)\n    _a['neural_fp'] = 0.5 * _a['neural_fp'] + 0.5 * _b['neural_fp'].fillna(0.0)\n    _a['neural_std'] = 0.5 * _a['neural_std'] + 0.5 * _b['neural_std'].fillna(0.0)\n    _a['neural_ok'] = _a['neural_ok'].astype(bool) & _b['neural_ok'].fillna(False).astype(bool)\n    _a.reset_index().to_parquet(ROOT / 'neural_candidate_features.parquet', index=False)\n    print('ENSEMBLE neural = 0.5*z(public) + 0.5*z(ours)', flush=True)\nBUNDLE = OURS_BUNDLE\n"
FIND_CELL_ID = 'ac4866c0'   # materialise_sources(): its find_bundle call
FIND_OLD = '    bundle = find_bundle(str(INPUT), ROOT, MODEL_PINS)\n'
FIND_NEW = "    from pathlib import Path as _BPath\n    def _find_bundle_safe():\n        try:\n            return find_bundle(str(INPUT), ROOT, MODEL_PINS)\n        except AssertionError:\n            _cands = [p.parent for p in _BPath('/kaggle/input').rglob('final_model.pt')\n                      if (p.parent / 'config.json').is_file() and 'casmi-spectrum-model-v1' in str(p)]\n            print('BUNDLE FALLBACK candidates', len(_cands), flush=True)\n            assert _cands, 'public model bundle not found'\n            return str(sorted(_cands, key=lambda p: len(str(p)))[0])\n    bundle = _find_bundle_safe()\n"
SWAP_SHA256 = None   # filled below
import hashlib as _h
SWAP_SHA256 = _h.sha256(SWAP_BLOCK.encode()).hexdigest()
ENS_SHA256 = _h.sha256(ENS_BLOCK.encode()).hexdigest()


def splice(text, edits, what):
    for old, new in edits:
        assert text.count(old) == 1, f'{what}: anchor found {text.count(old)} times: {old[:80]!r}'
        text = text.replace(old, new)
    return text


def driver_edits():
    assert ENS_BLOCK.startswith('BUNDLE = PUB_BUNDLE\n' + ANCHOR_ENS) and ENS_BLOCK.endswith('BUNDLE = OURS_BUNDLE\n')
    assert SWAP_BLOCK.endswith('BUNDLE = OURS_BUNDLE\n')
    return [(ANCHOR_SWAP, ANCHOR_SWAP + SWAP_BLOCK), (ANCHOR_ENS, ENS_BLOCK)]


# ---- NE_SRC != 'bgunshui' (ours): other extra scorers, equal-weight z ensemble (rank line only) ----
NE_SOURCES = {   # NE_SRC: [(label, path markers of its final_model.pt under /kaggle/input)], dataset / kernel sources
    'ahmed': dict(members=[('long', ['spectrum-model-long'])],
                  datasets=['ahmedberatozer/casmi26-spectrum-model-long'], kernels=[]),
    'both': dict(members=[('ours80', ['casmi26-scorer-train-public']), ('long', ['spectrum-model-long'])],
                 datasets=['ahmedberatozer/casmi26-spectrum-model-long'], kernels=[SOURCE_KERNEL])}
GEN_SWAP_TMPL = """
# ---- NE ensemble (ours, NE_SRC={src}): extra scorer bundles staged next to the public one ----
NE_SOURCES = {members!r}


def _ne_stage(base_bundle, label, markers):
    \"\"\"Stage an extra scorer: the public model.py / features.py / config.json + that source's final_model.pt.\"\"\"
    import shutil as _s
    from pathlib import Path as _P
    hits = sorted((p for p in _P('/kaggle/input').rglob('final_model.pt')
                   if any(m in str(p) for m in markers) and 'kit_staged' not in str(p)), key=lambda p: (len(str(p)), str(p)))
    if not hits:
        print(f'NE {{label}}: no final_model.pt under /kaggle/input matching {{markers}}; member skipped', flush=True)
        return None
    src = hits[0].parent
    dst = _P(f'/kaggle/working/ne_{{label}}_bundle'); dst.mkdir(parents=True, exist_ok=True)
    for n in ('model.py', 'features.py', 'config.json'):
        _s.copy2(_P(base_bundle) / n, dst / n)
    _s.copy2(src / 'final_model.pt', dst / 'final_model.pt')
    print(f'NE {{label}} BUNDLE READY from {{src}}', flush=True)
    return str(dst)


PUB_BUNDLE = BUNDLE
NE_MEMBERS = [(_l, _ne_stage(PUB_BUNDLE, _l, _m)) for _l, _m in NE_SOURCES]
NE_MEMBERS = [(_l, _b) for _l, _b in NE_MEMBERS if _b]
"""
GEN_ENS = """BUNDLE = PUB_BUNDLE
NEURAL_OK = score_final_lists()
NE_REPORT = dict(sources=[_l for _l, _ in NE_SOURCES], staged=[_l for _l, _ in NE_MEMBERS], used=[])
if NEURAL_OK and NE_MEMBERS:
    import pandas as _pd
    shutil.copyfile(ROOT / 'neural_candidate_features.parquet', ROOT / 'neural_public.parquet')
    _parts = [_pd.read_parquet(ROOT / 'neural_public.parquet').set_index(['molecule_id', 'smiles'])]
    for _l, _b in NE_MEMBERS:
        BUNDLE = _b
        if score_final_lists():
            shutil.copyfile(ROOT / 'neural_candidate_features.parquet', ROOT / f'neural_{_l}.parquet')
            _parts.append(_pd.read_parquet(ROOT / f'neural_{_l}.parquet').set_index(['molecule_id', 'smiles'])
                          .reindex(_parts[0].index))
            NE_REPORT['used'].append(_l)
    if len(_parts) > 1:
        def _z(_s):
            _g = _s.groupby(level=0)
            return (_s - _g.transform('mean')) / _g.transform(lambda x: x.std(ddof=0)).clip(lower=1e-6)
        _w = 1.0 / len(_parts)
        _a = _parts[0].copy()
        _a['neural_score'] = sum(_w * _z(_p['neural_score'].fillna(0.0)) for _p in _parts)
        for _c in ('neural_cos', 'neural_fp', 'neural_std'):
            _a[_c] = sum(_w * _p[_c].fillna(0.0) for _p in _parts)
        _ok = _parts[0]['neural_ok'].astype(bool)
        for _p in _parts[1:]:
            _ok = _ok & _p['neural_ok'].fillna(False).astype(bool)
        _a['neural_ok'] = _ok
        _a.reset_index().to_parquet(ROOT / 'neural_candidate_features.parquet', index=False)
        print('ENSEMBLE neural = equal z mean of public +', NE_REPORT['used'], f'(w {_w:.4f} each)', flush=True)
    else:
        shutil.copyfile(ROOT / 'neural_public.parquet', ROOT / 'neural_candidate_features.parquet')
        print('NE WARNING: no extra scorer scored the lists; published neural term kept', flush=True)
    BUNDLE = dict(NE_MEMBERS)[NE_REPORT['used'][-1]] if NE_REPORT['used'] else PUB_BUNDLE
"""


def generic_edits(src):
    """Driver-cell edits for NE_SRC in NE_SOURCES (outside candidates are then scored by the last extra scorer used,
    as upstream scores them by its extra scorer)."""
    members = NE_SOURCES[src]['members']
    return [(ANCHOR_SWAP, ANCHOR_SWAP + GEN_SWAP_TMPL.format(src=src, members=members).lstrip('\n')),
            (ANCHOR_ENS, GEN_ENS)]


MD_NOTE = ("> **NE480 (ours: make_pub_variant.py NE480=1).** The neural term is the ensemble of bgunshui's *CASMI26 480 neural "
           "ensemble* (public LB 0.480): the final lists are scored by the published saner model and by the 80-epoch refit "
           "of the same kit (kernel output bgunshui/casmi26-scorer-train-public, attached as a kernel source), neural_score "
           "= 0.5 z(published) + 0.5 z(refit) per molecule, neural_cos / neural_fp / neural_std the means; outside "
           "candidates are scored by the refit. Everything else is this build's.\n\n")


MD_NOTE_GEN = ("> **NE ensemble (ours: make_pub_variant.py NE480=1 NE_SRC={src}).** The neural term is the equal-weight "
               "within-molecule z mean of the published saner model and the extra scorer(s) {members} (the kit retrained "
               "with longer schedules; sources {datasets}); neural_cos / neural_fp / neural_std are the means; outside "
               "candidates are scored by the last extra scorer. After bgunshui's CASMI26 480 neural ensemble (public LB "
               "0.480). Everything else is this build's.\n\n")


def apply(nb, meta, cfg, elig=None, src='bgunshui'):
    """In place; elig = the loaded elig.py module for ELIG=1 builds (None otherwise); src = NE_SRC ('bgunshui': the
    upstream blocks verbatim; 'ahmed' / 'both': our generic equal-weight blocks, rank line only). Returns the record."""
    assert src == 'bgunshui' or src in NE_SOURCES, src
    assert src == 'bgunshui' or elig is None, 'NE_SRC other than bgunshui is not an eligible build'
    assert not cfg.get('OURS') and not cfg.get('SENS') and not cfg.get('ENG3_W'), 'NE480 alone on the neural term'
    cells = nb['cells']
    ids = [c.get('id') for c in cells]
    assert ids.count(DRIVER_ID) == 1, 'driver cell'
    i = ids.index(DRIVER_ID)
    cells[i]['source'] = splice(cells[i]['source'], driver_edits() if src == 'bgunshui' else generic_edits(src),
                                f'driver cell (ne480, {src})')
    assert ids.count(FIND_CELL_ID) == 1, 'materialise_sources cell'
    j = ids.index(FIND_CELL_ID)
    cells[j]['source'] = splice(cells[j]['source'], [(FIND_OLD, FIND_NEW)], 'materialise_sources (ne480)')
    t = cells[0]['source']
    head, rest = t.split('\n', 1)
    note = MD_NOTE if src == 'bgunshui' else MD_NOTE_GEN.format(src=src, members=', '.join(
        l for l, _ in NE_SOURCES[src]['members']), datasets=', '.join(NE_SOURCES[src]['datasets'] + NE_SOURCES[src]['kernels']))
    cells[0]['source'] = head + '\n\n' + note + rest.lstrip('\n')
    assert meta.get('kernel_sources') == [], meta.get('kernel_sources')
    if src == 'bgunshui':
        meta['kernel_sources'] = [SOURCE_KERNEL]
    else:
        meta['kernel_sources'] = list(NE_SOURCES[src]['kernels'])
        meta['dataset_sources'] = list(meta['dataset_sources']) + [d for d in NE_SOURCES[src]['datasets']
                                                                   if d not in meta['dataset_sources']]
    edited = [0, j, i]
    if elig is not None:
        i8 = ids.index('3e2e781f')
        assert '# ── Eligibility (ours, elig1)' in cells[i8]['source'], 'cell 8 (elig1 block)'
        cells[i8]['source'] = splice(cells[i8]['source'], elig.cell8_kernel_source_edits(meta['kernel_sources']),
                                     'cell 8 (elig1 mount check: kernel sources)')
        e = nb['metadata']['casmi']['elig']
        e['kernel_sources'] = list(meta['kernel_sources'])
        e['ne480'] = ('kernel output bgunshui/casmi26-scorer-train-public: the published saner kit (sanpier) retrained '
                      'with 80 epochs on train.parquet by a public kernel; reproduction (as F7) and its licence / '
                      'attribution still open')
        edited.append(i8)
    if src != 'bgunshui':
        record = dict(src=src, members=NE_SOURCES[src]['members'], datasets=NE_SOURCES[src]['datasets'],
                      kernels=NE_SOURCES[src]['kernels'], rule='neural_score := equal-weight within-molecule z mean of the published '
                      'model and the extra scorers; cos / fp / std means; ok = all; outsiders scored by the last extra scorer used',
                      driver_cell=i, cells_edited=sorted(edited))
        nb['metadata']['casmi'].setdefault('ours', {})['ne480'] = record
        return record
    record = dict(source=SOURCE_KERNEL + ' (kernel output: casmi_spectrum_model_ours_v2/final_model.pt + config_used.json)',
                  upstream='bgunshui/casmi26-480-neural-ensemble (public LB 0.480; alexchilton verbatim rerun 0.480)',
                  rule='neural_score := 0.5 z(published) + 0.5 z(refit) within molecule; cos / fp / std means; ok = both; '
                       'outsiders scored by the refit', swap_block_sha256=SWAP_SHA256, ens_block_sha256=ENS_SHA256,
                  driver_cell=i, cells_edited=sorted(edited))
    nb['metadata']['casmi'].setdefault('ours', {})['ne480'] = record
    return record
