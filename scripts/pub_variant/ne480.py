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


MD_NOTE = ("> **NE480 (ours: make_pub_variant.py NE480=1).** The neural term is the ensemble of bgunshui's *CASMI26 480 neural "
           "ensemble* (public LB 0.480): the final lists are scored by the published saner model and by the 80-epoch refit "
           "of the same kit (kernel output bgunshui/casmi26-scorer-train-public, attached as a kernel source), neural_score "
           "= 0.5 z(published) + 0.5 z(refit) per molecule, neural_cos / neural_fp / neural_std the means; outside "
           "candidates are scored by the refit. Everything else is this build's.\n\n")


def apply(nb, meta, cfg, elig=None):
    """In place; elig = the loaded elig.py module for ELIG=1 builds (None otherwise). Returns the record."""
    assert not cfg.get('OURS') and not cfg.get('SENS') and not cfg.get('ENG3_W'), 'NE480 alone on the neural term'
    cells = nb['cells']
    ids = [c.get('id') for c in cells]
    assert ids.count(DRIVER_ID) == 1, 'driver cell'
    i = ids.index(DRIVER_ID)
    cells[i]['source'] = splice(cells[i]['source'], driver_edits(), 'driver cell (ne480)')
    t = cells[0]['source']
    head, rest = t.split('\n', 1)
    cells[0]['source'] = head + '\n\n' + MD_NOTE + rest.lstrip('\n')
    assert meta.get('kernel_sources') == [], meta.get('kernel_sources')
    meta['kernel_sources'] = [SOURCE_KERNEL]
    edited = [0, i]
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
    record = dict(source=SOURCE_KERNEL + ' (kernel output: casmi_spectrum_model_ours_v2/final_model.pt + config_used.json)',
                  upstream='bgunshui/casmi26-480-neural-ensemble (public LB 0.480; alexchilton verbatim rerun 0.480)',
                  rule='neural_score := 0.5 z(published) + 0.5 z(refit) within molecule; cos / fp / std means; ok = both; '
                       'outsiders scored by the refit', swap_block_sha256=SWAP_SHA256, ens_block_sha256=ENS_SHA256,
                  driver_cell=i, cells_edited=sorted(edited))
    nb['metadata']['casmi'].setdefault('ours', {})['ne480'] = record
    return record
