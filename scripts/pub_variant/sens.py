"""SENS builds (ours): casmi26-pubm with the saner neural term := a seed ensemble of the published saner recipe.

Called by scripts/make_pub_variant.py SENS=K [SENS_MODE=raw|z] (K = ensemble size incl. the published model, 2..5;
not with OURS / PF_MID_W / ISO_ALPHA / ELIG / ENG3_W) after its own structural edits, the ISO cell included. Nothing of
the notebook is executed; every edit is an exact-anchor splice or an inserted cell:
  cell 1            SENS / SENS_MODE / SENS_SEEDS / SENS_MAX_SECONDS lines after the ISO knobs;
  new code cell     scripts/pub_variant/sens_cell.py (id ours-sens-ensemble), right after the PF-syn cell (id pfsyn0001):
                    definitions only, like the PF-syn cell;
  driver cell       (id 67514abd) a guarded call ours_seed_ensemble() after the PF-syn call and before
                    build_consensus(), i.e. after score_final_lists / score_outside_candidates;
  markdown cell 0   what this build runs; it and the section-4 markdown cell renumbered (the final layer is cell 17,
                    ISO cell 18 once the new cell sits at 12), as are casmi.ours final_layer / iso (ours_meta);
  metadata          notebook casmi.ours.sens (the record below); kernel-metadata kernel_sources :=
                    ['bryantyu/casmi26-saner-seeds'] (its output holds final_model_seed<s>.pt + final_models_sha256.json).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_KERNEL = 'bryantyu/casmi26-saner-seeds'
PUBLISHED_SEED = 20261023
SEEDS = [20262023, 20263023, 20264023, 20265023]     # the extra seeds, in the holdout evaluation's order
MAX_SECONDS = 3600     # 8 scorer runs, each re-featurising ~10k (lists) or ~8k (outsiders) SMILES with RDKit tautomers in
                       # the hidden rerun; the cell's own 8.0 h cap keeps the fragment check's 1500 s (it starts by 8.55 h)
CELL_ID = 'ours-sens-ensemble'
PF_CELL_ID = 'pfsyn0001'
DRIVER_ID = '67514abd'
C1_ANCHOR = ("ISO_FREEZE_LEADER = False     # True: rank 1 never moves (ranks 2-25 only; scope 'formula' keeps the leader's "
             "formula)\n")
DRIVER_ANCHOR = ("    PF_REPORT = {'status': 'wrapper_error', 'error': repr(_pf_exc)[:300]}\n"
                 "CONSENSUS_OK, REPORT, MOVES = build_consensus()\n")
DRIVER_BLOCK = ("try:\n"
                "    SENS_REPORT = ours_seed_ensemble()   # saner seed ensemble in the neural term; published scores on any failure\n"
                "except Exception as _sens_exc:\n"
                "    print('WARNING SENS wrapper error, published saner scores kept:', repr(_sens_exc)[:300], flush=True)\n"
                "    SENS_REPORT = {'status': 'wrapper_error', 'error': repr(_sens_exc)[:300]}\n")
TITLE0 = '# Variant M core + midx final layer + PF-syn + ISO isomer reranker\n\n'
FINAL_ID, ISO_ID = 'busyaprime-variant-m-merge', 'ours-iso-rerank'
SEC4 = '## 4. What this run did\n'
RENUMBER0 = [('the final layer (cell 16) is\n> cell 15 of', 'the final layer (cell 17) is\n> cell 15 of'),
             ('PeakFormer; cell 17 is\n> our ISO isomer reranker.', 'PeakFormer; cell 18 is\n> our ISO isomer reranker.')]
RENUMBER4 = [('fill\n(cell 16) and, when ISO_ALPHA > 0, the ISO reranker (cell 17);',
              'fill\n(cell 17) and, when ISO_ALPHA > 0, the ISO reranker (cell 18);')]
MODES = ('raw', 'z')


def _line(code, comment):
    return code.ljust(29) + ' # ' + comment + '\n'


def cell1_block(k, mode):
    assert isinstance(k, int) and 2 <= k <= 1 + len(SEEDS) and mode in MODES, (k, mode)
    return ("\n"
            "# Ours (SENS): saner seed ensemble in the neural term (the cell after PF-syn; scripts/make_pub_variant.py SENS=K):\n"
            "# neural_score / neural_cos / neural_fp of the final lists and the outside candidates := the ensemble of the\n"
            "# published saner model and the first SENS - 1 seeds of SENS_SEEDS (final_model_seed<s>.pt of the kernel output\n"
            "# bryantyu/casmi26-saner-seeds, sha256 checked); the published scores on any failure.\n"
            + _line(f'SENS = {k!r}', 'ensemble size incl. the published model (seed 20261023); 0 = off')
            + _line(f'SENS_MODE = {mode!r}', "'raw': mean neural_score; 'z': mean within-molecule z (cos / fp: means)")
            + f'SENS_SEEDS = {SEEDS!r}  # the extra seeds, in the holdout evaluation\'s order\n'
            + _line(f'SENS_MAX_SECONDS = {MAX_SECONDS!r}', 'budget of the extra scorings (also stops past 8.0 h of session)'))


def cell1_edits(k, mode):
    return [(C1_ANCHOR, C1_ANCHOR + cell1_block(k, mode))]


def check_config(cfg, k):
    """None, or why the cell-1 configuration cannot carry SENS = k."""
    if cfg.get('OURS') or cfg.get('PF_MID_W', 0) > 0 or cfg.get('ISO_ALPHA', 0) > 0:
        return ('SENS replaces the saner neural term; it is not combined with PF-syn (OURS / PF_MID_W) or ISO for now: '
                'OURS=False PF_MID_W=0 ISO_ALPHA=0')
    seeds = cfg.get('SENS_SEEDS')
    if not (isinstance(seeds, list) and all(type(s) is int for s in seeds) and len(set(seeds)) == len(seeds)
            and PUBLISHED_SEED not in seeds):
        return f'SENS_SEEDS {seeds!r}: a list of distinct int seeds without the published {PUBLISHED_SEED}'
    if len(seeds) < k - 1:
        return f'SENS={k} needs {k - 1} extra seeds; SENS_SEEDS has {len(seeds)}'
    t = cfg.get('SENS_MAX_SECONDS')
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not t > 0 or t != t or t == float('inf'):
        return f'SENS_MAX_SECONDS {t!r}: a finite number of seconds > 0'
    return None


def cell_source():
    return open(os.path.join(HERE, 'sens_cell.py'), encoding='utf-8').read()


def md0(text, k, mode, seeds):
    assert text.startswith(TITLE0), text[:80]
    return (f"# Variant M core + midx final layer + SENS: saner seed ensemble (K {k}, {mode})\n\n"
            f"> **SENS (this build: scripts/make_pub_variant.py SENS={k} SENS_MODE={mode}).** The neural term is a seed\n"
            f"> ensemble of the published saner recipe: the published final_model.pt (seed {PUBLISHED_SEED}) and the refits of\n"
            f"> seeds {', '.join(map(str, seeds))} (final_model_seed<s>.pt, output of our kernel\n"
            "> bryantyu/casmi26-saner-seeds: the published train.py refit with another seed, sha256 checked against its\n"
            "> final_models_sha256.json). The pinned neural_scores.py scores the final lists and the outside candidates once\n"
            "> per extra seed (the same inputs and command line as the published scorer, the same model code and config);\n"
            + ("> neural_score := the members' mean score" if mode == 'raw' else
               "> neural_score := the members' mean within-molecule z-score (z over the listed candidates; the outside\n"
               "> candidates mapped with the same per-molecule maps)")
            + ", neural_cos / neural_fp := the members' means, for every molecule\n"
            "> every member scored (neural_ok, finite); the published values otherwise and on any failure. Everything\n"
            "> after it (consensus, fragment check, final layer) is unchanged. PF-syn re-weighting (OURS, PF_MID) and ISO are\n"
            "> off (OURS = False, ISO_ALPHA = 0), as in casmi26-pubm. Report: sens_report.json and the SENS lines.\n>\n"
            + splice(text[len(TITLE0):], RENUMBER0, 'cell 0 (sens: cell numbers)'))


def ours_meta(ours):
    """casmi.ours final_layer / iso of the generator with the SENS build's cell numbers (17 / 18)."""
    return dict(final_layer=splice(ours['final_layer'], [('(cell 16)', '(cell 17)')], 'casmi.ours.final_layer'),
                iso=splice(ours['iso'], [('(cell 17)', '(cell 18)')], 'casmi.ours.iso'))


def splice(text, edits, what):
    for old, new in edits:
        assert text.count(old) == 1, f'{what}: {old[:70]!r} found {text.count(old)} times'
        text = text.replace(old, new)
    return text


def apply(nb, meta, k, mode, cfg):
    """Turn the generated pub variant (nb, meta) into the SENS build in place; returns a report (record = the notebook
    metadata casmi.ours.sens)."""
    assert isinstance(k, int) and 2 <= k <= 1 + len(cfg['SENS_SEEDS']) and mode in MODES, (k, mode)
    assert cfg['SENS'] == k and cfg['SENS_MODE'] == mode, 'cell 1 SENS lines'
    assert check_config(cfg, k) is None, check_config(cfg, k)
    assert 'ELIG' not in cfg and 'ENG3_W' not in cfg, 'SENS builds are neither ELIG nor ENG3 builds'
    cells = nb['cells']
    ids = [c.get('id') for c in cells]
    assert CELL_ID not in ids and ids.count(PF_CELL_ID) == 1 and ids.count(DRIVER_ID) == 1, 'notebook layout'
    i_pf = ids.index(PF_CELL_ID)
    assert ids.index(DRIVER_ID) == i_pf + 1, 'the driver cell follows the PF-syn cell'
    i_drv = i_pf + 1
    drv = cells[i_drv]['source']
    assert drv.index('OUTSIDERS_OK = score_outside_candidates(') < drv.index('PF_REPORT = ours_pf_blend()') < \
        drv.index('CONSENSUS_OK, REPORT, MOVES = build_consensus()'), 'driver order'
    cells[i_drv]['source'] = splice(drv, [(DRIVER_ANCHOR, DRIVER_ANCHOR.replace(
        'CONSENSUS_OK, REPORT, MOVES', DRIVER_BLOCK + 'CONSENSUS_OK, REPORT, MOVES'))], 'driver cell (sens)')
    cells.insert(i_pf + 1, {'cell_type': 'code', 'execution_count': None, 'id': CELL_ID,
                            'metadata': {'_kg_hide-input': True, 'jupyter': {'source_hidden': True}}, 'outputs': [],
                            'source': cell_source()})
    seeds = list(cfg['SENS_SEEDS'][:k - 1])
    cells[0]['source'] = md0(cells[0]['source'], k, mode, seeds)
    ids = [c.get('id') for c in cells]
    assert (ids.index(FINAL_ID), ids.index(ISO_ID)) == (17, 18), 'final layer / ISO cells after the insertion'
    i4 = [i for i, c in enumerate(cells) if c['cell_type'] == 'markdown' and c['source'].startswith(SEC4)]
    assert len(i4) == 1, i4
    i4 = i4[0]
    cells[i4]['source'] = splice(cells[i4]['source'], RENUMBER4, f'cell {i4} (section 4; sens: cell numbers)')
    assert meta.get('kernel_sources') == [], meta.get('kernel_sources')
    meta['kernel_sources'] = [SOURCE_KERNEL]
    record = dict(
        k=k, mode=mode, published_seed=PUBLISHED_SEED, extra_seeds=seeds, source=f'{SOURCE_KERNEL} (kernel output: '
        'final_model_seed<s>.pt, final_models_sha256.json)', cell=f'scripts/pub_variant/sens_cell.py (id {CELL_ID}, '
        f'cell {i_pf + 1})', call=f'driver cell {i_pf + 2} (id {DRIVER_ID}): after ours_pf_blend, before build_consensus',
        rule=("raw: neural_score := mean of the members' neural_score" if mode == 'raw' else
              'z: neural_score := mean of the members\' within-molecule z (listed candidates; outsiders the same maps)')
        + '; neural_cos / neural_fp := member means; molecules with every row processed, neural_ok and finite for every '
        'member; published values otherwise and on any failure', max_seconds=cfg['SENS_MAX_SECONDS'])
    return dict(record=record, cell_index=i_pf + 1, driver_index=i_pf + 2, cells_edited=[0, 1, i_pf + 2, i4],
                cell_inserted=i_pf + 1)
