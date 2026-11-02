"""SENS builds (ours): casmi26-pubm with the saner neural term := a seed ensemble of the published saner recipe.

Called by scripts/make_pub_variant.py SENS=K [SENS_MODE=raw|z|swap] (K = ensemble size incl. the published model, 2..5;
not with OURS / PF_MID_W / ISO_ALPHA / ELIG=0) after its own structural edits, the ISO cell included, and after
elig.py (ELIG=1) or eng3.py (ENG3_W) when the build has one of them. SENS=2 SENS_MODE=swap SENS_SEEDS=[s] is the
seed-swap placebo (docs/plan.md, NOV 3 SLOTS): the one extra seed's neural_score / neural_cos / neural_fp replace the
published model's, nothing averaged (the cell is sens_cell.py with the SOURCE_SWAP splices; cell 0 / 1, the elig1
statement, section 5 and the credits say so; every raw / z build is unchanged). Nothing of the notebook is executed;
every edit is an exact-anchor splice or an inserted cell:
  cell 1            SENS / SENS_MODE / SENS_SEEDS / SENS_MAX_SECONDS lines after the ISO knobs;
  new code cell     scripts/pub_variant/sens_cell.py (id ours-sens-ensemble), right after the PF-syn cell (id pfsyn0001):
                    definitions only, like the PF-syn cell;
  driver cell       (id 67514abd) a guarded call ours_seed_ensemble() after the PF-syn call and before
                    build_consensus(), i.e. after score_final_lists / score_outside_candidates;
  markdown cell 0   what this build runs; it and the section-4 markdown cell renumbered (the final layer is cell 17,
                    ISO cell 18 once the new cell sits at 12), as are casmi.ours final_layer / iso (ours_meta);
  metadata          notebook casmi.ours.sens (the record below); kernel-metadata kernel_sources :=
                    ['bryantyu/casmi26-saner-seeds'] (its output holds final_model_seed<s>.pt + final_models_sha256.json).
Bases (cfg of cell 1): 'pubm' (neither ELIG nor ENG3_W), 'eng3' (ENG3_W > 0) or 'elig1' (ELIG = 1). ENG3 / elig1 act inside
the frozen pipeline (frozen cells 32 / 39 (/ 45), before its evidence export); SENS acts on the neural files scored from
that export, in the driver cell: the two compose (SENS leaves the frozen notebook, Engine 2 and the ENG3 / ELIG
modules as they are). Besides the edits above:
  eng3   markdown cell 0: the ENG3 title names SENS, the SENS note follows the ENG3 note (with the composition).
  elig1  inserted cell: sens_source_dir accepts only the kernel output (SOURCE_ELIG1: notebooks/bryantyu/
         casmi26-saner-seeds or /kaggle/input/casmi26-saner-seeds), never another mount; cell 8: the ELIG mount check
         lists the kernel source (elig.cell8_kernel_source_edits; CELL8_ELIG1: the driver is cell 13); markdown cell 0:
         the eligibility statement gets the
         SENS bullet (the refits' eligibility: F7), the ISO cell's new number and F7 as closed; sections 5 / 6 / 8 name
         this build and the kernel output (credits row); notebook casmi.elig: kernel_sources, open / closed and
         notebook_cells_edited in the new numbering.
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
MODES = ('raw', 'z', 'swap')    # 'swap': seed-swap placebo (SENS = 2: the one extra seed's scores replace the published)


def _line(code, comment):
    return code.ljust(29) + ' # ' + comment + '\n'


def cell1_block(k, mode):
    assert isinstance(k, int) and 2 <= k <= 1 + len(SEEDS) and mode in MODES and (mode != 'swap' or k == 2), (k, mode)
    if mode == 'swap':        # SENS_SEEDS: the build command names the seed (SENS_SEEDS=[s], set after these lines)
        return ("\n"
                "# Ours (SENS swap): seed-swap placebo in the neural term (the cell after PF-syn; scripts/make_pub_variant.py\n"
                "# SENS=2 SENS_MODE=swap SENS_SEEDS=[s]): neural_score / neural_cos / neural_fp of the final lists and the outside\n"
                "# candidates := those of the ONE refit of the published saner recipe in SENS_SEEDS (final_model_seed<s>.pt of the\n"
                "# kernel output bryantyu/casmi26-saner-seeds, sha256 checked) in place of the published model's, nothing\n"
                "# averaged; the published scores where the refit did not score and on any failure.\n"
                + _line(f'SENS = {k!r}', 'the published model + the one extra seed (swap: always 2)')
                + _line(f'SENS_MODE = {mode!r}', "'swap': the extra seed's scores replace the published ones (placebo)")
                + _line(f'SENS_SEEDS = {SEEDS[:1]!r}', 'the one extra seed (swap; named by the build command)')
                + _line(f'SENS_MAX_SECONDS = {MAX_SECONDS!r}', 'budget of the extra scorings (also stops past 8.0 h of session)'))
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
    if cfg.get('SENS_MODE') == 'swap' and (k != 2 or len(seeds) != 1):
        return (f'SENS_MODE swap replaces the published model by ONE extra seed: SENS=2 and SENS_SEEDS=[s] '
                f'(here SENS={k}, SENS_SEEDS {seeds!r})')
    t = cfg.get('SENS_MAX_SECONDS')
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not t > 0 or t != t or t == float('inf'):
        return f'SENS_MAX_SECONDS {t!r}: a finite number of seconds > 0'
    return None


def cell_source(mode=None):
    """The inserted cell: sens_cell.py byte for byte ('raw' / 'z'), or with the SOURCE_SWAP splices (mode 'swap')."""
    src = open(os.path.join(HERE, 'sens_cell.py'), encoding='utf-8').read()
    return splice(src, SOURCE_SWAP, 'sens cell (swap)') if mode == 'swap' else src


# ---- SENS_MODE 'swap' (seed-swap placebo, SENS = 2): the cell's swap path, spliced in only for swap builds so that
# sens_cell.py (and every raw / z build that carries it verbatim) stays unchanged. Same molecule gate as the ensemble
# (every row processed, neural_ok and finite for the published model and the seed); the seed's own values, no
# averaging; outside candidates of a swapped molecule take the seed's values where both scored them, else keep theirs.
SOURCE_SWAP = [
    ("#   both modes       neural_cos / neural_fp := means of the members' values (the fragment checks gate on their absolute\n"
     "#                    differences, so they keep their units; in 'raw' mode the mean score stays cos / T + w fp)\n",
     "#   'raw' / 'z'      neural_cos / neural_fp := means of the members' values (the fragment checks gate on their absolute\n"
     "#                    differences, so they keep their units; in 'raw' mode the mean score stays cos / T + w fp)\n"
     "#   SENS_MODE 'swap' (seed-swap placebo; SENS = 2, one extra seed; scripts/pub_variant/sens.py SOURCE_SWAP):\n"
     "#                    neural_score / neural_cos / neural_fp := the extra seed's own values, nothing averaged (the\n"
     "#                    published model's units); a changed molecule's outside candidate takes them when the published\n"
     "#                    model and the seed both scored it (neural_ok, finite), else keeps its published values;\n"
     "#                    sens_members 1 = the seed's values, 0 = the published values kept\n"),
    ("    assert mode in ('raw', 'z'), f'mode {mode!r}'\n",
     "    assert mode in ('raw', 'z', 'swap'), f'mode {mode!r}'\n"),
    ("    assert members_main and len(seeds) == len(members_main) and len(set(map(str, seeds))) == len(seeds), seeds\n",
     "    assert members_main and len(seeds) == len(members_main) and len(set(map(str, seeds))) == len(seeds), seeds\n"
     "    assert mode != 'swap' or len(members_main) == 1, f'swap: exactly one extra member, not {len(members_main)}'\n"),
    ("        if mode == 'raw':\n"
     "            e = s.mean(0)\n",
     "        if mode == 'swap':                          # the extra seed's own values in place of the published ones\n"
     "            e = s[1]\n"
     "            maps[str(mid)] = None\n"
     "        elif mode == 'raw':\n"
     "            e = s.mean(0)\n"),
    ("            new[c][idx] = vals[c][:, idx].mean(0)\n"
     "        nmem[idx] = K\n",
     "            new[c][idx] = vals[c][1, idx] if mode == 'swap' else vals[c][:, idx].mean(0)\n"
     "        nmem[idx] = 1 if mode == 'swap' else K\n"),
    ("                if zm is None:\n"
     "                    e, e1 = s.mean(0), s[0]\n",
     "                if mode == 'swap':                  # the seed's values; not scored by the seed: published, as it is\n"
     "                    e, e1 = s[1], s[0]\n"
     "                elif zm is None:\n"
     "                    e, e1 = s.mean(0), s[0]\n"),
    ("                cf = {c: vo[c][:, idx].mean(0) for c in ('neural_cos', 'neural_fp')}\n",
     "                cf = {c: vo[c][1, idx] if mode == 'swap' else vo[c][:, idx].mean(0)\n"
     "                      for c in ('neural_cos', 'neural_fp')}\n"),
    ("            nmo[idx[full]] = K\n"
     "            nmo[idx[alone]] = 1\n",
     "            nmo[idx[full]] = 1 if mode == 'swap' else K\n"
     "            nmo[idx[alone]] = 0 if mode == 'swap' else 1\n"),
    ("        assert mode in ('raw', 'z'), f'SENS_MODE {mode!r}'\n",
     "        assert mode in ('raw', 'z', 'swap'), f'SENS_MODE {mode!r}'\n"
     "        assert mode != 'swap' or (K == 2 and len(SENS_SEEDS) == 1), \\\n"
     "            f'SENS_MODE swap replaces the published model by ONE extra seed: SENS 2, one seed ({K}, {SENS_SEEDS!r})'\n"),
    ("        print('SENS seeds | vs published on the ensembled lists | ' + ' | '.join(\n",
     "        print(f\"SENS seeds | vs published on the {'swapped' if mode == 'swap' else 'ensembled'} lists | \" + ' | '.join(\n")]


def _note(k, mode, seeds, base='pubm'):
    """The SENS note of markdown cell 0 ('pubm': casmi26-pubm-sens's text; the other bases add the composition)."""
    tail = {'pubm': '',
            'eng3': ("> SENS composes with ENG3: the merge acts inside the frozen pipeline (frozen cells 32 / 39), before its\n"
                     "> evidence export; the published scorer and SENS then score the exported final lists and outside candidates\n"
                     "> (cell 13: run_public_pipeline, score_final_lists, score_outside_candidates, then SENS).\n"),
            'elig1': ("> SENS composes with elig1: our Engine 2 acts inside the frozen pipeline (frozen cells 32 / 39 / 45), before\n"
                      "> its evidence export; the published scorer and SENS then score the exported final lists and outside\n"
                      "> candidates (cell 13: run_public_pipeline, score_final_lists, score_outside_candidates, then SENS).\n")}[base]
    if mode == 'swap':
        s = seeds[0]
        return (f"> **SENS swap (this build: scripts/make_pub_variant.py SENS=2 SENS_MODE=swap SENS_SEEDS=[{s}]).** A placebo\n"
                "> of the neural term: the published saner model is replaced by ONE refit of its own recipe, seed\n"
                f"> {s} (final_model_seed{s}.pt, output of our kernel bryantyu/casmi26-saner-seeds: the published\n"
                "> train.py refit with another seed, sha256 checked against its final_models_sha256.json); nothing is averaged.\n"
                "> The pinned neural_scores.py scores the final lists and the outside candidates once more with the refit (the\n"
                "> same inputs and command line as the published scorer, the same model code and config); neural_score /\n"
                "> neural_cos / neural_fp := the refit's values for every molecule whose rows the published model and the refit\n"
                "> both scored (processed, neural_ok, finite), and for its outside candidates that both scored; the published\n"
                "> values otherwise and on any failure (kept as neural_score_published, neural_cos_published and\n"
                "> neural_fp_published). Everything after it (consensus, fragment check, final layer) is unchanged. PF-syn\n"
                "> re-weighting (OURS, PF_MID) and ISO are off (OURS = False, ISO_ALPHA = 0), as in "
                f"{'casmi26-pube1' if base == 'elig1' else 'casmi26-pubm'}.\n"
                "> Report: sens_report.json and the SENS lines.\n" + tail + ">\n")
    return (f"> **SENS (this build: scripts/make_pub_variant.py SENS={k} SENS_MODE={mode}).** The neural term is a seed\n"
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
            f"> off (OURS = False, ISO_ALPHA = 0), as in {'casmi26-pube1' if base == 'elig1' else 'casmi26-pubm'}. Report: "
            "sens_report.json and the SENS lines.\n" + tail + ">\n")


THIS_VERSION = '> **This version** (made by scripts/make_pub_variant.py; every setting is in cell 1)'
ENG3_TITLE = '# Variant M core + midx final layer + ENG3: Engine 2 merged with our stage 1 (w {!r})\n\n'
ELIG1_TITLE = ('# Eligible fork elig1 · Variant M core + midx final layer, our stage 1 as Engine 2, upstream datasets only\n')


def _elig1_statement(k, mode, seeds):
    """Splices of elig.md0_elig1(False) (the eligibility statement) for elig1 + SENS (mode 'swap': _elig1_swap)."""
    if mode == 'swap':
        return _elig1_swap(seeds)
    sl = ', '.join(map(str, seeds))
    bullet = (
        "> - **SENS, the saner seed ensemble (not one of the fixes; cell 12).** The neural term averages the published\n"
        f">   final_model.pt (seed {PUBLISHED_SEED}, sanpier/casmi-spectrum-model-v1) and our refits of seeds {sl}\n"
        ">   (final_model_seed<s>.pt) from the output of our kernel bryantyu/casmi26-saner-seeds, attached as a kernel\n"
        ">   source (kernel-metadata.json kernel_sources, not a dataset); the SENS note below gives the rule. The refits have\n"
        ">   the eligibility status of the published final_model.pt: casmi26-saner-seeds ran sanpier's published train.py /\n"
        ">   prepare.py / model.py / features.py (Apache-2.0) with the published config.json and only the seed changed (plus\n"
        ">   F7's one-line normalized_smiles alias of prepare.py), on train.parquet only: it mounts\n"
        ">   sanpier/casmi-spectrum-model-v1, metric/rdkit-2026-3-3-wheel and the competition data, nothing else. F7\n"
        ">   retrained the published model bit-exactly with the same code (kernels/casmi26-saner-repro), and\n"
        ">   casmi26-saner-seeds' own refit of the published seed is bit-exact as well (so is its seed-20262023 holdout\n"
        ">   model against F7's run of that seed). Cell 12 takes the refits only from the folder where Kaggle mounts that\n"
        ">   kernel output (/kaggle/input/notebooks/bryantyu/casmi26-saner-seeds first; /kaggle/input/casmi26-saner-seeds\n"
        ">   in the older flat layout), not from any other mount, and checks each file's SHA-256 against its\n"
        ">   final_models_sha256.json; any failure keeps the published scores.\n")
    return [
        (ELIG1_TITLE, '# Eligible fork elig1 + SENS · Variant M core + midx final layer, our stage 1 as Engine 2, saner seed '
                      f'ensemble (K {k}, {mode}), upstream datasets only\n'),
        ("> **Eligibility build elig1** (scripts/make_pub_variant.py with ELIG=1; the audit",
         f"> **Eligibility build elig1 + SENS** (scripts/make_pub_variant.py with ELIG=1 SENS={k} SENS_MODE={mode}; the audit"),
        ("elig0 (casmi26-pube0) with fix F2b in place of F2a. Changed against\n> casmi26-pubm:",
         "elig0 (casmi26-pube0) with fix F2b in place of F2a, plus the saner seed\n"
         "> ensemble SENS (not a fix; see below). Changed against casmi26-pubm:"),
        ("with OURS = False, PF-syn (cell 11) and ISO (cell 17) are off.",
         "with OURS = False, PF-syn (cell 11) and ISO (cell 18) are off."),
        ("reports any mount that kernel-metadata.json does not list (EXTRA MOUNT)",
         "reports any mount that kernel-metadata.json does not list (EXTRA MOUNT; the output of its kernel source\n"
         ">   bryantyu/casmi26-saner-seeds counts as listed, and the ELIG line says whether it is mounted)"),
        ("> - Sections 6 and 8 name the datasets attached here.", bullet + "> - Sections 6 and 8 name the datasets "
                                                                  "(and the kernel output) attached here."),
        ("Not part of the fixes: the ISO cell (cell 17, off with", "Not part of the fixes: the ISO cell (cell 18, off with"),
        ("casmi26-pube1 is a new kernel with a score of its own.",
         "casmi26-pube1 and this SENS build of it are new\n>   kernels, each with a score of its own."),
        ("> public code (Engine 1's ranker_v4b and fe_v4 tables, saner, our own G4 / G5:",     # saner: bit-exact (F7)
         "> public code (Engine 1's ranker_v4b and fe_v4 tables, our own G4 / G5:"),
        ("count as using NIST? F7, a demonstrated retrain of saner's spectrum\n> model. F8, the licence",
         "count as using NIST? F8, the licence"),
        ("training-structure rows under the competition terms (a CC BY label conflicts with that).\n",
         "training-structure rows under the competition terms (a CC BY label conflicts with that). With SENS, F8 also\n"
         "> covers the kernel bryantyu/casmi26-saner-seeds (pushed as a private kernel): the attribution and licence of its\n"
         "> output (sanpier's Apache-2.0 training code; seed refits trained on the competition data, CC BY-NC 4.0).\n"
         ">\n"
         "> **Closed since casmi26-pube1:** F7, the saner retrain (Oct 31, kernels/casmi26-saner-repro: sanpier's published\n"
         "> train.py / prepare.py, with the one-line normalized_smiles alias, retrained the published final_model.pt\n"
         "> bit-exactly from train.parquet). The SENS refits come from the same code, config and data with other seeds.\n")]


def _elig1_swap(seeds):
    """_elig1_statement for SENS_MODE 'swap' (one seed): the same anchors, the swap's wording."""
    assert len(seeds) == 1, seeds
    s = seeds[0]
    bullet = (
        "> - **SENS swap, the saner seed-swap placebo (not one of the fixes; cell 12).** The neural term is that of our refit\n"
        f">   of the published saner recipe with seed {s} (final_model_seed{s}.pt) from the output of our kernel\n"
        ">   bryantyu/casmi26-saner-seeds, attached as a kernel source (kernel-metadata.json kernel_sources, not a dataset),\n"
        f">   in place of the published final_model.pt (seed {PUBLISHED_SEED}, sanpier/casmi-spectrum-model-v1, which still\n"
        ">   scores first: its values stay in the audit columns and wherever the refit did not score); the SENS note below\n"
        ">   gives the rule. The refit has the eligibility status of the published final_model.pt: casmi26-saner-seeds ran\n"
        ">   sanpier's published train.py / prepare.py / model.py / features.py (Apache-2.0) with the published config.json\n"
        ">   and only the seed changed (plus F7's one-line normalized_smiles alias of prepare.py), on train.parquet only: it\n"
        ">   mounts sanpier/casmi-spectrum-model-v1, metric/rdkit-2026-3-3-wheel and the competition data, nothing else. F7\n"
        ">   retrained the published model bit-exactly with the same code (kernels/casmi26-saner-repro), and\n"
        ">   casmi26-saner-seeds' own refit of the published seed is bit-exact as well (so is its seed-20262023 holdout\n"
        ">   model against F7's run of that seed). Cell 12 takes the refit only from the folder where Kaggle mounts that\n"
        ">   kernel output (/kaggle/input/notebooks/bryantyu/casmi26-saner-seeds first; /kaggle/input/casmi26-saner-seeds\n"
        ">   in the older flat layout), not from any other mount, and checks the file's SHA-256 against its\n"
        ">   final_models_sha256.json; any failure keeps the published scores.\n")
    swap = {
        ELIG1_TITLE: ('# Eligible fork elig1 + SENS swap · Variant M core + midx final layer, our stage 1 as Engine 2, saner '
                      f'seed swap (seed {s} in place of the published model: placebo), upstream datasets only\n'),
        "> **Eligibility build elig1** (scripts/make_pub_variant.py with ELIG=1; the audit":
            f"> **Eligibility build elig1 + SENS swap** (scripts/make_pub_variant.py with ELIG=1 SENS=2 SENS_MODE=swap "
            f"SENS_SEEDS=[{s}]; the audit",
        "elig0 (casmi26-pube0) with fix F2b in place of F2a. Changed against\n> casmi26-pubm:":
            "elig0 (casmi26-pube0) with fix F2b in place of F2a, plus the saner seed\n"
            "> swap SENS (a placebo, not a fix; see below). Changed against casmi26-pubm:",
        "> - Sections 6 and 8 name the datasets attached here.":
            bullet + "> - Sections 6 and 8 name the datasets (and the kernel output) attached here.",
        "casmi26-pube1 is a new kernel with a score of its own.":
            "casmi26-pube1 and this SENS swap build of it are new\n>   kernels, each with a score of its own.",
        "training-structure rows under the competition terms (a CC BY label conflicts with that).\n":
            "training-structure rows under the competition terms (a CC BY label conflicts with that). With SENS, F8 also\n"
            "> covers the kernel bryantyu/casmi26-saner-seeds (pushed as a private kernel): the attribution and licence of its\n"
            "> output (sanpier's Apache-2.0 training code; seed refits trained on the competition data, CC BY-NC 4.0).\n"
            ">\n"
            "> **Closed since casmi26-pube1:** F7, the saner retrain (Oct 31, kernels/casmi26-saner-repro: sanpier's published\n"
            "> train.py / prepare.py, with the one-line normalized_smiles alias, retrained the published final_model.pt\n"
            f"> bit-exactly from train.parquet). The SENS swap refit (seed {s}) comes from the same code, config and data\n"
            "> with another seed.\n"}
    edits = _elig1_statement(2, 'raw', seeds)            # the anchors (and the edits swap shares with the ensemble)
    assert set(swap) <= {old for old, _ in edits}, sorted(set(swap) - {old for old, _ in edits})
    return [(old, swap.get(old, new)) for old, new in edits]

def _title(k, mode, seeds):
    """What the SENS build adds to the cell-0 title (bases pubm / eng3)."""
    return (f'SENS swap: the published saner model replaced by seed {seeds[0]} (placebo)' if mode == 'swap' else
            f'SENS: saner seed ensemble (K {k}, {mode})')


def md0(text, k, mode, seeds, base='pubm', w=None, elig_statement=None):
    """Markdown cell 0 of the SENS build on `base`: 'pubm' (text = the generator's cell 0), 'eng3' (text = eng3.md0's,
    w its weight) or 'elig1' (text = elig.md0_elig1(False) + the generator's cell 0 body; elig_statement = that
    statement)."""
    if base == 'pubm':
        assert text.startswith(TITLE0), text[:80]
        return (f"# Variant M core + midx final layer + {_title(k, mode, seeds)}\n\n"
                + _note(k, mode, seeds) + splice(text[len(TITLE0):], RENUMBER0, 'cell 0 (sens: cell numbers)'))
    assert text.count(THIS_VERSION) == 1, 'cell 0 layout'
    i = text.index(THIS_VERSION)
    rest = splice(text[i:], RENUMBER0, f'cell 0 (sens on {base}: cell numbers)')
    if base == 'eng3':
        title = ENG3_TITLE.format(float(w))
        assert text.startswith(title), text[:100]
        note3 = text[len(title):i]
        assert note3.startswith('> **ENG3 (this build') and note3.endswith('\n>\n'), note3[-80:]
        return (title[:-2] + f' + {_title(k, mode, seeds)}\n\n' + note3
                + _note(k, mode, seeds, 'eng3') + rest)
    assert base == 'elig1' and elig_statement is not None and text[:i] == elig_statement, 'cell 0: the elig1 statement'
    return (splice(elig_statement, _elig1_statement(k, mode, seeds), 'cell 0 (elig1 statement, sens)')
            + _note(k, mode, seeds, 'elig1') + rest)


def ours_meta(ours):
    """casmi.ours final_layer / iso of the generator with the SENS build's cell numbers (17 / 18)."""
    return dict(final_layer=splice(ours['final_layer'], [('(cell 16)', '(cell 17)')], 'casmi.ours.final_layer'),
                iso=splice(ours['iso'], [('(cell 17)', '(cell 18)')], 'casmi.ours.iso'))


def splice(text, edits, what):
    for old, new in edits:
        assert text.count(old) == 1, f'{what}: {old[:70]!r} found {text.count(old)} times'
        text = text.replace(old, new)
    return text


# ---- elig1 + SENS (eligible build): the checkpoints come only from the kernel output; sections 5 / 6 / 8 ----
SOURCE_ELIG1 = [(
    '    """Folder of the bryantyu/casmi26-saner-seeds output: the one holding final_models_sha256.json (paths containing\n'
    '    casmi26-saner-seeds first, then the shortest)."""\n'
    "    hits = sorted(glob.glob(str(INPUT / '**' / 'final_models_sha256.json'), recursive=True),\n"
    "                  key=lambda p: (SENS_SOURCE not in p, len(p), p))\n"
    '    if not hits:\n'
    "        raise FileNotFoundError('final_models_sha256.json not under /kaggle/input: attach the output of '\n"
    "                                'bryantyu/casmi26-saner-seeds (kernel_sources)')\n",
    '    """Folder of the bryantyu/casmi26-saner-seeds output: the one holding final_models_sha256.json (paths containing\n'
    '    casmi26-saner-seeds first, then the shortest). elig1 (eligible build): only inside that kernel output, mounted at\n'
    '    /kaggle/input/notebooks/bryantyu/casmi26-saner-seeds (first, as the ELIG line of cell 8) or, in the older flat\n'
    '    layout, /kaggle/input/casmi26-saner-seeds; never another mount."""\n'
    "    hits = sorted(glob.glob(str(INPUT / '**' / 'final_models_sha256.json'), recursive=True),\n"
    "                  key=lambda p: (SENS_SOURCE not in p, len(p), p))\n"
    "    roots = (INPUT / 'notebooks' / 'bryantyu' / SENS_SOURCE, INPUT / SENS_SOURCE)   # elig1: the kernel output only\n"
    '    hits = [p for r in roots for p in hits if Path(p).is_relative_to(r)]          # owner-qualified mount first\n'
    '    if not hits:\n'
    "        raise FileNotFoundError('final_models_sha256.json not in the kernel output bryantyu/casmi26-saner-seeds '\n"
    "                                '(elig1: /kaggle/input/notebooks/bryantyu/casmi26-saner-seeds or '\n"
    "                                '/kaggle/input/casmi26-saner-seeds): attach it (kernel_sources)')\n")]
MD5_ELIG1 = [("casmi26-pube1 is a new\nkernel (the eligible fork elig1):",
              "This SENS build of casmi26-pube1 is a new\nkernel (the eligible fork elig1 with the saner seed ensemble):")]
MD5_ELIG1_SWAP = [("casmi26-pube1 is a new\nkernel (the eligible fork elig1):",
                   "This SENS swap build of casmi26-pube1 is a new\nkernel (the eligible fork elig1 with one seed refit "
                   "in place of the published saner model):")]
MD6_ELIG1 = [("Copy & Edit this notebook (the competition data and the datasets\nattached to this build, listed in section 8, "
              "stay attached).",
              "Copy & Edit this notebook (the competition data, the datasets\nand the kernel output attached to this build, "
              "listed in section 8, stay attached).")]
CELL8_ELIG1 = [("# Runs before the pipeline (the stage functions are defined in the next cell and called in cell 12):\n",
                "# Runs before the pipeline (the stage functions are defined in the next cell and called in cell 13):\n")]
CREDITS_LINEAGE = '\n**Upstream lineage, not attached here.**'
CREDITS_OURS_TAIL = '(our stage 1, casmi26-elig-assets).'


def credits_elig1(seeds, mode='raw'):
    """Section 8 splices: the kernel output's table (before the upstream lineage) and our part of the work."""
    swap = mode == 'swap'
    use = (f"SENS swap: final_model_seed{seeds[0]}.pt, the refit of the published saner recipe with seed {seeds[0]} "
           "(published config, another seed) whose scores replace the published model's, and final_models_sha256.json "
           "(its SHA-256)" if swap else
           "SENS: final_model_seed<s>.pt, refits of the published saner recipe "
           f"with seeds {', '.join(map(str, seeds))} (published config, another seed), and final_models_sha256.json "
           "(their SHA-256)")
    table = ("\n**Kernel output attached to this build** (kernel-metadata.json kernel_sources; read only by the SENS cell, "
             "cell 12):\n\n"
             "| kernel | author | used for | licence |\n|---|---|---|---|\n"
             f"| [{SOURCE_KERNEL}](https://www.kaggle.com/code/{SOURCE_KERNEL}) | us; training code by Saner Turhaner "
             f"(sanpier/casmi-spectrum-model-v1) | {use} | code Apache-2.0 (sanpier's published train.py / prepare.py / "
             "model.py / features.py, run "
             "with F7's one-line normalized_smiles alias); weights trained on train.parquet only (competition data, "
             "CC BY-NC 4.0); kernel pushed private, its licence and attribution are part of F8 |\n")
    ours = (" With SENS swap we also wrote the seed-ensemble cell (cell 12, here in its swap mode)\nand trained its "
            "seed refit (casmi26-saner-seeds) with Saner Turhaner's published code." if swap else
            " With SENS we also wrote the seed-ensemble cell (cell 12) and\n"
            "trained its seed refits (casmi26-saner-seeds) with Saner Turhaner's published code.")
    return [(CREDITS_LINEAGE, table + CREDITS_LINEAGE), (CREDITS_OURS_TAIL, CREDITS_OURS_TAIL + ours)]


def _load_elig():
    import importlib.util
    spec = importlib.util.spec_from_file_location('pub_variant_elig_for_sens', os.path.join(HERE, 'elig.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)            # our generator code (constants and helpers), not notebook code
    return mod


SWAP_RULE = ("swap (seed-swap placebo): neural_score / neural_cos / neural_fp := the one extra seed's own values, nothing "
             "averaged, for molecules with every row processed, neural_ok and finite for the published model and the seed; "
             "their outside candidates: the seed's values where both are neural_ok and finite; published values otherwise "
             "and on any failure (audit columns neural_*_published, neural_score_seed<s>; sens_members 1 = seed, 0 = "
             "published)")


def base_of(cfg):
    """'pubm', 'eng3' or 'elig1' from the cell-1 configuration (ELIG=0 and ELIG + ENG3 are refused)."""
    elig, w = cfg.get('ELIG'), cfg.get('ENG3_W')
    assert elig in (None, 1) and (w is None or w > 0) and not (elig is not None and w is not None), (elig, w)
    return 'elig1' if elig == 1 else 'eng3' if w is not None else 'pubm'


def apply(nb, meta, k, mode, cfg):
    """Turn the generated pub variant (nb, meta) into the SENS build in place; returns a report (record = the notebook
    metadata casmi.ours.sens). cfg: cell 1 (with ELIG = 1 / ENG3_W: the elig1 / ENG3 build made by elig.py / eng3.py)."""
    assert isinstance(k, int) and 2 <= k <= 1 + len(cfg['SENS_SEEDS']) and mode in MODES, (k, mode)
    assert cfg['SENS'] == k and cfg['SENS_MODE'] == mode, 'cell 1 SENS lines'
    assert check_config(cfg, k) is None, check_config(cfg, k)
    base = base_of(cfg)
    elig = _load_elig() if base == 'elig1' else None
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
    src = cell_source(mode)
    src = src if base != 'elig1' else splice(src, SOURCE_ELIG1, 'sens cell (elig1: kernel output only)')
    cells.insert(i_pf + 1, {'cell_type': 'code', 'execution_count': None, 'id': CELL_ID,
                            'metadata': {'_kg_hide-input': True, 'jupyter': {'source_hidden': True}}, 'outputs': [],
                            'source': src})
    seeds = list(cfg['SENS_SEEDS'][:k - 1])
    stmt = elig.md0_elig1(bool(cfg['OURS'])) if elig else None
    cells[0]['source'] = md0(cells[0]['source'], k, mode, seeds, base, w=cfg.get('ENG3_W'), elig_statement=stmt)
    ids = [c.get('id') for c in cells]
    assert (ids.index(FINAL_ID), ids.index(ISO_ID)) == (17, 18), 'final layer / ISO cells after the insertion'
    md = lambda head: [i for i, c in enumerate(cells) if c['cell_type'] == 'markdown' and c['source'].startswith(head)]
    i4 = md(SEC4)
    assert len(i4) == 1, i4
    i4 = i4[0]
    cells[i4]['source'] = splice(cells[i4]['source'], RENUMBER4, f'cell {i4} (section 4; sens: cell numbers)')
    edited = [0, 1, i_pf + 2, i4]
    assert meta.get('kernel_sources') == [], meta.get('kernel_sources')
    meta['kernel_sources'] = [SOURCE_KERNEL]
    if base == 'elig1':
        i8 = ids.index('3e2e781f')
        assert i8 < i_pf and '# ── Eligibility (ours, elig1)' in cells[i8]['source'], 'cell 8 (elig1 block)'
        cells[i8]['source'] = splice(cells[i8]['source'], elig.cell8_kernel_source_edits(meta['kernel_sources']) +
                                     CELL8_ELIG1, f'cell {i8} (elig1 mount check: kernel sources; driver cell number)')
        (i5,), (i6,), (i8c,) = md('## 5. Every submission'), md('## 6. Run it yourself'), md('## 8. Credits')
        cells[i5]['source'] = splice(cells[i5]['source'], MD5_ELIG1_SWAP if mode == 'swap' else MD5_ELIG1,
                                     f'cell {i5} (section 5, elig1 + sens)')
        cells[i6]['source'] = splice(cells[i6]['source'], MD6_ELIG1, f'cell {i6} (section 6, elig1 + sens)')
        cells[i8c]['source'] = splice(cells[i8c]['source'], credits_elig1(seeds, mode), f'cell {i8c} (credits, elig1 + sens)')
        edited += [i8, i5, i6, i8c]
        e = nb['metadata']['casmi']['elig']
        assert e['level'] == 1 and e['datasets'] == elig.UPSTREAM + elig.ELIG1_DATASETS, e['datasets']
        e['notebook_cells_edited'] = sorted(i + (i > i_pf) for i in e['notebook_cells_edited'])   # the SENS cell at 12
        e['kernel_sources'] = list(meta['kernel_sources'])
        assert [x[:2] for x in e['open']] == ['F5', 'F6', 'F7', 'F8'], e['open']
        e['open'] = [x + ('; SENS: the kernel bryantyu/casmi26-saner-seeds (pushed private), attribution and licence of '
                          'its output' if x.startswith('F8') else '') for x in e['open'] if not x.startswith('F7')]
        e['closed'] = ['F7 saner retrain demonstration (Oct 31, kernels/casmi26-saner-repro: the published final_model.pt '
                       'retrained bit-exactly from sanpier\'s published code and train.parquet); the SENS seed refits '
                       '(casmi26-saner-seeds) come from the same code, config and data with other seeds']
    record = dict(
        k=k, mode=mode, published_seed=PUBLISHED_SEED, extra_seeds=seeds, source=f'{SOURCE_KERNEL} (kernel output: '
        'final_model_seed<s>.pt, final_models_sha256.json)', cell='scripts/pub_variant/sens_cell.py'
        + (' + SOURCE_SWAP' if mode == 'swap' else '') + f' (id {CELL_ID}, cell {i_pf + 1})', call=f'driver cell {i_pf + 2} (id {DRIVER_ID}): after ours_pf_blend, before build_consensus',
        rule=SWAP_RULE if mode == 'swap' else
        ("raw: neural_score := mean of the members' neural_score" if mode == 'raw' else
         'z: neural_score := mean of the members\' within-molecule z (listed candidates; outsiders the same maps)')
        + '; neural_cos / neural_fp := member means; molecules with every row processed, neural_ok and finite for every '
        'member; published values otherwise and on any failure', max_seconds=cfg['SENS_MAX_SECONDS'])
    if base != 'pubm':
        record.update(base=base, composition=(
            'ENG3 (frozen cells 32 / 39)' if base == 'eng3' else 'elig1 Engine 2 (frozen cells 32 / 39 / 45)')
            + ' acts inside the frozen pipeline before its evidence export; SENS acts on the neural files scored from '
            'that export (driver cell, after score_final_lists / score_outside_candidates)')
    if base == 'elig1':
        record.update(cell=record['cell'].replace(' (id ', ' + SOURCE_ELIG1 (id ', 1),
                      source_rule='only the kernel output: /kaggle/input/notebooks/bryantyu/casmi26-saner-seeds (first) or '
                                  '/kaggle/input/casmi26-saner-seeds (flat layout) (sens_source_dir, SOURCE_ELIG1)',
                      eligibility='seed refits of the published saner recipe (sanpier, Apache-2.0) trained by us on '
                                  'train.parquet only; status of the published final_model.pt (F7 closed)')
    return dict(record=record, cell_index=i_pf + 1, driver_index=i_pf + 2, cells_edited=sorted(edited),
                cell_inserted=i_pf + 1, base=base)
