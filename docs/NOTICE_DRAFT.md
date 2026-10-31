# DRAFT: NOTICE for the eligible fork notebooks casmi26-pube0 and casmi26-pube1

> **DRAFT, not in force.** This file waits for the owner's LICENSE decision (docs/reproduce.md G19,
> docs/fork_eligibility.md F8). It is not a licence grant, and no LICENSE file is added to the repository (owner's
> decision). Once the owner chooses a licence, the text between the two rules below becomes the NOTICE that ships with
> the published fork notebook, together with the full Apache License 2.0 text. Licence statements come from copies
> saved during the Oct 31 audit and must be re-checked on the live pages before shipping. Datasets, data sources and
> their notices: [`fork_ATTRIBUTION_DRAFT.md`](fork_ATTRIBUTION_DRAFT.md).

---

NOTICE

casmi26-pube0 and casmi26-pube1 (Kaggle user bryantyu): eligible forks of the public CASMI 2026 notebook lineage
Spectral Parallax -> Variant M -> kunaldesale2408/enveda-casmi-2026-mid, with the midx final layer.

These notebooks contain third-party code under the Apache License, Version 2.0 (part 1) and run third-party code under
the MIT License from mounted Kaggle datasets (part 2). Our own code is described in part 4. The datasets and data the
notebooks read keep their own terms (fork_ATTRIBUTION_DRAFT.md); this NOTICE does not change them.

## Part 1. Apache License 2.0: code embedded in the notebooks

Copies of public Kaggle notebook code. Apache-2.0 is Kaggle's licence for public notebooks, and the embedded 0.465
notebook declares it itself. Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0 (the full text ships with
the notebook). "Changed" lists our modifications (Apache-2.0 §4(b)). All of them were made by scripts/make_pub_variant.py
of the bryantyu repository, and each patched cell is marked "ours" in a comment.

1. **Notebook body**: configuration, session and stage machinery, run, fragment-check and merge cells, and the text of
   sections 1-7 (cells 0-7, 9, 12, 13, 15 and 18-23).
   - Authors:
     - Roman Tamrazov, Spectral Parallax, version 22 (https://www.kaggle.com/code/romantamrazov/casmi-2026-spectral-parallax);
     - extended to "Variant M" by Kaggle user busyaprime (kernel 33-casmi26-px-m1);
     - published as kunaldesale2408/enveda-casmi-2026-mid (https://www.kaggle.com/code/kunaldesale2408/enveda-casmi-2026-mid).
   - Spectral Parallax credits Kozykorpesh, Balanced Acquisition Evidence
     (https://www.kaggle.com/code/kozykappa/lb-0-468-balanced-acquisition-evidence), and heliosli
     (https://www.kaggle.com/code/heliosli/happy-to-make-the-dataset-public-if-it-helps).
   - Changed:
     - cell 0 (eligibility notes);
     - cell 1 (switches ELIG, OURS, PF_W, OURS_MODE, PF_MID_W and ISO_*; final-layer weights, SPEC_W 0.0);
     - cell 8 (pinned sources: the patched frozen notebook and the eligibility code of part 4);
     - cells 20, 21 and 23 (text naming this build and its datasets).
   - The figure helpers and field record of cells 4-6 (busyaprime, engine/33_style.py at commit 4ebb73e9) are unchanged.
2. **Helper modules** runner.py, export_evidence.py, neural_scores.py, submission.py, ranking.py and bundle.py (cell 8,
   `MODULES`). Saner Turhaner, PubChem + Neural025 (https://www.kaggle.com/code/sanpier/pubchem-neural025-lb-0-466).
   Unchanged (sha256 pinned in `MODULE_SHA256`).
3. **consensus.py and outsiders.py** (cell 8, `MODULES`). Roman Rozen, From Peaks to Structures
   (https://www.kaggle.com/code/romanrozen/0-468-casmi-from-peaks-to-structures). Unchanged.
4. **The 0.465 pipeline notebook** (cell 8, `FROZEN_B64`; original sha256 d1e757c2b71b46e6e8a5777f14302a4737d0fdbcd56d8365093c039b93c5fd88).
   Dmitrii Gluzdov, The Data Behind Spectra to Structures
   (https://www.kaggle.com/code/dmitriigluzdov/casmi-26-the-data-behind-spectra-to-structures).
   - Changed:
     - casmi26-pube0, sha256 47db922027a44fcc91c96095e05ef743263bef805b86e117cc8ecdb745515e2a, cells 5, 9, 12, 14, 16,
       18, 26, 32, 36, 39, 45 and 54:
       - upstream datasets located instead of the v4g container (5, 26);
       - display cells skipped (9, 12, 14, 16, 18);
       - the second engine replaced by a null engine that returns the pipeline's own lists (32, 39, 45);
       - PubChem depositor-source bits off (26);
       - external reference spectra not added (36);
       - receipt hashes of the mounted datasets (54).
     - casmi26-pube1, sha256 67764f557a65c1b5c612f6384fd5ef68f3bc6a919be115fec56d00b5274773d9: the same cells plus 31,
       with the second engine replaced by our stage 1 (32, 39, 45) and a null-engine fallback.
   - Its section 6 table and its credits cell are kept unchanged. They credit, among others, the authors whose code
     it embeds:
     - canqiang, E006 (https://www.kaggle.com/code/canqiang/casmi26-e006-window-40-ice-fallback-true);
     - obstacledeveloper, CASMI26 V1.1 Metric Audit and Safe Tail, scriptVersionId 356393001
       (https://www.kaggle.com/code/obstacledeveloper/casmi26-v1-1-metric-audit-and-safe-tail);
     - bobthebot369, Enveda CASMI 2026 V18 Sovereign Zenith (https://www.kaggle.com/code/bobthebot369/enveda-casmi-2026-v18-sovereign-zenith);
     - Ahmed Berat Özer:
       - v4n (https://www.kaggle.com/code/ahmedberatozer/casmi26-v4n-inference);
       - v4m (https://www.kaggle.com/code/ahmedberatozer/casmi26-v4m-inference);
       - v7c (https://www.kaggle.com/code/ahmedberatozer/casmi26-v7c-e006-fraggap);
     - prvsiyan (https://www.kaggle.com/code/prvsiyan/analog-propagation-casmi-2026-baseline);
     - megayak (https://www.kaggle.com/code/megayak/casmi26-two-rankers-one-engine);
     - seyitkaangunes:
       - engine fusion (https://www.kaggle.com/code/seyitkaangunes/casmi26-v4n-engine-fusion-union-lb-0-399);
       - PubChem promotion threshold (https://www.kaggle.com/code/seyitkaangunes/casmi26-e006-pubchem-promo-rel-150-lb-0-446);
       - the N1A GLACIER proxies, glb_core.py and glb_driver.py, embedded verbatim with sha256 pins
         (https://www.kaggle.com/code/seyitkaangunes/casmi26-e006-n1a-cc-by-chebi-lb-0-452);
     - flexonafft:
       - N1 challenger, version 47 (https://www.kaggle.com/code/flexonafft/casmi26-closing-the-evidence-gap?scriptVersionId=360486019);
       - baseline adaptation (https://www.kaggle.com/code/flexonafft/casmi26-public-baseline-adaptation-experiments);
     - wangpenghua:
       - v21 (https://www.kaggle.com/code/wangpenghua/casmi26-v21-v4i-pop);
       - v29 (https://www.kaggle.com/code/wangpenghua/casmi26-v29-fusion-pop-glmh);
     - antriksharya, rank-2 placement (https://www.kaggle.com/code/antriksharya/casmi26-lb-0-443-v7c-engine-2-rank-2-insert);
     - guhongbin:
       - v465 (https://www.kaggle.com/code/guhongbin/casmi26-v465-v15-pop015);
       - v447 (https://www.kaggle.com/code/guhongbin/casmi26-v447-pubchemlite);
     - ghazarosbarseghyan91 (https://www.kaggle.com/code/ghazarosbarseghyan91/enveda-v7c-e006-fraggap-evgen-v14seeds),
       building on evgendvorkin, version 14 (https://www.kaggle.com/code/evgendvorkin/enveda-casmi-2026);
     - samanyu1808 (https://www.kaggle.com/code/samanyu1808/casmi26-iceberg-where-glacier-is-blind-lb-0-438);
     - the Kaggle competition metric, version 14 (https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank;
       Apache-2.0).
   - The second-engine code those credits also describe (two-ranker engine, PubChemLite block) lives in the dataset the
     forks no longer mount. It is not part of these notebooks.
5. **frag_rescore.py**, an untrained MetFrag-lite fragment scorer (cell 10 `_physical`, sha256 checked, and frozen
   notebook cell 51). Huseyin Emre Aksoy (https://www.kaggle.com/huseyinemreaksoy), reached through Ahmed Berat Özer's
   v7c notebook. Unchanged.
6. **parallax_physical.py** (cell 10 `_post`) and the **provenance-aware reranker** (cells 14-15). Roman Tamrazov,
   Spectral Parallax v22. Unchanged; the reranker is switched off (PROVENANCE = False).
7. **The final layer** (cell 16). Cell 15 of baidalinadilzhan/casmi26-midx-exact-dedup-fill
   (https://www.kaggle.com/code/baidalinadilzhan/casmi26-midx-exact-dedup-fill). Changed:
   - its configuration lines were moved to cell 1;
   - `_mid_neural_score` was added (PF_MID_W; 0.0 in these builds, which reproduces the original order);
   - SPEC_W was set to 0.0, so the spectral-library depositor flag column is not loaded.

## Part 2. MIT License: third-party code the notebooks run

The notebooks import this code from mounted Kaggle datasets and do not copy it, with one exception: the PubChem-channel
code inside the frozen notebook (part 1, item 4), which docs/fork_eligibility.md A26 rates as notebook code plus code of
the MIT `casmi` package. The copyright lines are listed here so that they travel with the notebook in either case.

1. **`casmi` package** (engine, fe_v4 feature families, FPNet, ranker, chemistry and scoring modules), from
   ahmedberatozer/casmi26-v4b-models and casmi26-v3-models. Copyright (c) 2026 Ahmed Berat Özer. MIT (dataset cards,
   licence section updated 2026-10-13).
2. **DreaMS re-implementation** `code/casmi/dreams_model.py` (ahmedberatozer/casmi26-v4b-models). Copyright (c) 2024
   Roman Bushuiev. MIT (LICENSE-NOTICE.txt of that dataset). The DreaMS weights behind fe_D.pt are MIT / CC BY 4.0.
3. **ms-pred inference subset** @708148c2a8eb (ahmedberatozer/casmi26-iceberg and casmi26-glacier, `src/ms_pred/`).
   Copyright (c) 2023 Samuel Goldman. MIT (`src/ms_pred/LICENSE`). The ICEBERG and GLACIER MassSpecGym checkpoints of
   the ms-pred authors are used under the same licence.
4. **einops 0.8.2** (wheel in ahmedberatozer/casmi26-iceberg, unmodified). Copyright (c) 2018 Alex Rogozhnikov. MIT.
5. **LinSATNet** (Wang et al., ICML 2023; MIT, Copyright (c) 2023 Thinklab@SJTU). Not copied:
   ahmedberatozer/casmi26-glacier ships an independent re-implementation of `linsat_layer` in `shim/`. Credited only.

Not MIT and not covered by this NOTICE:
- The runner glue ice_runner.py, fuse.py, gl_runner.py, gl_fuse.py and shim/ of ahmedberatozer/casmi26-iceberg and
  casmi26-glacier states no licence (fix F5a open).
- The Apache-2.0 code run from mounts, kept here for completeness:
  - Saner Turhaner's model.py and features.py (sanpier/casmi-spectrum-model-v1, Apache-2.0);
  - the training scripts of dmitriigluzdov/casmi26-fold-safe-fpnet (Apache-2.0; not run);
  - pytorch_lightning 2.6.6, torchmetrics 1.9.0 and lightning_utilities 0.15.3 (Apache-2.0 wheels in
    ahmedberatozer/casmi26-iceberg).
- RDKit (BSD-3-Clause) comes from metric/rdkit-2026-3-3-wheel and the iceberg wheels.

## Part 3. Licence texts

The Apache License 2.0 text ships as a separate file: https://www.apache.org/licenses/LICENSE-2.0.txt.

The MIT License, for the copyright holders in part 2 (items 1-5):

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

## Part 4. Our code in these notebooks

PENDING THE OWNER'S LICENSE DECISION (MIT planned: the competition requires winners to license the solution under MIT).

Our code in these notebooks:
- cell 11, the PF-syn scorer, off in these builds;
- cell 17, the ISO isomer reranker, off;
- the eligibility code of cell 8: mount and hash check, null_engine.py, and eng2_merge.py in pube1 (the module headers
  say "Ours, MIT");
- the patches listed in part 1;
- in pube1, the `casmi` modules of bryantyu/casmi26-elig-assets (LICENSE.txt prepared as MIT, "Copyright (c) 2026
  bryantyu").

Apache-2.0 §4 allows the combined notebook to carry other terms for our contributions, as long as part 1's conditions
are met (fork_eligibility.md F8).

---

## Owner checklist (not part of the NOTICE)
- Choose the LICENSE, then fill in part 4 with the final wording and copyright line.
- Ship this NOTICE and the Apache-2.0 text with the published fork. A Kaggle notebook is a single file, so append them
  as a final markdown cell or attach them as a dataset.
- Re-check every licence statement in parts 1-2 on the live pages. These were not re-checked on 2026-10-31; the
  Kaggle-terms basis for "Apache-2.0 for public notebooks" was not opened either.
- Part 2 is not the end state until F5a is answered: either the author's grant (add its notice here) or our own runner
  (fix F5b).
