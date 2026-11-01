# Public notebook / LB refresh, Nov 1 2026 07:00 UTC

## Survey

No public notebook gives a reproducible gain over our 0.479. The only one showing more is heliosli's at 0.480, and it depends on a private dataset that was still private at 06:50 UTC. Every new technique that has an LB reading came in at or below 0.479. All reads are dated Nov 1, 06:43–06:55 UTC.

**How I got the scores.** The kaggle CLI listings don't return notebook scores. I took `bestPublicScore` from Kaggle's anonymous web endpoint `kernels.KernelsService/ListKernels` (competitionId 143318). That gave 292 notebooks: the top 200 by score and the 200 most recently run. One caveat: the field name suggests it may be the best score across all versions of a notebook, not the current version. That matters for heliosli.

## Leaderboard now (4,493 teams)
- **Our position:** Bryant Yu is 0.479 at rank 256. There are 230 teams at 0.479, spanning ranks 45–274; ties are ordered by submission time, which is why we sit at 256 rather than ~45. Being 0.001 higher would put us around rank 32–44.
- **Top 50:** 1 savchenko 0.521 (Oct 31 16:58), 2 parfyon 0.519, 3 w1326627677 0.512, 4 jzhpfn 0.504, 5 shehabalshehabi 0.502, 6 lowasitgoes 0.495, 7–8 team051 and pawebiernatingenix 0.490, 9 aypyaypy 0.489, 10 heliosli 0.486, 11–14 at 0.484, 15–17 at 0.483, 18–26 at 0.482, 27–31 at 0.481, 32–44 at 0.480, 45 onward at 0.479. The full table with last-submission times is in the LB csv listed under Files.
- **Teams that moved above 0.479 since Oct 31 05:08:**
  - savchenko 0.517 → 0.521
  - parfyon 0.517 → 0.519
  - w1326627677 (democatXamer) 0.484 → 0.512, last submission Oct 31 12:49
  - shehabalshehabi 0.501 → 0.502
  - rentoyabuki 0.472 → 0.482; siruizeng 0.459 → 0.482; mikhailyurasov 0.481 → 0.482; musaturfarazi 0.480 → 0.482
  - mrooto and uki706 0.479 → 0.481
  - jojojablon 0.479 → 0.480; solenodon, sheng1207x, soyaoki, mitsuhiroiizuka, victorriposati and diamayer 0.478 → 0.480
  - None of these 18 users has a public notebook in this competition (checked per user; the CLI returns "Not found", and I confirmed that it does list heliosli's notebook).
- **Rerun noise:** alexchilton's notebook says identical code scores about ±0.002 from run to run, because time guards skip optional stages. Our own notes say up to 0.006. That fits the cluster of 0.478 → 0.480 moves, so some of them may be lucky reruns. Any gain like that is on the public LB only.

## Notebooks scoring ≥ 0.479, or run since Oct 31 05:00
Timing check: each page score should be at or below the author's team best, with a submission after the run. Every case below is consistent.

| Displayed score | Notebook (run time, UTC) | What it is | Timing |
|---|---|---|---|
| **0.480** | heliosli/happy-to-make-the-dataset-public-if-it-helps (Oct 31 14:29) | Learned same-formula reranker ("rr3": a model over peaks matched to each candidate's 1–2-bond-cut fragments) on top of the **0.468** base. Training-set structures keep their slots; lists with a library match ≥ 0.9 are skipped. The latest version went back from a 3-reranker average to one reranker. | Team best 0.486; last submission 15:00, 31 min after the run. Plausible. The reranker sits in a private dataset (blank data source), and alexchilton ran it verbatim and got 0.468. If the score is best-across-versions, the 0.480 may come from the earlier 3-reranker version. |
| 0.479 | flexonafft/casmi26-closing-the-evidence-gap v4 (10:26); takumuhata sp478gap is a byte-identical copy (Nov 1 03:08, not scored yet) | midx-equivalent, plus ranks 21–25 opened to scored outsiders, 40 outsider candidates / 150 identity checks | Team 0.468 → 0.479, submitted 11:39. Plausible. **Flat.** |
| 0.479 | cngnguynhuuz fork-479fch (17:47) and fork-glf (19:53) | midx plus a PubChem same-formula channel (needs `ahmedberatozer/casmi26-pubchem-tier`): top 5 by fragment explanation, or predicted spectra in glf, inserted at rank 8, only where library match < 0.9 | Team 0.477 → 0.479, last submission Nov 1 00:01. **Flat.** |
| 0.479 | ayodejiibrahimlateef midx-moderate-tail / elite-borda20 (11:20 / 11:28) | midx plus at most one guarded swap, using a private dataset | Team 0.478 → 0.479 at 11:32. **Flat.** |
| 0.479 | alexchilton (12:28), jianyicai v122 (07:12), hailonglv (14:36), takumuhata sp477midx (22:02), baidalinadilzhan (Oct 30) | midx verbatim (zero diff lines) | All teams reached 0.479 after these runs. |
| 0.478 | kunaldesale2408 MID v28 (15:58); andrewoscar borrow480 is a copy (23:29) | midx plus BIO_W 0.03 (pathway prior) plus library gate 0.87 → 0.80 | Kunal's team stays 0.478 (submitted 16:25). −0.001, read twice. |
| 0.478 | synthreaper v53 (14:59) and v54 (21:24) | midx plus TRAIN_W 0.035 (bonus for structures in train.parquet), ranking frozen at library match ≥ 0.95, plus a disagreement rescue (v53) or 60/220 deep outsiders (v54) | −0.001, read twice. |
| 0.478 | cngnguynhuuz fork-fch (09:12) | Formula channel on the older m1 base | — |
| 0.477 | d1m4official midx-top3-prior (13:00) | midx with the prior blocked from ranks 2–3 (PROTECT_TOP=2) | **−0.002.** The prior's gain sits in ranks 2–3. |
| 0.477 / 0.477 | cngnguynhuuz fork-wt010; romantamrazov spectral-parallax (09:02) | w_top 0.1 on the m1 base; Parallax update | Flat. |
| 0.476 | takumuhata sp477mid | Kunal's earlier version: w_pop 0.14 plus formula-gated isomer swaps | −0.003 |
| 0.472 | jianyicai v123 (08:21) and v124 (08:46) | midx plus "clean preprocessing" (esterification / formula / isotope repair, extra natural-product set, continuous CE); v124 adds library gate 0.80 | **−0.007** |
| 0.471–0.472 | d1m4official v123-noprior / pop15 / top3-prior / gate80-top3 | Prior variants on the v123 base | Prior on vs off: +0.001 |
| 0.477 / 0.472 / 0.472 / 0.469 | guhongbin v555 / v554 / v553 / v552 frag-min{4,20,15,10}pool | Fragment-check pool-size settings | All ≤ 0.477 |
| 0.470 / 0.467 / 0.467 / 0.465 | youyouwou v138off10, v132negproxy; denpugovkin protected-tail; andrewoscar borrow1640 | Assorted variants | All lower |
| 0.099 | prvsiyan clean-lineage | Plain IDF-cosine retrieval baseline | Irrelevant |

**Run but not scored yet (free reads if their authors submit):**
- guhongbin v557: prior off (0/0/0)
- youyouwou v93gl2: POP_W 0.20
- jianyicai v125: NP_W 0.04
- d1m4official midx-phys10: fragment-check order applied to ranks 2–10 after the prior
- adityaraj arm A / shalu arm B: midx layer on m1, prior on / off
- d1m4official flexon-v4 base / tail3 / deep
- cngnguynhuuz fork-rescue: rescue 3 on midx
- rajrajak99 v1700 / v1800: residual ranking over several engines, a gated Engine 2 promotion at rank 1, +0.40 for training-set structures. The team is stuck at 0.477 after 146 submissions.
- chyibo wmid125, guhongbin v556 (clone), yunlong v24 (identical to midx)

## Ranked candidate changes
1. **Fragment-aware same-formula reranker, built ourselves (heliosli-style).**
   - **Evidence:** it is the only public method above 0.479. It shows 0.480 on a 0.468 base (+0.012) and heliosli's team is at 0.486. It cannot be reproduced: the weights are private and the verbatim rerun gave 0.468.
   - **Caution:** our PeakFormer version of this idea lost (−0.005 / −0.016). A new attempt would need explicit fragment-explanation features rather than a re-weighting of PeakFormer scores. The `_RR_SRC` block in cngnguynhuuz fork-479fch has runtime code for these features (1–2 single-bond-cut fragment masses, explained intensity, complementary pairs, neutral losses); read it only, don't run it.
   - **Cost:** high. Expected gain uncertain, possibly large.
2. **Documentation-prior dose on ranks 2–3 (for example POP_W 0.20).**
   - **Evidence:** keeping the prior off ranks 2–3 cost 0.002; prior off vs on gave 0.471 vs 0.472 on another base; Kunal's w_pop 0.14 was confounded with isomer swaps (0.476).
   - Let youyouwou v93gl2 (POP 0.20), guhongbin v557 (prior off) and jianyicai v125 (NP 0.04) give us their pages first. Expected change ±0.002.
3. **ENG3 (adding our engine's candidates).** The closest public analogues were flat: the formula channel inserted at rank 8 and outsiders in ranks 21–25 both stayed at 0.479. That supports the Nov 2 prediction (+0.003 ± 0.008), with a lean toward flat.
4. **Identical resubmission of pubm.** This is a noise read, not a method. Some of the 0.478 → 0.480 jumps are probably rerun luck, so it can lift the public rank but does not change the private LB.
5. **Do not use**, all measured ≤ 0.479:
   - library gate 0.80
   - BIO_W
   - TRAIN_W / training-structure bonus
   - freezing ranks at library match ≥ 0.95
   - outsiders in ranks 21–25, or deep outsiders (40/150, 60/220)
   - PubChem formula-channel insert
   - single-swap policies
   - w_pop 0.14 with isomer swaps
   - protecting ranks 2–3 from the prior
   - "clean preprocessing" (−0.007)
   - fragment-check pool size
   - w_top 0.1
   - rescue (−0.006 on m1, per alexchilton)

## Files
Everything is under `/tmp/claude-0/-home-user-claude-code-tests/09a15150-bdf1-5786-8fb7-e9929f1c29ca/scratchpad/` (about 34 MB; 771 MB disk left). Nothing was executed, pushed or committed.
- LB csv: `pubrefresh/lb/enveda-CASMI26-molecule-id-mass-spectra-publicleaderboard-2026-11-01T06:43:11.csv`
- CLI listings: `pubrefresh/list/daterun_p1.csv`, `pubrefresh/list/score_p1.csv`, `pubrefresh/list/score_p2.csv`
- Web listings with `bestPublicScore`: `pubrefresh/web_SCORE_DESCENDING_1.json`, `pubrefresh/web_SCORE_DESCENDING_2.json`, `pubrefresh/web_DATE_RUN_1.json`, `pubrefresh/web_DATE_RUN_2.json`
- Pulled notebooks (40): `pubrefresh/nb/`, with text dumps in `pubrefresh/txt/`
- My scripts: `pubrefresh_tools/lbdiff.py` (LB diff), `pubrefresh_tools/ndiff.py` (notebook diff that repairs the Cyrillic-decoded box characters)

## Diffs

# Top-3 public candidates vs our fork casmi26-pubm (LB 0.479): diffs, expected LB, eligibility, drop-in

I could not save `/tmp/claude-0/-home-user-claude-code-tests/09a15150-bdf1-5786-8fb7-e9929f1c29ca/scratchpad/pubrefresh/report.md`: the harness blocks subagents from writing report files. The full report is below, ready to save as is.

**Headline:**
- **Candidate 1 (heliosli's reranker):** the only public gain above 0.479, but its dataset is still private, so we cannot run it.
- **Candidate 2 (prior dose):** flat in every reading so far; wait for three pending public scores instead of spending a slot.
- **Candidate 3 (extra candidates):** the public versions of the idea were flat, so I lowered the ENG3 prediction to +0.001 ± 0.006.

Reads were taken Nov 1, 06:43–07:08 UTC.

## 0. How the candidates map onto pubm

Most candidates are forks of the midx notebook (`baidalinadilzhan/casmi26-midx-exact-dedup-fill`, read through alexchilton's verbatim copy). I matched cells by content after repairing the garbled box characters. pubm equals midx cell for cell, except:

| pubm cell | midx cell | Difference |
|---|---|---|
| 1 config | 3 | Adds our OURS / PF / ISO lines; the final-layer lines are midx's |
| 8 pinned sources | 10 | Identical, including the frozen 0.465 notebook (59 cells) |
| 11 PF-syn | none | Ours; OFF in pubm (`OURS = False`) |
| 12 stage calls | 13 | +5 lines: the `ours_pf_blend()` wrapper, a no-op with OURS off |
| 16 final layer | 17 | midx cell verbatim (SHA `06093f58…`, already pinned in the generator) plus our `_mid_neural_score` hook, a no-op at `PF_MID_W = 0` |
| 17 ISO | none | Ours; OFF (`ISO_ALPHA = 0`) |

**Drift since the push.** Rebuilding today with `make_pub_variant.py casmi26-pubm OURS=False` gives pushed pubm plus:
- two cell-1 lines (`ISO_COS_W`, `ISO_FREEZE_LEADER`);
- the newer ISO cell 17.

Both are no-ops at `ISO_ALPHA 0` and are already in the verified eng3a / eng3b builds. Every variant below is "today's rebuild plus the hunk".

**Hunk classes:**
- final layer: cell 16 onwards, list edits after fusion
- fusion weights: CUSTOM / VARIANT / RRF knobs
- pipeline: stages, run mode, frozen cells
- data: datasets and model files

---

## 1. Candidate 1: learned same-formula isomer reranker (heliosli "rr3")

**Sources:**
- `heliosli/happy-to-make-the-dataset-public-if-it-helps`: run Oct 31 14:29, page 0.480; heliosli's team best 0.486 predates Oct 31 05:08.
- The Oct 31 pull of the same notebook, which was then the "rr3e" version averaging three rerankers.
- `romanrozen/0-468-casmi-from-peaks-to-structures` (page 0.472), which embeds an older copy of the overlay.
- The `_RR_SRC` block in cngnguynhuuz fork-479fch cell 22 (fragment-feature code).

**Diff vs pubm:**

| # | heliosli | pubm | Class | What |
|---|---|---|---|---|
| H1.1 | cell 25 (8,381 chars, sha256 `dd4af66c…`) | none; would go after cell 16 | final layer + data | The `_rr3_overlay()`, described below |
| H1.2 | cell 2 config | cell 1 | fusion weights | `VARIANT 'V2'` (w_top .25 / w_mid .50) vs our `CUSTOM` .25 / .75; `OUTSIDERS_PER_MOLECULE` 12 vs 20; no fragment-check, final-layer, OURS or ISO lines. Base difference, do not port. |
| H1.3 | cell 11 pinned sources | cell 8 | pipeline | Same modules and same frozen pipeline. Only frozen cell 28 differs: heliosli asserts Python 3.12 and a T4; pubm allows 3.13 and any GPU. No LB effect. |
| H1.4 | no fragment check, provenance or final layer | cells 10, 13, 14–15, 16 | final layer | heliosli's base is plain V2 (0.468) with no fragment check and no prior. This matters for transfer. |
| H1.5 | stage cells 13 / 15 / 17 / 19 | cell 12 | pipeline | Same calls split over four cells; pubm adds the PF wrapper (off) |

**What the overlay (H1.1) does:**
- Loads `rr3_config.json`, `rr_deploy.py`, `rr_v3.py`, the encoder, the checkpoint and a self-test file from a mounted dataset whose source is blank, i.e. private.
- Runs a three-molecule self-test and requires correlation ≥ 0.999 with the offline scores.
- Skips any list whose best library match is ≥ 0.9.
- "FREEZE" mode keeps every training-set structure in its slot and reorders the rest by z(−log rank) + alpha · z(model). Mode, alpha and delta sit in the private config.
- Budget 2,400 s. Any error leaves submission.csv unchanged.
- Without the dataset it prints `[rr3] skipped` and does nothing. That is why alexchilton's verbatim rerun scored 0.468, the bare base.
- It imports the casmi package from `/kaggle/working/v43_v4code`.
- The model is a frozen "FS0" encoder (the v4b fingerprint network) plus graph and transformer heads over peaks matched to 1–2-bond-cut fragments.
- The rr3e version's comments name the dataset packages: `heliosli/casmi26-rr-v3` ("the 0.483 scorer"), `casmi26-rr-v5` and `casmi26-rr-v6`.

All inputs the overlay needs already exist in a pubm run:
- `frozen_candidate_evidence.parquet` (written in cell 9);
- `/kaggle/working/v43_v4code` (written by frozen cell 27);
- the competition test and train parquet files, which are the shortest glob hits.

**Evidence and expected LB:**
- **Single reranker on the 0.468 base: +0.012** (page 0.480).
- **Which version scored it:** the page read 0.478 in alexchilton's notes, written by Oct 31 12:28, while the three-reranker version was already live. So the 0.480 most likely belongs to the later single-reranker version (14:29 run, heliosli submitted at 15:00), and the three-reranker version scored ≤ 0.478.
- **Expected if ported to pubm:** +0.004 to +0.010, centre about +0.006. Our fragment-check rank-1 moves (worth about +0.005 in this lineage, 0.472 → 0.477) do part of the same job, and ranks 2–25 already carry w_mid .75 plus the prior.
- **Unlearned fragment re-orders are measured negative:**
  - guhongbin `v549-frag-adjswap` writes the fragment check's order into ranks 2–10: 0.472 vs 0.477 on the older m1 base (−0.005).
  - d1m4official `casmi26-midx-phys10` is the same hunk on midx: cell 16, +15 lines, `PHYS_REORDER`, ranks 2–9 re-sorted by the physical order. It is unscored; I predict −0.003 to −0.005.
  - Our own PF isomer re-sorts lost too: ISO 0.75 −0.005, ISO 1.5 −0.016, PF on ranks 2–25 −0.007.
  - So the positive reading needs a learned model, not a re-weighting.

**Eligibility:**
- **The data is unpublished.** At 07:00 UTC, `datasets list --user heliosli` shows only RSNA and QCC datasets, and searches for "rr3" and "casmi isomer reranker" return nothing. It cannot be attached, so a verbatim port does nothing.
- **Even if published, it is rank-line only:**
  - training data and licence are undisclosed, so NIST and non-commercial status are unknown;
  - it depends on the v4b code and the FS0 encoder, which in pubm come through the CC BY-NC-SA v4g container (A5, ineligible) and upstream are A6 / A11 (unclear).
- `_RR_SRC` is Apache-2.0 notebook code, eligible with attribution. `frag_rescore.py`, already in pubm cell 10, is A24, eligible.

**Drop-in (make_pub_variant.py terms):**

*A. Port, if the dataset becomes public.*
1. Apply `patches/c2_tuple_setter.diff`, then `patches/c13_fch_rr3_switches.diff`.
2. Copy `patches/rr3_cell25.txt` (SHA-pinned in `EXTRA_CELLS`) into `scripts/pub_variant/`.
3. Build: `python scripts/make_pub_variant.py casmi26-pubm-rr3 OURS=False RR3=1`.

What the build does (prototype verified: only that cell differs from today's pubm rebuild, and `ast.parse` passes):
- New cell 17 is heliosli cell 25 verbatim plus a two-line attribution header; ISO moves to cell 18.
- The cell is inserted after `elig.apply` and `eng3.apply`, which address cells by template index.
- `heliosli/casmi26-rr-v3` is appended to dataset_sources.
- `RR3=1` refuses `ISO_ALPHA > 0`.

Before any push, confirm the slug is public; the current cell globs any `rr3_config.json`, so one package is enough. In the commit log, look for `[rr3] self-test … 0.999xx` and `[rr3] done: {… reranked …}`. A `[rr3] skipped` line means the run equals pubm. In the commit smoke run all 12 visible molecules have library match 1.0, so `reranked = 0` there is expected.

*B. Our own learned reranker (the real candidate 1; cost: days).*
- In `scripts/pub_variant/iso_cell.py`, add `ISO_MODEL` (`'pf'` default, `'rr'` new). In `iso_apply`, when it is not `'pf'`, build `pf_of` from a scorer instead of `iso_pf_lookup`.
- Features per (molecule, candidate):
  - `frag_rescore.score_molecule` scores under DEFAULT_CFG and BASELINE_CFG, plus per-energy votes. The module is `ROOT/frag_rescore.py`, written by cell 10, SHA `eeacdcf0…`; spectra come via `parallax_physical.select_spectra`.
  - The `_RR_SRC` features: explained intensity, matched fraction, complementary pairs, neutral losses.
  - neural_score, baseline_rank and pop.
- The model goes in `bryantyu/casmi26-models`.
- In make_pub_variant.py:
  - add `ISO_MODEL = 'pf'` to the ISO block of CFG;
  - assert `cfg['ISO_MODEL'] in ('pf', 'rr')`;
  - relax the OURS check to `(PF_MID_W > 0 or (ISO_ALPHA > 0 and ISO_MODEL == 'pf')) and not OURS`.
- **Gate:** push only if, on our structure-disjoint validation lists, the model beats the base order inside the leader-formula group (base-order first-rate 0.748; PF alone 0.460) with a bootstrap CI that excludes 0.
- Do not add an unlearned `'frag'` mode or the phys10 / adjswap re-order; both are measured negative.

---

## 2. Candidate 2: documentation-prior dose (POP_W 0.10 → 0.20)

**Diffs vs pubm** (all exact):

| Notebook (status) | Hunk in pubm terms | Class |
|---|---|---|
| youyouwou `v93gl2` (run 06:24, unscored) | cell 1: `POP_W, SPEC_W, NP_W = 0.10, 0.05, 0.02` → `0.20, 0.05, 0.02` | final layer |
| youyouwou `v93gl2` | cell 2: forces `IS_RERUN = True` and sets `KAGGLE_IS_COMPETITION_RERUN=1` (their own workaround for a 0.000 score). Our guard demonstrably scores, so **do not port**. | pipeline |
| guhongbin `v557` (unscored) | cell 1 → `0.0, 0.0, 0.0` | final layer |
| jianyicai `v125` (unscored) | cell 1 → `0.10, 0.05, 0.04` | final layer |
| d1m4official `midx-top3-prior` (page 0.477) | cell 1: `+PROTECT_TOP = 2`; cell 16: the single sort of ranks 2–25 becomes "keep the first two, sort the rest" | final layer |
| d1m4official `v123-pop15-prior` (pulled now, page 0.472, v123 base) | → `0.15, 0.05, 0.02` | final layer |
| d1m4official `v123-top3-prior` (pulled now, page 0.472) | `PROTECT_TOP = 2` hunk on v123 | final layer |

**How strong the knob is.** From the local `pool_popularity.csv` (710,701 rows), `pop` has median 2.08, 90th percentile 4.41, 99th 10.1, max 23.7.

| POP_W | Prior term (POP_W · pop / 20) at median / p90 / p99 |
|---|---|
| 0.10 | 0.010 / 0.022 / 0.051 |
| 0.20 | 0.021 / 0.044 / 0.101 |

The base-rank gaps `1/(3+r)` are 0.033 between ranks 2 and 3, 0.014 between 5 and 6, and 0.0055 between 10 and 11; the neural term adds up to 0.75 × the same. So 0.10 mostly reorders ranks ≥ 6, while 0.20 starts flipping ranks 3–4 and occasionally rank 2. That makes v93gl2 the right test of "prior dose on ranks 2–3".

**Evidence:**
- **v123 base (0.472):** POP 0.15 → 0.472 (0.000); protect ranks 2–3 → 0.472 (0.000); prior off → 0.471 (−0.001); library gate 0.80 with protect → 0.472. No dose response.
- **midx base:** protect ranks 2–3 → 0.477 (−0.002). This reading is weaker than the survey implied: d1m4official's team best is 0.477 and their own verbatim midx run (`casmi26-midx-0479-base`) is unscored, so it compares across accounts and sits at the documented ±0.002 rerun noise.
- **Kunal's MID (w_pop 0.14 + isomer swaps) → midx (0.10 + gate / pin / fill):** confounded.

**Expected on pubm:**

| Change | Expected LB |
|---|---|
| POP_W 0.20 | −0.001 ± 0.002 |
| NP_W 0.04 | 0.000 ± 0.002 |
| prior off | −0.001 ± 0.002 |

Free reads are pending: v93gl2, v557 and v125 are midx-identical except for the one line, so each page is a direct pubm proxy. adityaraj arm A and shalu arm B (prior on / off on the m1 layer) are also pending.

**Eligibility:** no new data. `pop` and the natural-product columns come from `pool_popularity.csv` (A17). Only the SPEC_W term uses the NIST-depositor flag (unclear); prior-off removes it.

**Drop-in:**
- **Patch:** `patches/c2_tuple_setter.diff`. Today the generator cannot set POP_W at all: `config_values` / `set_config` only accept single-name assignments, and this line is a tuple. The patch adds tuple support (`_tuple_names`), keeps the comment column, and leaves elig.py's literal splice working, because KEY=VALUE settings apply after the elig splice.
- **Tested in a scratch copy** (repository untouched). Each build differs from today's pubm rebuild in exactly one cell-1 line:
  - `python scripts/make_pub_variant.py casmi26-pubm-pop20 OURS=False POP_W=0.2` → `POP_W, SPEC_W, NP_W = 0.2, 0.05, 0.02`
  - `… casmi26-pubm-np04 OURS=False NP_W=0.04` → `0.1, 0.05, 0.04`
  - `… casmi26-pubm-prior0 OURS=False POP_W=0.0 SPEC_W=0.0 NP_W=0.0` → `0.0, 0.0, 0.0`
  - `ELIG=0 … POP_W=0.2` → `0.2, 0.0, 0.02    # elig0 (F3) …`; elig's AST check still passes.
- **Log check:** `Final layer: … POP_W 0.2, NP_W 0.02` (cell 16's existing print).
- **PROTECT_TOP** is not recommended (0.000 / −0.002). If wanted: a CFG line `PROTECT_TOP = 0` (0 = midx exactly) plus one cell-16 splice of `rest.sort(key=_score_cand, reverse=True)` into the keep-first-two form, guarded by `if PROTECT_TOP > 0`.
- **Recommendation:** don't spend a slot. Read v93gl2 / v557 / v125 when their authors submit; build pop20 only if v93gl2 shows ≥ 0.480.

---

## 3. Candidate 3: extra candidates in the list (our ENG3 and its public analogues)

**Our ENG3** (`kernels/casmi26-pubm-eng3a` / `-eng3b`) vs pubm. Verified and already set for Nov 2 slots 4–5:

| Hunk | Class |
|---|---|
| cell 1: ENG3_W block plus the two ISO lines | fusion weights |
| cell 8: frozen cells 32 / 39 patched (`d1e757c2…` → `5d159bb1…`) plus the ENG3 runtime modules | pipeline |
| cell 9: stage label and regex (2 lines) | pipeline |
| cell 17: newer ISO cell (no-op) | final layer |
| metadata: + `bryantyu/casmi26-elig-assets`, + `bryantyu/casmi26-pool-v18c` | data |

**Public analogues (diffs vs pubm):**

| Notebook (page) | Hunks in pubm terms | Class |
|---|---|---|
| cngnguynhuuz `fork-479fch` (0.479); cells 0–21 are midx verbatim | New cell after 16: the formula-channel overlay (FCH), described below. Its extra dataset `ahmedberatozer/casmi26-pubchem-tier` is redundant for pubm, which already mounts the tier inside `dmitriigluzdov/casmi26-pubchem-popularity-prior`. | final layer + data |
| cngnguynhuuz `fork-fch` (0.478) | Same overlay on the older m1 base (0.477) | final layer |
| cngnguynhuuz `fork-glf` (0.479) | FCH restricted to lists with library match < 0.9; top 60 re-scored by GLACIER predicted spectra through the casmi26-iceberg / -glacier runner code; same rank-8 insert | final layer + pipeline |
| flexonafft `closing-the-evidence-gap` v4 (0.479); takumuhata `sp478gap` is byte-identical | 40 outsiders / 150 identity checks; cell 9 passes `OUTSIDER_MAX_CHECKS`; ranks 21–25 opened to outsiders (`TAIL_SLOTS = 5`). Also rank 1 decided by busyaprime's m1 rule (stricter guards than midx), the prior re-implemented, and midx's in-pool backfill dropped. | fusion weights + pipeline + final layer |
| d1m4official `flexon-v4-deep` (unscored) | Same, with 60 outsiders / 220 checks | pipeline |

**What FCH does,** for every list with ≥ 8 answers:
- Takes the median neutral mass and searches the 105.9 M-structure PubChem tier within ±10 ppm.
- Keeps structures with the rank-1 candidate's formula that are not already in the list.
- Scores them by fragment explanation (0.7 × explained intensity + 0.3 × matched-peak fraction).
- Inserts the top 5 at ranks 8–12, so old ranks 21–25 fall off.
- Budget 3,000 s, in submission order; molecules after the budget are untouched.
- **No library gate** (the survey's "< 0.9" applies to glf only).
- Dedups by plain InChIKey14 without tautomer normalisation, so it can re-add a duplicate.

**Evidence and expected LB:**

| Change | Public reading | Expected on pubm |
|---|---|---|
| FCH | m1 0.477 → 0.478 (+0.001); midx 0.479 → 0.479 (0.000) | +0.0005 ± 0.002, up to +50 min runtime |
| glf | 0.479 | no better than FCH |
| Tail-5 with 40 / 150 outsiders | flat on its own non-midx base; a 3-slot tail rescue on m1 measured −0.006 | −0.001 ± 0.003 |
| ENG3 | analogues all flat; but ENG3 enters through Engine 2 and fusion, so it can reach rank 1 (local run: rank 1 changed 0 of 12, all library spectra) | **revised to +0.001 ± 0.006** (was +0.003 ± 0.008); the ≥ +0.004 keep rule stands |

**Eligibility:**
- FCH: PubChem tier (A15) and Apache-2.0 notebook code — eligible with attribution, no non-commercial data.
- glf: the ICEBERG / GLACIER runner code is unlicensed (A21 / A22).
- evidence-gap: no new data.
- ENG3 sits on pubm, which is rank-line only (v4g container A5, Engine 2 A29–A35); the assets it adds are ours and eligible.

**Drop-in:**
- **ENG3:** as built. `python scripts/make_pub_variant.py casmi26-pubm-eng3a ENG3_W=1.0 OURS=False PF_MID_W=0 ISO_ALPHA=0` (eng3b uses `ENG3_W=0.5`).
- **FCH** (optional, near-zero effect):
  1. Apply `c13_fch_rr3_switches.diff`.
  2. Copy `patches/fch_cell22.txt` (sha `de4b7b27…`) to `scripts/pub_variant/`.
  3. Build: `python scripts/make_pub_variant.py casmi26-pubm-fch OURS=False FCH=1`.
  - Result: new cell 17, ISO moves to 18, and ISO's final checks also validate FCH's output. No dataset change.
  - It combines with ENG3: a `casmi26-pubm-eng3a-fch … FCH=1` build passes.
  - Log check: the `[fch] overlay {… inserted …, budget_hit …}` line and `fch_overlay.json`.
- **Tail-5 / deep outsiders:** not recommended. If wanted, it needs:
  - `OUTSIDERS_PER_MOLECULE=40` (already settable);
  - a CFG line `OUTSIDER_MAX_CHECKS = 30`;
  - a cell-9 splice adding it to the outsiders.py arguments;
  - a cell-16 splice for the tail admission, which needs per-candidate neural scores.

---

## Summary

| Rank | Change | Best public reading | Expected on pubm | Eligibility | Cost | Action |
|---|---|---|---|---|---|---|
| 1a | heliosli reranker port | +0.012 on 0.468 base (page 0.480; team 0.486) | +0.006 (+0.004 to +0.010) | private dataset; v4b / FS0 encoder via the non-commercial container in pubm; training data unknown | ~1 h once public | You could ask heliosli on the notebook (its title offers the dataset) to publish `casmi26-rr-v3` — a public post under your account, not done by me. Then `RR3=1`. |
| 1b | our own learned fragment reranker in ISO | none; unlearned versions −0.005 to −0.016 | 0 ± 0.008 before offline evidence | eligible if trained on our data | days | Only behind the offline within-formula gate |
| 2 | POP_W 0.20 / NP_W 0.04 / prior off | v123: 0.15 flat, off −0.001; midx protect-ranks-2–3 −0.002 (cross-account) | −0.001 ± 0.002 | neutral | ~0, patch ready | Wait for the free pages (v93gl2 / v557 / v125) |
| 3 | ENG3 (built) / FCH / tail-5 | FCH +0.001 (m1) / 0.000 (midx); tail-5 flat | ENG3 +0.001 ± 0.006; FCH +0.0005 ± 0.002; tail −0.001 | ENG3 / FCH eligible-friendly; glf runner code unlicensed | done / ~0 / small | Keep Nov 2 slots 4–5; FCH only as a filler |

**Tie block, unverified.** 230 teams show 0.479 (ranks 45–274), and the tail of the block is roughly ordered by last submission. If those ties are exact, any change that nudges the exact score up clears the block (about rank 45) even while the display stays 0.479, and any loss drops us below it. That would favour changes with non-negative expectation (FCH, ENG3) over ones leaning negative (POP_W 0.20).

## Corrections to the survey
- fork-479fch has **no** library gate; only glf gates at < 0.9.
- heliosli's 0.480 most likely belongs to the current single-reranker version, not the three-reranker one.
- The dataset slugs are `heliosli/casmi26-rr-v3` (plus `-rr-v5`, `-rr-v6` in the three-reranker version); all private at 07:00.
- d1m4official's −0.002 for protecting ranks 2–3 compares across accounts and sits at the noise floor; on the v123 base the same hunk is 0.000, and POP 0.15 is 0.000.
- `midx-phys10` is the same hunk as guhongbin's `v549-frag-adjswap`, which scored −0.005.
- closing-the-evidence-gap is not midx-equivalent at rank 1 (it uses the m1 rule) and drops midx's in-pool backfill.
- The pushed pubm lacks the two newer ISO config lines and the newer ISO cell; both are no-ops.

## What I ran
- **Kaggle calls:** `datasets list --user heliosli`, two `datasets list -s` searches, and `kernels pull` of d1m4official's `v123-pop15-prior` and `v123-top3-prior`.
- **Notebooks:** read only (JSON, base64, `ast.parse`, difflib), nothing executed.
- **Generator:** a patched copy of our own `make_pub_variant.py`, run in the scratchpad; it only `ast.parse`s cells.
- **Repository:** unchanged (`git status` clean); nothing pushed or committed.

## Files
All under `/tmp/claude-0/-home-user-claude-code-tests/09a15150-bdf1-5786-8fb7-e9929f1c29ca/scratchpad/` (about 36 MB; 762 MB of disk left):
- Per-hunk diffs: `pubrefresh/diffs/c1_*.diff`, `pubrefresh/diffs/c2_*.diff`, `pubrefresh/diffs/c3_*.diff`
- Generator patches: `pubrefresh/patches/c2_tuple_setter.diff`, then `pubrefresh/patches/c13_fch_rr3_switches.diff` on top
- Pinned cell sources: `pubrefresh/patches/fch_cell22.txt`, `pubrefresh/patches/rr3_cell25.txt`
- Patched generator copy: `pubrefresh_tools/mpv/scripts/make_pub_variant.py`
- Test builds, in `pubrefresh_tools/mpv/out/`:
  - `casmi26-pubm`
  - `casmi26-pubm-pop20`
  - `casmi26-pubm-np04`
  - `casmi26-pubm-prior0`
  - `casmi26-pubm-fch`
  - `casmi26-pubm-rr3`
  - `casmi26-pubm-eng3a-fch`
  - `casmi26-x` (ELIG=0 + POP_W 0.2)
- New pulls: `pubrefresh/nb2/v123-pop15-prior/`, `pubrefresh/nb2/v123-top3-prior/`
- Tools: `pubrefresh_tools/cells.py`, `pubrefresh_tools/cmatch.py`, `pubrefresh_tools/cdiff.py`, `pubrefresh_tools/frozen_cmp.py`
