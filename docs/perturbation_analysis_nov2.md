# Why perturbations of the 0.479 line lose on the public LB (Nov 2)

## Synthesis

## Synthesis: what the Nov 1–2 public losses mean, a Nov 3 design and Dec 14 picks

Nothing was executed from the notebooks, pushed or committed. I wrote and ran one small simulation of our own: `/tmp/claude-0/-home-user-claude-code-tests/09a15150-bdf1-5786-8fb7-e9929f1c29ca/scratchpad/synth/swap_mc.py`. It extends gate_mc.py to a single-seed swap.

### 1. Which explanations the evidence supports

| Explanation | Verdict | Basis |
|---|---|---|
| **Selection optimism on the 124 public molecules** | Best-supported explanation for the small re-weighting and neural-term readings (PF_MID, ISO 0.75, PF blend, SENS, mostly ISO 1.5). | It predicts both the sign and the size of 5 of the 6 readings, and the dose-response. It also explains why no community perturbation read above +0.001. Optimism is about 0.015 (0.009–0.019). Caveat: the existing readings only reject "neutral with no selection". They cannot reject "the line is at a true optimum and every change is mildly harmful". |
| **Absolute cos/fp threshold coupling** (plan hypothesis b) | Not supported. | Analyses B and C agree. The .23 relative margin dominates the absolute gates unless the list's score std is below .13. SENS averages cos and fp in their own units (raw mean = z mean, no unit break). |
| **Relative rank-1 gate coupling** (V1 flip rule, the .23·sd margin, the V2 contender pool, all tuned on the published model's noise) | Plausible for SENS and PF. Cannot be separated from selection on the public board, and is partly the same thing. | Averaging flips about 23% of gate decisions and lowers the pass rate from .32 to .24. This is the one open question worth slots (section 2). |
| **Distribution shift** (holdout differs from hidden) | Supported as the reason the SENS prediction was wrong, not as the cause of the negative sign. | The holdout is 66.5% Enamine; the hidden set is timsTOF natural products. SENS gains +0.042 on the train-like visible 12. In pubm the neural term is small, so the ceiling is about +0.001 even with perfect transfer. SENS has no upside in any pick, whichever cause is true. |
| **ENG3: real harm from Engine 2 dilution** | Supported. | ENG3 is the only reading −1.8σ beyond the neutral expectation. The mechanism is concrete: Engine 2's unique picks are displaced, which changes b1 and the baseline ranks. It is consistent with our Engine 2 being weaker (pube1 is 0.044 below pubm). Close the eng3 line; it has no value for any pick. |
| **ISO** | Real harm likely at α ≥ 0.75, whatever the selection effect. | The offline B0gf run predicted −0.022 at α 0.75 (double-counted fragment evidence). Stays closed. |

**General lesson:** this board can price structural changes, where the spread of true effects is 0.02 or more (public/private correlation 0.8–0.98: pube0, pube1, eng3a). It cannot price re-weightings (correlation 0.04–0.7).

**Where I disagree with analyses B and C on SENS-mid:** I would not build it.
- At its σ (about 0.003), selection predicts −0.004.
- The "rank-1 mechanical" explanation predicts about 0, which is no more than about 1σ away.
- B's rule (≥ −0.004 means mechanical) would therefore count a reading that selection itself predicts as evidence for the mechanical explanation.
- Its private upside is at most about +0.0005.

### 2. Nov 3 slots: seed-swap placebo, 2×2 factorial

**The placebo.** Replace the published saner model with one refit seed of the same recipe. By exchangeability its true private effect is 0.
- **Selection (H1):** it should read about −c·σ on public.
- **Averaging-specific real harm (H3):** it should read about 0, because a swap leaves the gate pass rate unchanged.

Simulation, single-seed swap vs the 5-seed ensemble:

| | Single-seed swap | 5-seed ensemble |
|---|---|---|
| Gate-decision flips | 0.33–0.36 | 0.22–0.24 |
| Pass rate | 0.31 → 0.32 | 0.31 → 0.24 |
| Neural top-1 changed | 0.60–0.63 | 0.42–0.44 |
| Spearman vs published | 0.85–0.90 | 0.93–0.95 |

So σ_swap is about 0.012–0.014.

**Patch.** Add SENS_MODE `'swap'` (about 20 lines).
- Add `'swap'` to `sens.MODES` and to the SENS_MODE check at make_pub_variant.py line 103.
- In `sens.check_config`, require SENS == 2 for swap.
- In `sens_cell.sens_ensemble_frames` (main lists): `e = s[1]`, and neural_cos / neural_fp `= vals[c][1, idx]`.
- For outsiders: `e = s[1]` and `cf = vo[c][1, idx]` where the seed scored; the published value where only the published model scored.
- Add swap wording to `_note` and `_elig1_statement`, and one test in test_sens_cell.py.
- The seed is chosen with the existing cell-1 override `SENS_SEEDS=[s]`. It is applied after the SENS edits, and `apply()` reads `cfg['SENS_SEEDS']`.

| slot | kernel | build | H1 (selection) | H3 (averaging harm) |
|---|---|---|---|---|
| 1 | casmi26-pubm-swap1: `make_pub_variant.py casmi26-pubm-swap1 SENS=2 SENS_MODE=swap SENS_SEEDS=[20262023] OURS=False PF_MID_W=0 ISO_ALPHA=0` | new | 0.466 ± 0.011 (−0.007 to −0.019) | 0.479 ± 0.013 |
| 2 | casmi26-pubm-swap2: same command with `SENS_SEEDS=[20263023]` | new | 0.466 ± 0.011 | 0.479 ± 0.013 |
| 3 | casmi26-pube1-swap1: `ELIG=1 SENS=2 SENS_MODE=swap SENS_SEEDS=[20262023] OURS=False PF_MID_W=0 ISO_ALPHA=0` | new | 0.426 ± 0.011 (depends on how much optimism survives the base) | 0.435 ± 0.013 |
| 4 | casmi26-pube1-sens | already verified | 0.427 ± 0.010 | 0.420 ± 0.010 |
| 5 | casmi26-pube1-swap2 (`ELIG=1 … SENS_SEEDS=[20263023]`) | new | as slot 3 | as slot 3 |

- **Seed choice:** seeds 20262023 and 20263023 have holdout MRR 0.486 and 0.482, about equal to the published 0.484.
- **Verification line to expect:** "SENS applied | K 2 | mode swap", failures 0, changed top1 (neural) 6–9/12, rank corr 0.88–0.91, MERGE applied 400. The extra hidden-run cost is about 10 minutes.
- **Fallback:** if a swap kernel isn't verified by 00:05, use eng3a-sens for slot 5, or for slots 2 and 5. If the patch fails entirely, submit pube1-sens and eng3a-sens only, as a weak difference-in-differences comparison. Leave the remaining slots empty rather than spend them on eng3b or wm100, which have no decision value.

**Rules** (pre-register before reading). P = mean(slots 1, 2) − 0.479; Q = mean(slots 3, 5) − 0.435.

- **H1 supported:** P ≤ −0.008 and (−0.018 − P) ≥ −0.008. SENS's −0.018 is then read as consistent with neutral, and the public board is not used for any re-weighting or neural-term decision on lineage-derived bases.
- **H3 supported:** P ≥ −0.004 and (slot 4 − 0.435 − Q) ≤ −0.008. Ensembles of the neural term are then treated as harmful on private, and the published single model stays.
- **Otherwise inconclusive.** The default actions are the same.
- **Eligible-base calibration:**
  - Q ≤ −0.007: the eligible base still carries optimism tied to the neural term. Any eligible change read on public must beat (neutral expectation, Q scaled by σ_change/σ_swap) + 1.2σ.
  - Q ≥ −0.003: the existing 0.008 rule stands.
- **Noise check:** |swap1 − swap2| on each base is a direct estimate of paired noise (expect 0.010–0.016). If it is above 0.025, widen every interval by 1.5×.

**Statistical power:** about 1.3σ from P alone, perhaps 1.5–2σ combined. The two bases share the neural-tied component, so they are not independent. That is the most 124 molecules allow.

**Decisions that do not depend on the outcome:**
- SENS, PF_MID, ISO and eng3 enter no pick.
- No rank-line slots after Nov 3.
- Later slots go to eligible-line structural reads: PF-x Engine 2 in pube1, the F5b no-harm rebuild, and the own-line insurance builds. s2v18cuPe and s2v18cPe are still unbuilt and unread, against roadmap D2.

### 3. Dec 14 final picks

- **Eligible only.** The plan's rule is binding. pubm and its variants are excluded: they mount the non-commercial v4g container and run Engine 2 as published. Even on rank merit, pubm's private expectation is about 0.464 in public-equivalent units, which every verbatim copy, including the 230 teams at 0.479, would share.
- **Pick A: the eligible fork family (pube1).**
  - Its private expectation is about 0.425–0.435, after giving back part of the lineage optimism.
  - It leads our own line by about 0.087 public, against roughly 0.04 of paired split noise, so P(A beats the own line on private) ≈ 0.97 or better.
  - **Version:** plain pube1, rebuilt with the clean F5b runners if no glue licence arrives by Nov 7, plus a no-harm read. Swap in the PF-x Engine 2 (v18cx) only if two things hold:
    - its Engine 2 lists show a gain on hidden-like or COCONUT offline checks (the D3 GO was spectral +0.042);
    - its public read is no worse than the Q-calibrated neutral expectation minus 1σ.
  - Public differences below max(0.008, 1.2σ + |bias|), about 0.012–0.015 for re-draw-type changes on this base, never decide.
- **Pick B: our own clean line (per plan section 6) as eligibility insurance.** It almost never beats A on private; its value is the case where pube1 is disqualified. Things still open:
  - F5: the ICEBERG/GLACIER glue licence, or the clean-room ICEBERG runner, which isn't built yet;
  - F6: the host's answer on reconstructed or not-bit-exact retrains, with ranker_v4b's training rows unpublished;
  - the PubChem NIST-depositor flag question.
- **Switching B.** Only if all of those close by about Nov 20: replace B with a second, structurally different fork version (with and without the PF-x Engine 2). Diversification gives about 0.4·σ_D, roughly +0.003–0.005 on private.
- **Timeline:** every pick must be scored at least 72 h before D8 (Nov 23), so the F5b and PF-x versions need reads by about Nov 19–20. The owner selects in the UI; kernels freeze Dec 5; verify Dec 8 and Dec 12.
- **Uncertainty:** the ±0.05 public/private split noise in the absolute level is mostly common to both picks, so it doesn't affect the ordering. The real unknowns are Q, meaning how much of pube1's 0.435 is optimism it inherits (up to about 0.01), and the host's compliance rulings.

## Analysis A (selection)

**Analysis A: how much of the 0.479 is selection optimism on the 124 public molecules**

**Bottom line.** About 0.015 of the lineage's public 0.479 is probably selection optimism (range 0.009 to 0.019). The lineage is tuned to the public split, so a truly neutral change touching more than about 3% of lists should read −1.2σ (central case, range −0.6σ to −1.8σ) on the public board, where σ is the change's paired noise, capped at about −0.015. Five of our six readings, all except ENG3, are what neutral changes would show, so they say little about private value either way. ENG3 is the only reading clearly below that line. Nothing was executed from the notebooks, pushed or committed.

**1. Noise on 124 molecules**
- **Score noise for one submission.**
  - Class mix 19/45/36 with class-2 MRR 0.665 gives an overall MRR of about 0.47. The standard deviation of one molecule's reciprocal rank is 0.46–0.48.
  - The standard error is 0.042 on the 124 public molecules and 0.028 on the 276 private ones. The same submission's public and private scores differ with SD 0.050.
  - One molecule moving from rank 2 to rank 1 is worth 0.0040; one moving from absent to rank 1 is worth 0.0081.
- **Paired noise between two variants.** σ ≈ RMS_d × √(k/124), where k is the share of molecules whose truth rank changes and RMS_d is the typical size of the reciprocal-rank change on those molecules. From the ISO grid, RMS_d is 0.07 when only ranks 2–25 move, 0.26 for ISO 0.75, 0.37 for ISO 1.5 and 0.46 for ISO 3.

  | k | σ, ranks 2–25 only | σ, ISO-0.75-like | σ, ISO-1.5-like | σ, ISO-3-like |
  |---|---|---|---|---|
  | 5% | 0.0014 | 0.0052 | 0.0074 | 0.0092 |
  | 10% | 0.0020 | 0.0074 | 0.0105 | 0.013 |
  | 20% | 0.0028 | 0.0104 | 0.0149 | 0.0185 |

  Private-split noise is 0.67 × these.
- **Estimated σ of our variants:**
  - PF_MID ≈ 0.0045 and ISO 0.75 ≈ 0.0045 (each 0.0035–0.006)
  - ISO 1.5 ≈ 0.008 (0.006–0.011)
  - PF blend ≈ 0.010 and SENS ≈ 0.010 (each 0.007–0.013)
  - ENG3 ≈ 0.012 (0.009–0.016)
- **Basis for those estimates.**
  - ISO: the offline grid, scaled down by the public test's class mix.
  - Visible-12 diffs (`selopt/cmpsub.py`): the neural-term variants (pfmid, pfall, sens, wm100) re-order 22–31 of the 60 top-5 slots across the 12 visible molecules (37–52%). sens, pfall, wm100 and eng3a-sens all swap ranks 1 and 2 on the same near-tied molecule, m_068354.
  - eng3a puts about 3 new candidates per list into the top 25.

**2. Lineage history: selection intensity**
- **Submissions.** 4,493 teams made 91,177 submissions. Teams scoring ≥ 0.464 made about 48,300 of them.
  - 230 teams sit at exactly 0.479 (median 36 submissions each).
  - 44 teams sit at ≥ 0.480 (median about 81 submissions each).
  - Among lineage-band teams (0.477–0.484), the share above 0.479 rises from 0.6% for teams with ≤ 10 submissions to 13% for teams with 101–250.
- **Public notebooks (Oct 28).** 678 were scored, 138 of them on the lineage's v4g assets.
  - Between Oct 22 and Oct 27, 9–35 lineage notebooks were scored per day. Their within-day spread was SD 0.0026–0.0061 (0.0043 over all 120).
  - Each day's best sat 0.004–0.011 above that day's median. Picking the best of 9–35 noisy variants with that spread alone would put it about 0.006–0.008 above.
- **Earlier chain study** (`docs/research_oct25.md`): 481 scored versions; 95% of 470 steps were smaller than one standard error.
- **The steps from 0.465 to 0.479** were +0.001, +0.002, +0.004, +0.005, +0.001 and +0.001. Each is at or below one paired σ of a change that touches rank 1.
- **Perturbations of the tuned line.**
  - Community: rank-1 weight 0.50 −0.029; GLACIER weight 1.5 −0.020; long-trained saner −0.019; NX reranker −0.058; rescue −0.006; clean preprocessing −0.007; BIO_W / TRAIN_W −0.001 (two reads each); protecting ranks 2–3 −0.002; FCH / glf / single swap / tail-5 all 0.000.
  - None of about 20 community perturbations read more than +0.001. None of our 6 read above zero.

**3. Optimism: best-of-N and simulation**
- **Naive best-of-N.** The expected maximum of N standard normals is 1.17 / 1.54 / 1.87 / 2.25 / 2.51 for N = 5 / 10 / 20 / 50 / 100.
  - The optimism of one step is σ²/√(σ²+τ²) × E[max_N], where τ is the spread of the variants' true effects.
  - With σ = 0.004, τ = 0.002 and N = 20, that is about 0.007 per step. Six independent steps would add up to about 0.04.
  - It does not add up that far because every step works the same ~50 contested class-2 public molecules.
- **Simulation** (`sim.py`): a linear-scoring generative model calibrated to MRR 0.476, class-2 top-1 rate 0.61, a 124/276 split and a 20,000-molecule population. The lineage is a greedy climb that picks the best public score at each step.

  | Search | Optimism (public gain − true gain) |
  |---|---|
  | 4 steps × 5 variants | +0.009 |
  | 6 steps × 10 variants | +0.013 |
  | 8 steps × 30 variants, small / medium / large tweaks | +0.0125 / +0.017 / +0.019 |
  | 10 steps × 100 variants | +0.019 |
  | 8 steps × 30, start far from the optimum | +0.015 (alongside a true gain of +0.023) |

  - The 10th–90th percentile range across runs is +0.002 to +0.030.
  - An oracle search over 20,000 weight vectors gives a ceiling of about 0.020 for a five-knob family.
  - Where little real gain was available, public climbs of +0.011 to +0.022 came with true gains of only +0.001 to +0.004.
  - The real line has more knobs, including thresholds that can fit single molecules (margin 0.095, nmargin 0.23, library gate 0.87). That argues for the upper half of this range.
- **Estimate.** Optimism in the lineage's 0.479 is about 0.015 (0.009–0.019). Its private expectation in public-equivalent units is about 0.464 (0.460–0.470), with ±0.05 split noise in the absolute level. That noise is common to all copies of the line, and the 230 teams at 0.479 (and every verbatim midx copy, including our pubm) would get the same private score.

**4. Expected public delta of a truly neutral variant, and the public/private correlation**
- **Control:** with no selection, the bias is 0.000 ± 0.001 and P(reading < 0) is about 0.5.
- **After selection,** the bias is −c·σ, capped at −O, where O is the lineage optimism.
  - c ≈ 0.56 for light search, 1.15–1.5 for moderate, 1.8 for heavy.
  - The residual SD is 0.83–1.0 σ, and P(reading < 0) is 0.8–0.98 once σ ≥ 0.004.
  - The bias is the same whether the variant is a weight tweak, a new feature or a re-noised feature (for example −0.008 / −0.011 / −0.009 at mean k of 8–12%). It depends on how many contested molecules the change re-draws, not on its direction.
- **Neutral-expected readings, central (c 1.2, O 0.015):**
  - σ 0.0045: −0.005
  - σ 0.008: −0.010
  - σ 0.010–0.012: −0.012 to −0.014
  - The light-to-heavy range at σ 0.0045 is −0.003 to −0.008.
- **Correlation** between the public and private delta, with residual 0.9σ:

  | Variants | σ 0.004 | σ 0.008 | σ 0.012 |
  |---|---|---|---|
  | Small tweaks (τ 0.002) | 0.29 | 0.09 | 0.04 |
  | Our variant types (τ 0.005) | 0.72 | 0.39 | 0.22 |
  | Big structural changes (τ 0.02) | 0.98 | 0.91 | 0.82 |

  - The regression slope of private on public is 0.18–0.33 for our variant types.
  - The simulation agrees: correlation 0.93 for tweaks (true-effect spread 0.03), 0.57–0.65 for new features (0.010), 0.10–0.18 for re-noised features (0.004).

**5. What −0.005 to −0.034 implies about private value** (central case; priors are our pre-registered predictions)

| Variant | Public | Neutral expect | Residual z | Posterior private Δ | P(≥ 0) |
|---|---|---|---|---|---|
| PF_MID | −0.007 | −0.0054 | −0.4 | −0.0005 ± 0.0034 | 0.44 |
| ISO 0.75 | −0.005 | −0.0054 | +0.1 | +0.0009 ± 0.0038 | 0.60 |
| PF blend | −0.014 | −0.0120 | −0.2 | +0.0002 ± 0.0067 | 0.51 |
| ISO 1.5 | −0.016 | −0.0096 | −0.9 | −0.0041 ± 0.0065 | 0.26 |
| SENS | −0.018 | −0.0120 | −0.7 | −0.0006 ± 0.0067 | 0.46 |
| ENG3 | −0.034 | −0.0144 | −1.8 | −0.004 (prior ±0.006); −0.008 with a ±0.010 prior | 0.23 |

- **Joint test of all six readings:** combined z is −4.4 if there were no selection, −3.6 under light search, −1.6 central and 0.0 under heavy search. "Neutral with no selection" is rejected; "neutral with selection" fits five of six.
- **What this means:**
  - These readings are not evidence of private harm, except ENG3, which is likely truly negative at roughly −0.004 to −0.01.
  - They are not evidence of private gain either. Through this board, a true +0.005 change on a σ ≈ 0.008 change reads about −0.005 on average.
  - The dose-response (ISO 0.75 → 1.5, PF_MID → blend) is predicted by selection alone, because the bias grows with σ. The plan's "rank-1 PF moves hurt / PF-syn dropped" is a public-split fact, not an established private one.
  - SENS's holdout gain of +0.047 is neither confirmed nor refuted.
  - On this line, only a public gain of at least about 1.2σ above the neutral expectation (≥ +0.005 for small changes, ≥ +0.01 for σ ≈ 0.008) is evidence of a real improvement. Changes need offline or holdout evidence instead.
- **Caveat:** public readings alone cannot separate selection from mechanical coupling. The coupling is that the fragment check's neural thresholds were tuned on the published saner's score distribution, so swapping the neural term could cause real private harm. ISO acts after the final layer, with no threshold coupling, and its readings fit the selection model. For the neural-term variants (SENS, PF blend, PF_MID) coupling remains possible.

Everything is in `/tmp/claude-0/-home-user-claude-code-tests/09a15150-bdf1-5786-8fb7-e9929f1c29ca/scratchpad/selopt/`; the scripts below are all there:
- `sim.py` (simulation), `summarize.py`, `neutral.py`, `biastab.py` (neutral bias vs σ), `posterior.py` (table in section 5), `analytic.py` (noise and best-of-N), `ceiling.py` (oracle ceiling), `cmpsub.py` (visible-12 diffs)
- Results: `sim_K*_N*_S*_R*.json` (8 runs: K = lineage steps, N = variants per step, S = tweak size, R = replicates); `field_oct28.json` (Oct 28 notebook scores)

## Analysis B (mechanics)

The fragment check is the only rule that compares absolute neural values, and its absolute cos/fp limits probably aren't what shifted the leaders. PF blend's loss splits about evenly between rank 1 and ranks 2-25; the cleanest test for the seed ensemble is a ranks 2-25-only build. eng3 is a different mechanism: it changes the frozen pipeline's own candidate lists, not any neural threshold.

I ran no notebook code. I parsed pubm cells 10/11/16/17, the SENS cell, cell 8 MODULES, the frozen cells 41/45/48/49/51-53 and the eng3a frozen diff, and ran three scripts of our own on the commit outputs.

## 1. Which rules can change the final lists, by rank

From pubm cells 13 and 16, the submitted list is built like this:

- **Rank 1** is one of three things:
  - the frozen baseline leader b1;
  - the V1/V2 leader (neural weight 0.25), which differs from b1 only when b2 is neural #1 and b1 is neural rank 18 or worse (from 1/(3+b) + 0.25/(3+n));
  - the fragment check's winner `p[0]`.
- **Rank 2** is the old leader, pinned there after a fragment-check move. Otherwise it is the top of the midx re-sort.
- **Ranks 2-25** are the midx re-sort by 1/(3+br) + 0.75/(3+nr) + priors, which uses neural ranks only. The CUSTOM w_mid has no effect: `rest` is fully re-sorted.
- **Fragment-check swaps and tail are discarded.** `pxm_merge` reads only `p[0]`, so the adjacent swaps and tail changes the fragment check makes never reach the submission.

**Fragment-check leader gates** (`decision`, cell 10 `_post`):
- molecule gates: lib < .87 and at least 3 spectra;
- contender: V2 positions 2-6, same formula as the leader;
- fragment scores: frag ≥ .27, margin ≥ .095, old margin ≥ .028, energy votes ≥ .70;
- **neural margin ≥ .23** of the list's neural-score std (relative, so scale-free);
- **Δneural_cos ≥ +.003 and Δneural_fp ≥ −.012** (the only absolute values);
- support (Engine 2, PubChem or main rank ≤ 5) or simulator vote ≥ 1, and simulator vote not −1.

The absolute cos/fp gates probably don't bind. Since neural_score = 10·cos + 2·fp, the .23 margin only fails to dominate the cos gate if the list's score std is below .13. The relative margin gate is the one that matters.

## 2. What the 12-molecule commit runs show

All 12 visible molecules have a library match ≥ .87 (`lib_hit_molecules` 12/12), so fragment-check leader moves are blocked in every variant. In every merge report, `leader_moves`, `old_leader_pinned`, `outsider_fills` and `dedup_swaps` are all 0. The fragment check's own changes (5/4/5/4/3 rows; 6/5/5/5/5 adjacent swaps for pubm/pfall/sens/eng3a/eng3b) are all discarded.

The only rank-1 event is a V1 flip on **m_068354**:
- pubm flips the baseline leader, which is also Engine 2 #1 and pre-fusion #1.
- Matching all 6 test spectra to train.parquet gives truth DHWCXOPDOJDBNV, which is that baseline leader. So pubm's flip is wrong.
- pfall, sens, sens-wm100 and eng3a-sens remove the flip. Visible MRR goes 0.917 → 0.958; every other variant stays at 0.917.
- The visible set therefore points the opposite way from the LB, and tells us nothing about leader moves.

| | pfmid | pfall | sens | eng3a | eng3b |
|---|---|---|---|---|---|
| neural top-1 changed | 0 | 6/12 | 6/12 | – | – |
| rank corr with saner | – | .65 | .957 | – | – |
| final rank 1 changed vs pubm | 0 | 1 | 1 | 0 | 0 |
| final rank 2 changed | 3 | 4 | 3 | 0 | 0 |
| mean \|Δposition\| | 1.34 | 1.35 | 1.01 | 1.22 | 0.83 |
| candidate sets changed | 0 | 0 | 0 | 12/12 | 11/12 |
| new candidates | 0 | 0 | 0 | 36 | 19 |

pfmid and pfall lists are identical except m_068354 positions 1-2.

## 3. Each perturbation

### PF blend (−0.014)
- **What changes:** neural_score becomes 0.5 z(saner) + 0.5 z(PF) everywhere, so the V1 flip rule, the V2 contender pool and the .23 margin gate all move, as do midx neural ranks and outsider order. neural_cos and neural_fp stay saner.
- **Exact decomposition:** given the same leader, pfall's ranks 2-25 equal pfmid's (same z-blend in `_mid_neural_score`; checked on the visible lists).
  - pfmid (−0.007) is the ranks 2-25 part. It is a rank-only change with no thresholds.
  - pfall − pfmid (−0.007) is the rank-1 part.
- **Simulation:** in my toy seed-noise simulation, a PF-like score (corr .70 with saner, top-1 change .56) flips **29%** of fragment-check gate decisions. The pass rate barely moves (.32 → .35).
- **Most likely cause:** the leader-move gate was designed as "fragment evidence and neural agree". PF correlates .27 with the fragment score and is trained on GLACIER-simulated spectra, so blending it in weakens the neural side's independence. PF also chooses which candidates sit in V2 positions 2-6.
- **Minimal variant:** columns mode, i.e. pfmid. That still lost −0.007, so no PF variant avoids the loss; keep PF dropped.

### ISO (−0.005 at 0.75, −0.016 at 1.5)
- **No neural thresholds involved.** ISO re-sorts the leader-formula group, rank 1 included, by z(−log rank) + α·z(PF), after the final layer.
- **Rank-1 takeover thresholds** (n = 25; z(−log r) is 2.84, 1.99 and 1.49 for ranks 1-3):

  | α | rank 2 takes rank 1 if its PF z leads by more than | rank 3 takes rank 1 | share of random same-formula pairs swapping at rank 1/2 |
  |---|---|---|---|
  | 0.75 | 1.13 | 1.79 | about 21% |
  | 1.5 | 0.57 | 0.90 | about 34% |

- **What it overrides:**
  - a fragment-check move, whose new leader shares a formula with the pinned old leader at rank 2;
  - the Engine 2 rank-2 item;
  - the midx order inside the formula group.
- **Minor gap:** ISO's library skip is .9, while the fragment check and midx use .87, so ISO alone acts on molecules in [.87, .9).
- **On the visible molecules,** the dry run with the library skip off changed 8/12 rows and moved 64/79 of 300 candidates. Top-1 changed 0 because 96 training structures, the leaders among them, are protected.
- **Most likely cause:** PF double-counts evidence the base already has. The offline B0gf simulation predicted −0.007 at α .5 and −0.022 at α .75. This is not threshold coupling.
- **Minimal variant:** freeze ranks 1-2 (`ISO_FREEZE_LEADER` plus rank 2). It is still expected ≤ 0, since PF in ranks 2-25 already lost.

### Seed ensemble (SENS, −0.018)
- **What changes:** neural_score, neural_cos and neural_fp all become the 5-seed means. Rank-1 exposure is the same as PF's (6/12 neural top-1 changes in both, and the same visible rank-1 change). Ranks 2-25 also move, through midx neural ranks.
- **Simulation (`gate_mc.py`):** a model that reproduces the observed seed statistics (seed corr .898 vs .886-.910 observed; seed top-1 agreement .37 vs .25-.50; ensemble corr .951 vs .957; ensemble top-1 change .44 vs .50) gives:
  - fragment-check margin-gate pass rate .32 → .24 (averaging shrinks noise-driven isomer margins about 2× relative to the list std, which itself shrinks only about 4%);
  - **23% of gate decisions flip**;
  - single-model outlier flips disappear, as on m_068354.
- **The absolute cos/fp gates keep their units under averaging and probably don't bind.** A variant that leaves neural_cos and neural_fp unchanged would therefore be nearly identical to SENS, so it can't tell the causes apart.
- **The rank-1 vs ranks 2-25 split can't be read from the LB:**
  - scaling pfmid by reorder size gives about −0.005 for ranks 2-25 and −0.013 for rank 1;
  - scaling by gate-flip rate against PF gives about −0.006 for rank 1 and −0.012 for ranks 2-25.
- **Minimal variant: SENS-mid.** Put the ensemble in a separate column read only by midx neural ranks; leave neural_score/cos/fp for the consensus and fragment check. Rank 1 is then identical to pubm by construction, as pfmid was for PF.
  - If SENS-mid lands ≥ −0.004, the loss was in the rank-1 gates (mechanical).
  - If it lands near −0.018, ranks 2-25 regress too, which points to selection.
- One deterministic reading covers only 2-4 molecule events (a rank-1 hit is worth .008; a 1↔2 swap .004). A change of equal quality flips about a quarter of gated rank-1 decisions, so expect about ±0.01 spread per reading.

### eng3 (−0.034)
- **No neural thresholds involved.** At the end of cell 39, `ENG` becomes RRF(Engine 2, ours) at w = 1, so every Engine 2 reader sees the merged list: the ICE/GLACIER union (cell 41), the fusion (cells 45, 49, 52), the safe tail (48), the fragment gap (51), the rank-2 overlay (53), the exported `engine2_rank` support, and the outsiders.
- **Merged-list numbers** (`eng3_lists.py`: local 12-molecule run merged with pubm's Engine 2 lists; same displacement counts as the Kaggle commit, 143 and 82):
  - Engine 2 rank → merged rank, median (max): 3→3 (7), 4→5 (9), 5→8 (11), 6→9 (14), 10→14 (21).
  - Engine 2's top 5 stay in the merged top 5 in 42/60 cases at w = 1, 54/60 at w = 0.5.
  - 21/60 of Engine 2's top-5 entries are missing from our top 40.
  - Our identities take 143/480 (30%) of the Engine 2 top-40 slots; the ICE union drops 814 → 805.
  - The rank ≤ 5 support is lost for Engine 2's #5 in 12/12 molecules, #4 in 4/12, #3 in 2/12.
  - Fusion vote ratio for Engine 2's items: 0.73-0.91 at ranks 4-10 (≥ 0.92 at w = 0.5). These are library molecules where the engines agree, so on hidden molecules the dilution should be larger.
- **When our top pick replaces Engine 2's top-1:**
  - at w = 1: whenever our #1 sits at Engine 2 rank r and we rank Engine 2's #1 below r, or not at all;
  - at w = 0.5: only for r = 2 with our rank of Engine 2's #1 ≥ 4, r = 3 with ≥ 10, or r = 4 with ≥ 26.
- **Most likely cause:** Engine 2's unique picks are demoted and displaced in the Engine 1 + Engine 2 fusion. That fusion fixes the 25-candidate set, the leader b1 (final rank 1 in most molecules) and the baseline ranks that every later layer weights.
  - Engine 2 is the critical component: null Engine 2 cost 0.085, and ours recovers 0.041 of that.
  - eng3a's −0.034 is 77% of the pubm − pube1 gap of 0.044.
  - The rank-2 overlay (historically worth +0.002) and the support gate are secondary.
- **Minimal variants:**
  - **(a)** leave Engine 2 untouched for every reader and add our list as a third, low-weight fusion term (w ≤ 0.3, in cells 45/49/52 plus the ICE union only), so Engine 2's unique picks keep their score;
  - **(b)** eng3b (w = 0.5), already built and verified, as the dose control;
  - **(c)** eng3a with the rank-2 overlay and the `engine2_rank` export reading `ENG3_ENGINE2`, to measure the overlay and support channels in isolation.

## Files
All in `/tmp/claude-0/-home-user-claude-code-tests/09a15150-bdf1-5786-8fb7-e9929f1c29ca/scratchpad/anaB/`:
- `scripts/vis_truth.py`: train-spectrum truth for the 12 visible molecules and MRR per variant
- `scripts/eng3_lists.py`: Engine 2 vs merged-list rank shifts
- `scripts/gate_mc.py`: the seed-noise and PF-blend simulations of the fragment-check gate
- `pubm_c*.py`, `c10__post.py`, `sens_c12.py`, `frozen/c*.py`, `eng3a_frozen.ipynb`: the extracted cells I read

## Analysis C (hidden shift)

The 5-seed saner ensemble's holdout gain should not have been expected to reach the hidden set. Two reasons. First, the holdout is mostly a different chemistry from the hidden test. Second, even if the gain transferred perfectly, it would be worth only about +0.001 in pubm, which is a quarter of one public molecule moving between rank 1 and rank 2 (0.004).

Calibration in the unit or scale sense is not the cause. Every neural value pubm reads is either rank-based, sign-like or normalised by the list's standard deviation, and SENS averages cos and fp in their own units. The part the rank fusion cannot absorb is the rank-1 decisions. Those were tuned on the public split with the published model's outputs, and averaging changes which leaders move.

The scripts are in `/tmp/claude-0/-home-user-claude-code-tests/09a15150-bdf1-5786-8fb7-e9929f1c29ca/scratchpad/anaC/`: comp.py writes `val1000_strata.parquet`; hl_seed.py writes `hl_seed.out`; listdiff.py, flips.py and truth2.py cover the 12 visible molecules.

**1. The holdout is not the hidden test** (988 of the 1,000 sanpier validation queries joined to train_meta)
- **How the holdout is built:** a random 10% identity split of train.parquet (its own audit says "not scaffold-disjoint"). Each query has the truth plus 255 training identities as decoys: same-formula first (median 20, none for 8.4% of queries), then nearest mass.
- **Holdout composition:**
  - enveda-180 (Enamine, timsTOF): 66.5%
  - COCONUT: 10.3%, none of it timsTOF
  - timsTOF natural products: 0.4% (4 queries)
- **Training set:** 66.3% Enamine identities, 9.1% COCONUT. Only 1,401 of 1.155M timsTOF spectra are COCONUT structures.
- **Hidden test:** all timsTOF. LB-implied mix: training structures 0.18, COCONUT 0.29, Enamine ~0.056, other 0.03, unreachable ~0.44. The holdout's main stratum is about 6% of the hidden set; the hidden set's main chemistry is 0.4% of the holdout.
- **Published model by stratum:**

| stratum | n | MRR | top-1 |
|---|---|---|---|
| Enamine | 657 | 0.508 | 0.364 |
| other, not COCONUT | 229 | 0.462 | – |
| COCONUT | 102 | 0.404 | 0.265 |

  Natural products are its weakest stratum. How the +0.047 splits across strata is not measured.
- **Decoys:** all are training identities, so the holdout cannot see the training-membership effects of hidden lists, which are mostly COCONUT-not-train and PubChem candidates.

**2. Where the gain sits versus what pubm uses**
- **The gain is mostly at rank 1:** of the +0.0471, top-1 accounts for +0.043 (0.344 to 0.387, 91%). Ranks 2-25 add +0.004 (0.1404 to 0.1445); recall@25 rises +0.027.
- **pubm guards rank 1:**
  - **Rank fusion:** with leader weight 0.25 at offset 3, only baseline #2 can take rank 1, and only if it is neural rank 1 while the leader is neural rank 18 or worse out of 25.
  - **Fragment check:** contenders come only from V2 ranks 2-6 and must pass value gates.
  - So the top-1 gain mostly enters as a reorder of ranks 2-25 (weight 0.75).
- **Size of the neural term on public:** 0.465 (V0) to 0.466 (V1) to 0.472 (0.75 order), about +0.007 in total. A model 10-12.5% better gives about +0.001.
- **Why the prediction was too high:** the pre-registered +0.006 ± 0.010 used our pipeline's conversion factor (0.2 × 0.6). In our pipeline PeakFormer carries most of the weight; in pubm the neural term is a small share.

**3. Our own transfer evidence**
- **Hidden-like validation, our seed pair** (v18c / v18cr1 stage 1, same rows; per-query Spearman 0.929, top-1 agreement 0.904). Ensemble minus mean member:

| | all | class 1 | COCONUT | mix-weighted |
|---|---|---|---|---|
| simulated (train-like) | +0.0019 [-0.0020, +0.0064] | +0.0012 | +0.0024 | +0.0009 |
| hidden-like (documentation transplanted) | +0.0014 [-0.0007, +0.0035] | – | +0.0017 | +0.0005 |

  - **LB:** the full score was +0.0055 (0.342 vs 0.339 / 0.334), inside seed noise. The COCONUT probe was -0.006 (0.207 vs 0.215 / 0.211).
  - **Reading:** the hidden-like transform shrinks the gain by about 30%.
  - **Caveat:** this is a weak analogue. The saner seeds agree on top-1 only 0.41-0.47 of the time (Spearman 0.877), so their ensemble gain is about 25 times larger.
  - **Limitation:** run without the population-prior emulation, because its tables are keyed to pool v4.
- **Offline-to-LB history:** gains measured on hidden-like COCONUT probes transferred into our own pipeline (LB/offline 0.49 for PF-d to PF-b, ~1.4 for the pilot synthetic run; full/spectral 0.16-0.21). Every gain measured on a train-distribution holdout, or put into the public lineage, landed at or below zero:
  - long-trained saner (holdout 0.484 to 0.718): -0.019, or -0.011 with the fragment check off
  - 5-seed saner: -0.018
  - channel re-ranker (+0.18 offline): -0.003 to -0.010
  - FP-b: -0.017
  - token 3: -0.005 full (+0.010 on the COCONUT probe)
  - PF-syn in pubm: -0.005 to -0.016
- **The 12 visible molecules** (truths found by exact spectrum match to train.parquet; all are enveda-180 copies, i.e. the holdout's own distribution):
  - V0: MRR 0.9583. With the published model (V1, V2, pubm) it drops to 0.9167, because its rank-fusion flip demoted the correct leader of m_068354 (DHWCXOPDOJDBNV, rank 1 to 2).
  - SENS, pfall and sens-wm100 undo that flip: 0.9583.
  - So on train-distribution molecules SENS gains +0.042, as the holdout predicts; on hidden public molecules it loses 0.018. Same code, opposite sign.

**4. What the fusion absorbs and what it does not**
- **Rank only, scale fully absorbed:** the V1/V2/CUSTOM consensus, the midx final layer `0.75/(3+nr)`, and the outsider fill order.
- **Reads values (fragment check on V2 only):**
  - leader move: `Δscore/sd ≥ .23`, `Δcos ≥ +.003`, `Δfp ≥ −.012`
  - adjacent swap: `score ≥ prev − .08·sd`
  - tail: `Δscore ≥ .22·sd`, with cos and fp both higher
- **No unit break:**
  - Raw mean equals z mean on the holdout (0.53148 vs 0.53145).
  - `neural_std` is left un-averaged, but nothing in pubm reads it.
- **What does change is the noise:**
  - **Model fit:** member-vs-published Spearman on the visible lists is 0.886-0.910. An equal-correlation model with r ≈ 0.90 predicts 0.959 for the ensemble against the published model; observed is 0.957.
  - **Effect on scores:** the list's standard deviation shrinks to about 0.96×. Noise in a candidate-vs-leader difference falls from 0.45 to 0.21 list SDs.
  - **Effect on gates:** a contender with no real consensus margin passes the 0.23 gate 30% of the time with one model, 14% with the ensemble. At a margin of 0.5 it is 73% vs 92%.
  - **Consequence:** thresholds tuned on a noisy single model become sharper consensus gates. Averaging removes seed noise but not bias the seeds share, and on timsTOF natural products the shared bias is where the model is weakest.
  - **Visible-run trace:** adjacent swaps went 6 to 5, and the neural top-1 changed in 6 of 12 molecules.
- **Lineage evidence that neural influence at rank 1 loses on hidden data:**
  - leader weight 0.5 scored 0.448 (-0.024); 1.0 scored 0.359
  - the fragment check was worth +0.005 with the published saner model but -0.008 with the long-trained one
- **Rough split of the -0.018 (speculative):**
  - SENS moved ranks 2-25 less than pfmid did (13.8 vs 17.0 positions per list; Spearman 0.971 vs 0.957), and pfmid cost -0.007. So ranks 2-25 plausibly cost about -0.005.
  - That leaves about -0.013, roughly 3 net leader losses, through the rank-fusion flips, the V2 fragment check and pinning the old leader.
  - Both variants that fixed the visible leader (sens -0.018, pfall -0.014) lost on the LB.

**5. Implications**
- Do not choose the neural term on the sanpier holdout or any random train split.
- If it is retested, apply SENS only to the midx ranks-2-25 neural rank. Keep the rank-fusion leader rule and the fragment check on the published scores. Expected result: 0 ± 0.004. With no rank-1 changes, this separates public-split selection from distribution shift.
- **Missing measurement:** the per-candidate holdout scores (`saner_seeds/scores/holdout_<seed>.npz` in the casmi26-saner-seeds output) are not downloaded. With them and `val1000_strata.parquet`, the +0.047 can be split into Enamine (657), COCONUT (102) and other (229) queries. If the gain sits mainly in Enamine, the shift explanation is confirmed directly.
