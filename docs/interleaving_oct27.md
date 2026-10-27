# Cross-source interleaving: diagnosis and plan for Oct 29 to Nov 2

## 1. Diagnosis

- **The loss is about 0.040, not 0.046.** One part is exact: P_T + P_C − F = 0.157 + 0.215 − 0.339 = 0.033, so the total loss is 0.033 + P_O. The "other ~0.013" figure included pc-only (0.006), which is a separate PubChem-channel list that v18c does not use (it sets no `PC_TIER`). Our best estimate of v18c's own other-only score is P_O ≈ 0.007 (range 0.004–0.013, never measured). That puts the total at 0.040 (0.037–0.046).
- **Almost all of it is train and COCONUT answers displacing each other.**
  - The v5w source-tiering run (olast 0.324, against probes 0.156 / 0.208) puts the train↔COCONUT loss at 0.040–0.042.
  - It also shows that "other" answers keep at least 0.008 in the full list, so the loss involving other sources is only 0–0.004.
  - v18c's gap (0.033) is the same as v5w's (0.032).
- **The hidden lists merge about 3x worse than the validation lists.**
  - Offline, the merge loses 3.9–4.4% of the restricted-list sum at every class mix. Per query that is 0.046 on class 1 and 0.032 on class 2.
  - On the LB it is 11.9% for v18c and 2.8% for the public chain.
  - Interleaving at validation level predicts a loss of only 0.009–0.020. The LB needs it 2–4.3x higher.
  - So no offline gain from a merge policy can be taken at face value.
- **The split is unknown, and it is really one number.**
  - By identity, L_T + L_C = 0.033 + M_O, so only the split between the two sides is open.
  - The model fit puts 0.019 [0.013, 0.027] on train answers and 0.021 [0.014, 0.027] on COCONUT answers. The train share of the loss is 0.45 [0.34, 0.65], and every scenario puts at least 0.008 on each side.
  - The competing hypothesis comes from the host's statement that library search reaches MRR about 0.93 on class 1. If that holds, train answers are protected and L_T ≤ 0.010.
  - Only masked probes can tell these apart. The restricted train-only probe cannot see this cost by construction.
- **The loss comes from confusion inside individual queries, not from a source-wide bias.**
  - The best global shift recovers only 14–30% of the offline loss.
  - Where a training decoy sits above a class-2 truth, it leads by a median 0.59 SD; only 23% lead by less than 0.25 SD.
  - The stage-2 margin already predicts the truth's source with AUC 0.968, and a learned gate recovers −1% of the oracle gap.
  - Class-1 truths that lose have weak library matches (library similarity median 0.62, against 0.83 for those that do not).
- **The public chain loses only 0.013** (up to about 0.02 if the CPU fprobe variant actually scores below 0.451).
  - Applied at its difficulty, our merge behaviour would lose 0.041 [0.034, 0.049], so its merge is about 3x better.
  - Its structural recipe (rankers with no provenance features, popularity added afterwards, RRF fusion) is negative for us in every eligible copy. Offline: provenance-free stage 2 −0.010, source-factorised −0.005, learned gate −0.001 to −0.004. On the LB: v17bpop −0.016.
  - Of our 0.112 gap to 0.451, interleaving is only 0.027. The rest is other-source 0.049, COCONUT within-source ranking 0.037, and train −0.002.
- **v18cw moves the loss between classes rather than shrinking it.** With C1_DUP=1:
  - RR lost on class 1 to COCONUT decoys rises 0.0456 → 0.0514;
  - RR lost on class 2 to training decoys falls 0.0279 → 0.0264;
  - within-source MRR stays flat.
  - The model predicts v18cw = 0.339 [−0.003, +0.004]. probe18cw-train cannot see the class-1 cost.

## 2. Estimated hidden class mix (public 124 molecules)

| host class | our source | share | basis |
|---|---|---|---|
| 1 | train | 0.19 [0.165, 0.225] | fit; hard bound ≥ 0.157 |
| 2 | COCONUT-not-train | 0.32 [0.26, 0.40] | hard bound ≥ 0.252 |
| 2 | other reachable (PubChem / Enamine) | 0.08 [0.06, 0.12] | hard bound ≥ 0.057; Enamine part ≈ 0.056 |
| 3, plus class 2 we cannot reach | unreachable | 0.42 [0.30, 0.50] | hard bound ≤ 0.534 |

- In host terms that is roughly 19 / 40 / ≤42%. The private 276 molecules add about ±0.03 sampling SD to each share.
- The community estimate 16/45/39 is consistent with this.
- The estimate 28/20/52 is ruled out: the public chain's own probes already need class 2 ≥ 0.309.
- The mix that matters for merge decisions is w1, the class-1 share among train + COCONUT answers: **0.37 [0.31, 0.42]** (0.42 if matched on RR mass).
  - We trained at 46% effective (v18c) and 30% (v18cw).
  - The offline break-even between v18cw and v18c is w1* = 0.54. At the plausible w1 that favours v18cw by +0.002–0.004, which the LB cannot resolve.

## 3. Experiments for Oct 29 – Nov 2 (kept candidates only)

### Branching on the Oct 28 readings

The pre-registered slot-3 rule stands: adopt v18cw only if it scores ≥ 0.347 AND probe18cw-coco ≥ 0.220 AND probe18cw-train ≥ 0.151.

| Oct 28 outcome | base | Oct 29 kernels | follow-ups |
|---|---|---|---|
| **A**: v18cw adopted | v18cw | mask18cw-train / -coco | Predict L_T = f_T × G_cw and L_C = (1 − f_T) × G_cw, where G_cw = P_T(cw) + P_C(cw) − F(cw) + 0.007 and f_T = 0.50 [0.25, 0.75]. The same branch thresholds apply. |
| **B**: 0.333 < v18cw < 0.347, or a probe misses its bar | v18c | mask18c-train / -coco | Predictions as in #1. Close the class-mix line: no v18cq, no v18c3, no masks on v18cw. |
| **C**: v18cw ≤ 0.333 | v18c | mask18c-train / -coco | Class-1 weight matters. Pre-registered lean: mask18c-train 0.135 [0.126, 0.147]. No mirror dose (C1_DUP=3) either. |

Separate readings from Oct 28 that cost no slot:
- **G_cw vs v18c's 0.033** (information only; paired SE about 0.005–0.008). G_cw ≥ 0.041 means C1_DUP=1 widened the merge loss, which is the expected sign since class-1 interleaving rises. G_cw ≤ 0.025 means it narrowed.
- **probe18cw-train < 0.151** fails the cap, so v18cw is not adopted.
- **probe18cw-coco ≥ 0.222 with v18cw < 0.347** means the COCONUT gain was eaten by the merge. No action follows.

### #1 E1: masked split probes on the base chosen above (Oct 29, 2 slots)

**What it measures.** Each kernel keeps the base's final 25-lists and replaces every off-source candidate in place with a distinct n-alkane placeholder. Positions are preserved, so the score M_s is exactly the full-list RR carried by source-s answers. Then L_s = P_s − M_s, and M_O = F − M_T − M_C. This is exact because run_own caps every list at 25 (`casmi/run_own.py:117`) and `casmi/metric.py` counts by position without de-duplicating.

**Build**

1. Change `/home/user/claude_code_tests/scripts/make_probe_kernel.py`:
   - add an optional 4th argument, `mode = sys.argv[4] if len(sys.argv) > 4 else 'filter'`, with `assert mode in ('filter', 'mask')`;
   - in mask mode, the appended cell replaces the filter line with:
   ```python
   PH = ['C' * (i + 1) for i in range(25)]   # n-alkanes C1..C25: valid, distinct, never answers
   def _mask(c):
       c = [x[0] for x in c][:25]
       return ';'.join(s if _src(s) == PROBE_SOURCE else PH[i] for i, s in enumerate(c)) or 'C'
   sub['smiles'] = [_mask(L.get(str(m)) or []) for m in sub.molecule_id]
   ```
   - also print the number of kept and masked slots.
2. Generate the kernels:
   ```
   python scripts/make_probe_kernel.py casmi26-own-s2v18c casmi26-mask18c-train train mask
   python scripts/make_probe_kernel.py casmi26-own-s2v18c casmi26-mask18c-coco coco mask
   ```
   For branch A, run the same with `casmi26-own-s2v18cw` → `casmi26-mask18cw-train` / `-coco`.
3. Confirm that `diff` against the base notebook shows only the appended cell.
4. `kaggle kernels push` both. Push the mask18c pair now, because branches B and C (≈ 80% likely) both need it. Generate the mask18cw pair now but push it only on branch A. Push both before any new casmi26-code version: the only casmi/ change since the v18c commit is the off-path `OWN_SPEC_SRC=cotr` branch.

**Checks before submitting** (`scripts/safe_submit.sh` passes masked lists trivially, so it is not a gate here)

- (a) `own_lists.json` from `kaggle kernels output` must be byte-identical to `/home/user/work/kout/casmi26-own-s2v18c/own_lists.json` (or to the v18cw v1 output in branch A).
- (b) Every row has min(25, len(L[m])) entries, with on-source candidates at identical positions.
- (c) The log shows GLACIER 371/400 and no Traceback.
- Any kernel not COMPLETE and verified by 00:05 is left unsubmitted.

**Cost.** 2 Kaggle CPU sessions of about 62 min each, plus the hidden reruns; about 20 min local; 2 LB slots. A single-probe version is rejected: mask-train alone cannot tell placeholder collapse from fully protected class-1 answers.

**Pre-registered prediction** (branch B, 80% intervals)

| quantity | prediction |
|---|---|
| mask18c-train | 0.139 [0.128, 0.150] → L_T 0.018 [0.007, 0.029] |
| mask18c-coco | 0.194 [0.184, 0.204] → L_C 0.022 [0.012, 0.031] |
| M_O | 0.007 [0.002, 0.013] |
| D = L_C − L_T | +0.004 [−0.015, +0.020] |

- As a population quantity D has SE about 0.012 (SE(L_T) ≈ 0.007, SE(L_C) ≈ 0.010); LB rounding adds ±0.001.
- If class-1 answers really are protected, expect mask18c-train ≥ 0.148 and mask18c-coco ≤ 0.185.

**Rules**

- **VOID** (fix and rerun; no reading) if any check fails, or if M_T + M_C > F + 0.001 (placeholder collapse or de-duplication), or if M_T + M_C < F − 0.016 (a source-key or metric mismatch).
- **COCONUT-heavy**: L_T ≤ 0.010 and L_C ≥ 0.028 (P ≈ 0.18). Run #2.
- **Train-heavy**: L_T ≥ 0.025 (P ≈ 0.2). Run #3 offline only.
- **Otherwise (balanced, P ≈ 0.6)**: close the soft source-shift family (A/B/G) for good, with no LB slot, and close the interleaving line for this window.
- **If valid and M_T ≤ 0.150**: from then on, mask-train replaces probe-train as the class-1 cost cap of any change that touches interleaving. The cap is mask-train ≥ parent − 0.006 (paired SE 0.004–0.006).

### #2 Conditional on COCONUT-heavy: library-gated COCONUT prior G(δ 0.3, τ 0.8) on the base (Oct 30, 1 slot)

**What it does.** In lists whose best library match is below 0.8, add 0.3 × the per-list SD of the stage-2 score to COCONUT-not-train candidates. δ and τ are fixed now: no δ 0.2, no δ 0.45, no τ 0.9.

**Build**

1. `casmi/run_own.py` `stage2()`: add env `SRC_COCO_LIBGATE='d,tau'` (unset = off). Right after `f['z'] = rk2.predict(...)` and before the sort:
   ```python
   if lg and len(f) > 1 and float(f.lib.fillna(0).max()) < lg[1]:
       f.loc[((f.in_coco == 1) & (f.in_train == 0)).values, 'z'] += lg[0] * (float(f.z.std()) or 1.0)
   ```
   This uses pandas std (ddof=1) over the top-30 frame. Add the number of gated lists to the existing `stage 2:` print.
2. Unit check on `fwd_v18c` val (load with `train_stage2.load`, predict with `stage2_v18c`): it must reproduce class 1 −0.0043 / class 2 +0.0074.
3. Commit, then `scripts/push_code.sh --push`.
4. Build the kernel:
   ```
   python scripts/make_own_kernel.py casmi26-own-s2v18cg ret_ranker_v18c stage2_v18c POOL_DS=casmi26-pool-v18c OWN_PF=pf_model_b.pt OWN_FP=fp_model_v1r.pt LIB_CAP=10 SRC_COCO_LIBGATE=0.3,0.8
   ```
   On branch A, use `ret_ranker_v18cw stage2_v18cw` instead.
5. Verify 400/400 lists, GLACIER 371/400, no Traceback, and that every list where the gate did not fire is identical to the base v1 output.

**Cost.** About 30 min local, 1 casmi26-code version, 1 CPU session of about 1 h, 1 slot.

**Prediction.** dF +0.004 [−0.004, +0.012] on v18c; +0.003 [−0.006, +0.012] on v18cw.

**Rule**

- dF ≥ +0.005: adopt; base + libgate becomes the pick-#1 candidate.
- dF ≤ 0: close A/B/G and add the family to the STOP list.
- In between: no adoption, the family closes, and there is no δ step.

### #3 Conditional on train-heavy: class-1 library trust (offline only; no slot unless it clears)

- Allowed work: protecting library-supported train candidates, and matched-score arbitration (the C4 family reopens only here).
- Offline gate:
  - at least +0.005 at w1 0.37 under both the as-is and the documentation-transplant yardsticks;
  - measured on hardness-matched lists (within-source MRR about 0.84 for train, about 0.65 for COCONUT).
- If it clears: 1 slot, held to the +0.008 rule plus the mask-train cap. Expect no LB slot before Nov 2.

### Slot calendar

| date | interleaving slots | content |
|---|---|---|
| Oct 29 | 2 | #1 (3 slots stay free for other lines) |
| Oct 30 | ≤ 1 | #2, COCONUT-heavy branch only |
| Oct 31 – Nov 2 | 0 | unless #3 clears its offline gate |

## 4. What not to do

- **No unconditional source shift** (G, B, or A on v18c), no v18cw G replicate or 2x2 factorial, and no E2+E3 bundle.
  - G(0.3, 0.8) is the in-sample best of about 287 grid points. The family's out-of-sample gain is about 35% of in-sample (+0.0014 at w1 0.30).
  - Unconditionally it predicts dF +0.000 [−0.009, +0.009], against a paired SE of about 0.005.
  - Post-hoc list policies are 0 for 6 on the LB, each predicted positive (misses of −0.004 to −0.032).
- **No re-picked δ or τ after the fact.** That covers δ 0.45, δ 0.2 on v18cw, and τ 0.9 (which turns negative at w1 ≥ 0.41).
- **No provenance-free per-source COCONUT ranker.**
  - Its offline sign flips with the yardstick: −0.011 as-is, +0.011 under the transplant; with popularity added, +0.018 / −0.006.
  - At LB scale |dP_C| ≤ about 0.004, against a probe-coco paired SE of 0.009–0.013.
- **No masked pair on v18cw as a class-mix readout** unless v18cw becomes the base. Paired SE is 0.004–0.006 for dM_T and 0.009–0.013 for dM_C, so the C1 thresholds sit at 0.5–1.5 SE.
- **No class-mix dose retrains** (v18cq with C1_SUB=0.5, mirror v18c3 with C1_DUP=3).
  - Their trigger would fire on noise, and w1 ≈ 0.37 already lies between the two trained mixes (30% and 46%).
  - P(clearing the bar) is about 0.1, at the cost of a stage-D rebuild, a GLACIER session and 3 slots.
  - Close the class-mix line after Oct 28 unless v18cw is adopted.
- **No v18cx** (stage 2 with source-context features). Offline it nets 0 ± 0.004 and flips sign with the class weight (−0.004 on the v18cw base).
- **No learned cross-source arbitration, source-factorised stage 2, or provenance-free rebuild.** All are −0.001 to −0.024 offline.
- **No tiering or slot gating** of out-of-pool sources (−0.008 to −0.018 on the LB), no POP_BOOST (−0.016), no post-hoc same-formula GLACIER fusion (−0.002 to −0.019 offline), and no RRF or rank-2 insertion with our correlated engines.
- **Do not use probe-train as the class-1 cost cap** of any merge change; it is blind to that cost by construction.
- **Do not quote 0.046 as the interleaving loss; use 0.040 = 0.033 + P_O.** Do not compare absolute validation MRR (~0.83) with LB numbers.