# F5b ICEBERG: clean-room runner plan, effort, and the "drop ICEBERG" alternative

Status 2026-10-31. Companion to `glue_spec.md` (same directory; section numbers below refer to it) and to the GLACIER
drop-in built today (`casmi/glrun_dropin.py`, `casmi/glfuse_dropin.py`).

## 1. What ICEBERG does in the eligible fork today

Where its scores act (spec §1, §9.3, §11.4):

| use | where | effect |
|---|---|---|
| PubChem-list re-order | FZ c41:1327-1337 (P1) | `rerank(ice)`, then the joint `rerank_multi([ice, gl])`, for molecules with a PubChem list and `lib_max < 0.9`; this can change which PubChem candidate the gated merge promotes |
| E006 `ice_fallback` | FZ c52:100-113 (P6.3) | ICE-only re-rank of GLACIER-blind molecules (in practice [M+Na]+ without [M+H]+) |
| E006 joint ICE + GL | FZ c52:77-91 (P6.1) | only for N1 proxy molecules (`N1_GL_MIDS`) |
| gates | c41:1328, c45:53, c49:38, c52:77 | a truthy per-molecule ICE dict gates **every** GLACIER use; `frag_gap` requires no finite ICE value |
| commit asserts | FZ c50:29-41 | the ice stage meta must say status ok, device cuda, rdkit 2025.03.6, no errors |

ICEBERG is not used for the base list (`BASE_ICE = False`), the fused list (`CELL19_ICE = False`, so it passes `{}`) or
the E004 window overlay (GL-only).

In the visible smoke (pubm commit) ICE scored 793 of 814 candidates in 11 of 12 molecules. Even so,
E006 reported `fallback_rows 0` and `joint_rerank_rows 0`, and the PubChem re-rank stats were 0. So ICEBERG had no
visible effect on that submission.

## 2. What a clean-room ICEBERG runner needs

### 2.1 ms-pred inference API (MIT, @708148c2a8eb, `src/ms_pred/iceberg/`)
- `joint_model.JointModel(gen_model_obj, inten_model_obj)`.
  - The parts are `gen_model.FragGNN` and `inten_model.IntenGNN`.
  - `JointModel.from_checkpoints(gen_ckpt, inten_ckpt)` loads Lightning checkpoints.
  - A `{hyper_parameters, state_dict}` re-save is loaded by constructing each part from its hyper-parameters (keys
    starting with `_` dropped), then a strict `load_state_dict`. This is the same pattern `glrun_dropin.load_model`
    uses for GLACIER.
- `JointModel.predict_mol(smi=[...], collision_eng=[...], precursor_mz=[0.0, ...], adduct=[...], threshold=0.0,
  device=dev, max_nodes=100, instrument=[token, ...], binned_out=False)` (MP iceberg/joint_model.py:65-200).
  - It returns `{"spec": [Tensor[rows, 2] (m/z, intensity)], "frag": [...]}` per input.
  - `instrument` is used only when both parts have `embed_instrument`.
  - With `canonical_root_smi=False` (the default), it strips stereo and round-trips each SMILES through InChI itself.
    That uses the tautomer-transform file under `chem_utils.user_cache_dir`, so that function must be redirected to
    a temp dir (spec E6; `glrun_dropin._cache_dir` already does this).
- With RDKit 2025.03, `chem_utils` uses the rdMolStandardize v2 tautomer canonicaliser (spec E7). It only matters
  inside `predict_mol`'s InChI round trip.

### 2.2 Weights
- **Source:** the upstream `msg_all` ICEBERG 2.1 checkpoints (MassSpecGym-trained, allowed by the host). They are
  linked from the ms-pred README (line 281, a Dropbox folder).
- **What we have today:** only `bryantyu/casmi26-fwd/iceberg/ckpt/gen.pt` (14.6 MB) and `inten.pt` (13.8 MB). These
  are byte copies of Ahmed's re-saves (sha deef0451…, c49e330e…). The dataset also ships his glue, so it must not
  be mounted by the eligible fork (spec §13.4).
- **Steps:**
  1. Download `msg_all` (gen + inten `.ckpt`) into a fresh scratch dir. Record the sha256 and the download date.
     Dropbox through the agent proxy is untested; the fallback is a manual download by the owner.
  2. Compare tensor by tensor with gen.pt / inten.pt: same keys, shapes and values; hyper-parameters equal apart
     from `_`-keys. Identical tensors also document the re-saves' provenance.
  3. Publish a new dataset, e.g. `bryantyu/casmi26-iceberg-clean`, with:
     - the upstream checkpoints (or our own re-save, with a manifest);
     - the ms-pred MIT notice and the MassSpecGym / ICEBERG citations;
     - **no** glue files.
  4. Add a "weights" line for it to docs/ATTRIBUTION.md.

### 2.3 Behaviour to implement (from the spec only)
Most of the runner is the GLACIER drop-in, reused unchanged:
- environment E1-E9, including the wheel-site marker, the RDKit pin and the placeholder modules;
- query side Q1-Q6;
- plans and jobs, including the SUM_CE path;
- similarity S1-S3;
- priority-ordered, all-or-nothing chunks with per-batch deadlines;
- the out.json / meta.json contract;
- the CLI shape (`<pkg> <in> <out>` plus the same options, `--bs` default 96).

The ICE-specific parts:

| rule | ICEBERG behaviour (spec) | differs from GLACIER in |
|---|---|---|
| C1 acceptance | RDKit parses with ≥ 1 atom; all elements in `VALID_ELEMENTS`; the InChI round trip of the stereo-stripped SMILES gives a parseable SMILES. Predictions are cached per **input** string (no canonical de-duplication). | no canonicalisation and no ≤ 100-atom rule; an InChI check instead |
| P-ICE | one batched `predict_mol` per (adduct, token, CE bucket) condition; precursor 0, threshold 0, max_nodes 100, binned off | the API |
| post-processing | drop intensity ≤ 0; merge identical m/z (4 dp) by **SUM**; top 100; sort by m/z; divide by the max (float32); an empty result counts as a failure (None) | GLACIER merges by MAX |
| batching | conditions in sorted order; SMILES sorted by (length, string); batch size from the length L of the last SMILES in the window: L ≤ 60 → bs, ≤ 90 → bs/2, ≤ 140 → bs/4, else bs/8 (at least 1) | GLACIER sizes batches by heavy atoms |
| failures | quarter-split retry, OOM batch reduction, environment guard (≥ 50 checked and all rejected → error) | ICE has no fail-fast-after-64 rule |
| meta | same fields without `model`, `n_feat_fail` and `bs_final`; log prefix `[ice ...]` | — |
| E11 speed patches | a mask-based `pack_padded_tensor` and a per-edge-type `GatedGraphConv.forward`, described as "same math" | optional: leave out of v1 and measure the speed first |

The score written to `ice_out.json` (`ice_ex_mean`) is defined exactly like GLACIER's (spec §8):
- Per covered spectrum ([M+H]+ / [M+Na]+, positive), the prediction is the one at (candidate, adduct, token, CE
  bucket). When the bucket is NaN, it is the sum-merged 20 / 40 / 60 eV prediction.
- The query and the prediction both go through `prep_query` (floor 0.002, top 256, entropy weights with
  w = 0.25 + 0.25·S when S < 3).
- sim = 1 − (2 S_AB − S_A − S_B) / ln 4, using the two-cursor match with tolerance max(0.01 Da, 20 ppm), clipped to
  [0, 1].
- The score is the mean over the spectra that have a prediction, or null if none has one.

### 2.4 Tests
- **T1-style units:**
  - C1 cases: stereo, InChI-hostile SMILES, invalid elements, the empty string;
  - SUM vs MAX post-processing;
  - the length-bucket batch sizes.
- **T5 golden:** `ice_work/ice_in.json` → `ice_out.json` of the pubm commit.
  - Expect the same null pattern: 793 / 814 scored, 11 of 12 molecules (m_06e13d has no spectrum).
  - Expect a numeric residual of the same kind as GLACIER's: GLACIER's local CPU run gave mean |Δ| 1.2e-3 against the
    GPU golden, because predictions depend on batch composition and device. Whether ICEBERG is batch-dependent too
    is unknown.
- **CPU speed:** ~1.35 predictions/s/thread (plan.md, Oct 18). So T5 on 4 threads takes ~5 min, which is fine
  locally. The decisive comparison is a Kaggle GPU commit run (T6).

## 3. Effort and runtime

| step | hours |
|---|---|
| runner on top of `glrun_dropin` (shared helpers imported, or a sibling module with the shared part factored out) | 3-4 |
| weights: download, tensor check, clean dataset, attribution line | 1.5-2.5 (plus Dropbox risk) |
| local tests T1 / T5 | 1-2 |
| Kaggle GPU validation in a commit run (golden diff, c50 asserts) | 1-2, plus one GPU session |
| **total** | **7-10** (fork_eligibility.md F5b said 6-10) |

Runtime:
- Glue ICE on a T4, with its speed patches: 42 predictions/s plus 50-60 s of load.
- Without the patches: unmeasured; plausibly 1.5-3x slower.
- **Commit:** 1,599 predictions against a 300 s budget, so 40-110 s; OK.
- **Rerun:** scaling the 12-molecule smoke (814 candidates, 1,599 predictions) to 400 molecules gives ~53k
  predictions. That is ~1,300 s with the patches and ~2,600-3,900 s without, against a budget of 5,400 s.
  - That fits, but with little margin without the patches.
  - Priority order (no-library-match molecules first) limits the damage if the budget runs out.

## 4. Alternative: drop ICEBERG

### 4.1 Mechanics (pipeline patches, no new science)
1. Set `ICE_SCORES = glfuse_dropin.placeholder_scores(items)` instead of calling `run_ice`.
   - This gives all-null dicts for every ICE-input molecule, so the truthy-dict gates stay open and GLACIER is used
     exactly as now (spec §9.3, §12.1).
   - Plain `{}` would silently switch GLACIER off in P1, P3, P4 and P6.
   - Covered by `test_placeholder_keeps_gl_only_behaviour`: the joint re-rank then equals the GLACIER-only re-rank.
2. Remove the ICE pieces from the cells:
   - FZ c26:105-106: the unguarded `find('casmi26-iceberg/**/ice_runner.py')`;
   - FZ c41:59-61, 104-106: the ICE package, the `fuse` import and the pre-seed from ICE_PKG.
   - The wheel site then comes from our `casmi26-glacier/wheels`. `glrun_dropin` installs it itself, keeping only the
     interpreter's cp-tag wheels.
3. FZ c50:29-41 asserts on `ice_work/ice_out.json.meta.json` (status ok, cuda, rdkit 2025.03.6). This must become an
   explicit "ICE stage disabled" branch. Writing a fake "ok" meta would misreport the run.
   - FZ c43:103-104 reads `n_mols_scored` and prints −1 when the meta is missing. That is informational only.

### 4.2 Behaviour changes (spec §12)
- **P1 PubChem list:** becomes GLACIER-only (single-scorer arithmetic, no tie grid).
- **P6 joint ICE + GL on N1 molecules:** becomes GLACIER-only.
- **P6 `ice_fallback`:** never fires.
  - In the visible test, 5 of 400 molecules have positive [M+Na]+ but no [M+H]+, and 2 more have only [M+NH4]+ /
    water-loss spectra.
  - Those 5 now go to `frag_gap` whenever FRAG_GAP has them. Before, finite ICE values kept them out of `frag_gap`.
    This is a policy change, not just a loss.

### 4.3 LB cost estimates
- **docs/fork_eligibility.md (F5b row, §2):** −0.004 to −0.008 if ICEBERG is dropped. The lineage credits:
  - ICEBERG fallback: +0.005 (older E006 lineage);
  - joint ICE + GLACIER: +0.008. This is bundled with "proxy GLACIER kept where ICEBERG scores exist", so it is not
    attributable to ICEBERG alone.
- **Bound from reach:**
  - The fallback acts only on GLACIER-blind molecules: ~1.25 % of the visible test.
  - A +0.005 LB gain from them alone would need ~+0.4 MRR on those molecules, which is implausible for a re-order
    inside formula groups.
  - The joint term acts only on N1 molecules (~0.5 % visible) and on PubChem-list leaders with `lib_max < 0.9`.
- **Noise:** one public rank-1 hit is worth ~0.008, and identical reruns differed by up to 0.006. One LB slot cannot
  measure the cost.
- **Our estimate:** central about −0.003, range 0 to −0.008, i.e. the lower half of the doc's range. Part of the
  fallback loss may be recovered by `frag_gap` taking over those molecules (sign unknown).
- **Option C (new policy, not a reproduction):**
  - Score GLACIER-blind [M+Na]+ molecules with GLACIER [M+Na]+. The runner already covers [M+Na]+ (Q1); the pipeline
    sends only [M+H]+ ("v29: +0.006").
  - This would replace the ICE fallback without ICEBERG, but needs an offline evaluation on our validation harness
    before any LB slot.

## 5. Recommendation
1. If F5a (licence grant) has no answer by Nov 7: ship the GLACIER drop-in, with ICEBERG dropped via
   `placeholder_scores` and the c50 patch, as the eligible default. Its expected cost (~−0.003) is inside LB noise.
2. Build the clean ICEBERG runner (7-10 h) only if:
   - an A/B slot pair (eligible with ICE vs without) is affordable; or
   - offline evidence shows the fallback matters on hidden-like molecules (HL-val, [M+Na]+-only queries).
3. Weights first: steps 2.2.1-2.2.2 (1.5-2.5 h) are worth doing regardless. They settle the provenance of the
   re-saved ICEBERG weights that the audit currently relies on.
