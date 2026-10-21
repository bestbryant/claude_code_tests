# Plan: close the "other" gap (public 0.057, ours 0.013)

**Where the 0.057 comes from** (traced, split not measured): the PubChem channel gives about 0.03-0.045, engine-2 ChEBI/LMSD about 0.005-0.01 and engine-1 generated structures about 0.003-0.01. The generated share may net to about 0, because those structures push PubChem items down. The public PubChem-only list scored 0.037 (N1=1000, no pop), against 0.006 for our documented channel. So the gap is mainly the PubChem channel: what it reaches, and where its proposals land.

**Noise.** Paired probes on the same 124 molecules move in steps of 0.008 (one top-1 hit). Probes restricted to one source isolate a component far better than full-LB A/B (±0.03).

## 1. Fix the full-tier PubChem-only probe (today, 1 h)
- **Bugs:**
  - The staged `/home/user/ds/casmi26-code/casmi/pcchannel.py:81` still truncates at cap=60000 (HEAD fixed this in 45ba523).
  - `casmi/run_own.py:127` takes `head(25)` before deduplicating keys, so stereo duplicates waste slots.
  - The PC_ONLY branch (`run_own.py:162-172`) does not exclude the pool, so train/COCONUT structures leak in.
- **Build:** in `propose_full`, dedup by key14 and drop srckeys train+COCONUT keys before `head(25)`. Keep PubChemLite/ChEBI structures, which count as "other". Then restage casmi26-code.
- **Probe:** casmi26-pcfull-only v2, compared with 0.037 and 0.006. Read the pending v1 result as a contaminated lower bound.
- **Decision:**
  - ≥0.025: go on to items 2-3.
  - ≤0.012: PeakFormer f.z is the bottleneck on the tail, so fix scoring first.

## 2. Documentation prior + pop union on the full tier (2-3 h)
- **Build:**
  - Add a per-row `pop = ln1p(SID)+ln1p(PMID)` to `FullTier`, taken from the local `/home/user/ext/pubchem/key_pop.parquet`. It is keyed by IK14, so it is already summed over stereoisomers; `cid_ik.npy` maps tier CID to key.
  - Ship pop as float16 inside the tier dataset. PubChem counts are on our allowed list, and nothing needs internet at run time.
  - In `propose_full`: n1=5000, union the 200 top-pop rows into pass 2, and score `z(fz)+z(cos)+λ·pop` with λ∈{0.25, 0.5}. Retune λ because we z-score over about 2k rows against their about 5.2k.
- **Gain:**
  - The public notebook credits the prior with +0.004 on the full LB, measured before promotion existed. It counts 2-4x in the "other" list, so expect +0.005-0.015 on the PubChem-only probe.
  - It is also required for the S/pop promotion in item 3.
- **Probe:** pcfull-only v3 against v2, paired, same day.

## 3. Gate, slots and promotion in v11l (1-2 h after item 2)
- **Build:** in `pc_channel`:
  - Exclude all pool-v4 keys, which stage 1/2 already scored.
  - Gate on lib_max<0.9.
  - Use slots 2,4,6,8,10 when rel is high, otherwise 4,8,12,16,20.
  - Promote pc[0] to rank 1 if (S>6 and pop≥5) or rel>τ.
  - Recalibrate the public thresholds (600 and 150, in FPNet units) as the same quantiles of our rel.
  - PC_PROMO (e12783b) already has the rel branch; add the S/pop branch.
- **Gain:** the public notebook credits promotion with ≥+0.024 on the full LB (cell 110: 0.417 against 0.374). Expect +0.01-0.03 if item 1 reaches ≥0.025.
- **Probe:** the full-LB list, plus the same list restricted to "other" with the fprobe-other rule (standard IK14 against srckeys).

## 4. Rank-2 insertion of an independent ranker's top-1 (6-10 h; 2-3 h if `kernels/casmi26-own-noprior` can be reused)
- **Why:** we already reach 65,388 of engine 2's 74,767 bio-other structures, so the gap is placement, not reach. The public cell 120 moves engine 2's top-1 to rank 2 unconditionally.
- **Build:** a second stage-1 LightGBM without the in_*/pc_*/pcl_* features, with class-2 rows weighted about 0.55. `casmi/overlay.py` inserts its top-1 at rank 2.
- **Gain:** 0.005-0.015 in "other". A correct rank-2 answer is worth 0.004.
- **Probe:** full-LB A/B against v11l, plus the "other"-restricted list.

## 5. Generated derivatives in the main list (8-15 h, last)
- **Build:**
  - Run NovelEngine (`casmi/novel.py`, `derive.py`) for molecules with lib_max<0.9, adding is_gen features.
  - Add class-3 simulation rows to stage-1 training.
  - Exempt generated candidates from the documentation prior, and allow at most 3 in the top 10.
- **Gain:** 0.002-0.01. Our novel-only probe scored 0.010, and generated candidates displace PubChem items.
- **Probe:** a generated-only restricted list against 0.010.

## Slots
- **Day 1:** pcfull-only v2; pcfull-only v3; v11l with the PubChem gate, slots and promotion (full LB); the same list restricted to "other"; the pending ZEN probe. The Zenodo extras are 96.7% in PubChem by IK14, so they overlap the tier.
- **Day 2:** λ/threshold variants; rank-2 A/B.

## Refuted: do not pursue
- **DreaMS mass-tail fill (cell 115):** dead code. `dreams_np_mass.npy` is in no mounted input, and the only public file with that name holds training structures only.
- **Engine-2 COCONUT copy:** 393 off-set keys, worth about 0.0002.
- **A source-blind ranker as the way to lift ChEBI/LMSD:** the premise is false. Bio-other structures are better documented than COCONUT (33% have a PMID, against 22%). The lever is the rank-2 slot.
- **Restoring the 9,379 bio-other structures our pool drops:** 7,807 of them are above 1,500 Da.
- **Derivation features, base-list popularity re-rank, PubChem tail fill:** none of them adds "other" structures.
- **Raising N1 from 1000 to 5000 alone:** panel MRR stayed at 0.51.
- **"Zenodo extras lack PubChem CIDs":** false; 96.7% are in PubChem.
- **Still unproven:** that the answers sit in the undocumented ~70% of the tier, and how much the generated channel contributes (smoke counts come from training spectra and are inflated).

## Verified mechanisms (workflow wf_a82a089e-6e1)

- **PubChem tier: an almost complete PubChem structure store (the population)** (pubchem; holds: True): The PubChem channel accounts for about 0.035 (0.025-0.045) of the 0.057. The upper bound comes from the standalone PubChem-only LB probe at 0.037; cell 101's runner docstring says the channel is "identical to the probe that scored 0.037" (run before the prior, with N1=1000). Only about 5 PubChem items per molecule enter the final list: slots 2/4/6/8/10 or 4/8/12/16/20 when lib_max<0.9, plus promotion of pc[0] to rank 1 (cell 110), then RRF with engine 2. Most of that share needs the undocumented
- **Documentation prior aligned to the tier (pc_lsid + pc_lpmid)** (pubchem; holds: True): The PubChem channel as a whole is probably the main source of the 0.057. Its output is excluded from the pool by tautomer key and counts as 'other' in the probe. As an anchor, the earlier channel without pop (N1 = 1000, top 25 by f.z) scored 0.037 on the LB when submitted as the whole list (cell101, CORE and RUNNER docstrings). The pop prior's own share of the 0.057 is at least 0.004, and my estimate is about 0.005-0.02:
- it decides which 5 of the 25 proposals take slots;
- the probe removes th
- **Two-pass spectral scoring (FPNet A+B f.z), pool exclusion, 25 proposals (probe_core2 + V17 patch)** (pubchem; holds: True): I estimate the PubChem channel's part of the 0.057 at about 0.03-0.045, which is below its standalone reach. Its standalone reach is 0.037 measured (N1=1000, no pop); 0.04-0.05 with N1=5000 and pop is plausible but unmeasured. The final list cuts that down: PubChem proposals enter only when lib_max < 0.9, only the top 5 (after the ICE/GL reorder) get slots, and fusion and the formula rerank move them further. In the restricted list they also compete with engine-1 generated derivatives and engine
- **E2 candidate bank: ChEBI + LIPID MAPS (BIO), the only real 'other' population** (engine2; holds: True): E2-route bio-other: a plausible 0.005-0.015 (central about 0.01), not measured. One molecule whose correct bio-other answer is E2's top-1 adds 0.004 at rank 2 or 0.008 at rank 1, so the range is about 1-3 hits out of 124 molecules. Some of these same structures could also arrive through the PubChem channel, so E2 and PC credit overlap. The rest of the 0.057 (about 0.04-0.05) has to come from three other routes: the PubChem channel (gated slots, rel > 150 or S/pop promotion, PC tail fill), engine
- **E2 COCONUT copy (prvsiyan coconut-casmi26-candidates)** (engine2; holds: True): About 0 (at most about 0.0002). Only the 393 off-set keys can count as "other", which is 0.09% of the copy. Scaling the COCONUT-not-train probe (0.252) by 393/435,996 gives about 0.0002. In the 12-molecule visible smoke run, 0 of the 101 E2 COCONUT entries in the top 40 came from the 393. The "other" items in E2's lists came from the ChEBI/LMSD addition: 18 of 463 entries, 13 of them in the top 25. The 89 "other" entries in the final list came from v4g or the PubChem channel, not E2. So E2's sha
- **E2 ranker and features: source-blind, no documentation prior** (engine2; holds: True): The ranker's source-blindness on its own adds about nothing to the 0.057. A documentation prior would leave the ChEBI/LMSD 'other' structures mid-pack (median documentation rank 7 of 33 in their window), so removing one is not what lifts them.

Whatever engine 2 contributes to 'other' comes from three things:
- the 74,767 ChEBI/LMSD 'other' structures in its candidate pool;
- the class-2-weighted training, which lets a candidate with no library match be top-1;
- the slot rules: cell 120 puts e2'
- **Engine-1 analogue-derivative generation (the 'G' channel, EngineCfg(generate=True))** (generated; holds: True): Generation probably accounts for about 0.008 of the 0.057, with a plausible range of 0.002-0.02; this is not measured. The lower bound is softer than claimed, because the PubChem channel's share inside 'other' is underestimated: the 0.037 measurement was taken before the popularity prior and promotion were added. The upper side is less tightly capped than claimed, because the fork's generated rows include PubChem-present non-COCONUT derivatives that our PubChem-absent novel-only probe (0.010) co
- **fe_v4 'derivation' feature family (ranking of generated and derivable candidates; adds no structures)** (generated; holds: True): The family adds 0 structures. Its contribution to the 0.057 'other' probe is bounded by the generated (G) channel's own restricted-list MRR. That has never been measured for the public lineage. The closest LB analogue is our own novel-only class-3 probe (casmi26-novel-probe v2, our engine), which scored 0.010. The derivation family is a multiplier on that channel. Offline it gave +0.07 to +0.09 MRR on class-3 simulation groups (truth removed from the pool) and ALL +0.008. So its LB share of 'oth
- **hengck23 DreaMS mass-tail fill ("mass-matched structures that fill vacant places at the end of a list")** (generated; holds: True): 0.000. The code path is not executed in the 0.451 version, or in our fork and fprobe kernels. Even the one public file with matching names (youyouwou/casmi26-dreams-np-pool) contains only training structures, so mounting it could never feed the "other" probe.
- **S0 Main-engine class-3 generation (EngineCfg(generate=True))** (merge; holds: True): About 0.003-0.010 of the 0.057, and probably at the low end. The ceiling comes from our novel-only probe (0.010), whose reach is wider than the public generator's (it also adds conjugates). Generated candidates are almost never PubChem compounds, so they can only score where the answer is novel or nearly so. In the 'other' probe they also push PubChem-channel proposals down, so their net effect on the 0.057 could be close to zero or negative. The PubChem channel (about 105M structures, cell 101)
- **S1 Base list top-60 + pool popularity re-rank** (merge; holds: True): S1 is a pass-through, not a source of 'other'. Pool candidates add about 0 to the 0.057 'other' score; at most a tiny amount from the 972 COCONUT-side pool rows (0.14% of the pool) that the probe keys call 'other'. The 'other' entries that do pass through S1 are S0's generated derivatives. Generated derivatives are pid=-1, at most 150 per molecule, built from the top 6 analogs with similarity >=0.3, at most 60 per parent, 2-step transforms, tolerance max(0.004 Da, 10 ppm) (engine.py:42-47, hooks
- **S2 PubChem-only channel (near-complete PubChem minus the pool)** (merge; holds: True): The claim of 0.035-0.050 is plausible but unproven.

The probe that restricts the list to one source compresses ranks. PC entries at slots 2-10 become "other" ranks 1-5, so 0.057 is an upper bound on what "other" adds to the full 0.451 list. The old 0.037 PC-only probe used the top 25 for every molecule. Now only the top 5 enter, after the ICEBERG/GLACIER reorder, and only when lib_max<0.9. The measured upgrades are small: +0.004 from the prior, and about 0 from N1 on the panel.

My estimate is 
