# Class-1 trust tools (workflow wf_a232667e-c38, Oct 29; docs/plan.md "Class-1 library trust")

Analysis code written by the workflow agents, kept for later spectral-model swaps (copied from the session scratchpad;
absolute scratchpad paths inside refer to that session's working files and must be pointed at a new output dir).

- `gate_eval.py`: offline gate for list changes on the v18c stage-2 rows (fwd_v18c/val): hardness-matched as-is and
  documentation-transplant yardsticks at w1 0.37 (0.31 / 0.42), paired bootstrap, class-1 mask cap. A reject filter:
  it rejected every LB-flat / LB-negative known pair and under-predicts LB losses.
  Usage: `import gate_eval as ge; T = ge.load_val(); T['base'] = ge.stage2_scorer()(T); R = ge.gate(T, 'base', new, new_fn=fn)`.
- `redump_same.py` + `common.py` + `evalpipe.py` + `part3_eval_redump.py`: exact same-query re-dump of the v18c
  validation class-1 queries with the truth's library spectra restricted to other instruments ("xinst"); its class-1 loss
  is 74-80% of the LB's (the hidden-like class-1 yardstick). ~4 min dump + ~1 min per evaluated model.
