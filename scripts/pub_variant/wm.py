"""Fusion-weight options of scripts/make_pub_variant.py (ours): CUSTOM_W_TOP, CUSTOM_W_MID and MIDX_NEURAL_W.

The ranks 2-25 neural weight sits in two places of the pub variants:
  consensus    cell 1 CUSTOM = dict(w_top=0.25, w_mid=0.75, rescue=0), read by build_consensus (cell 9) into the pinned
               consensus.py: rank 1 by 1/(k + b) + w_top/(k + n), ranks 2-25 by 1/(k + b) + w_mid/(k + n) (k = RRF_OFFSET
               3, b the baseline rank, n the neural rank) -> variants/submission_CUSTOM.csv;
  final layer  the midx cell (cell 16; 17 in SENS builds) re-sorts ranks 2-25 of that CUSTOM order (every listed
               candidate) by 1.0/(3.0 + br) + 0.75/(3.0 + nr) + the documentation prior, with the literal 0.75; CUSTOM's
               w_mid then decides only exact ties of that score (the stable sort keeps the CUSTOM order; dyadic ties such
               as 1/4 + 0.75/12 = 1/8 + 0.75/4 with equal priors do occur) and the answer when the final layer fails
               (restored_custom_after_error submits the CUSTOM file); the dedup backfill adds nothing here (every CUSTOM /
               V2 candidate is already in the merged list), and rank 1 keeps w_top.
Options (generator arguments, not cell-1 KEY=VALUE settings; each a finite float in [0, 2]):
  CUSTOM_W_TOP / CUSTOM_W_MID (defaults 0.25 / 0.75): a value other than the default rewrites the CUSTOM line of cell 1
    and nothing else in cell 1 (the default value leaves the build byte-identical);
  MIDX_NEURAL_W (default: absent = the midx literal 0.75): when given (any value, 0.75 included) cell 1 gets the line
    MIDX_NEURAL_W = <w> after PF_MID_W, and the final layer reads that global in place of the literal (one expression;
    MIDX_NEURAL_W = 0.75 is midx exactly: the same float division) and names it in its 'Final layer:' log line.
With any of them away from the default, markdown cell 0 says so (title suffix + a note after the title) and the notebook
metadata gets casmi.ours.weights. Nothing of the notebook is executed: exact-anchor splices only.
"""
import ast
import math

W_TOP, W_MID = 0.25, 0.75                       # the template's CUSTOM weights ('verified 0.477 optimal')
MIDX_LITERAL = 0.75                             # the midx cell's ranks 2-25 neural weight
RANGE = (0.0, 2.0)
OPTIONS = ('CUSTOM_W_TOP', 'CUSTOM_W_MID', 'MIDX_NEURAL_W')
CUSTOM_LINE = 'CUSTOM = dict(w_top=0.25, w_mid=0.75, rescue=0)\n'
C1_MIDX_ANCHOR = ('PF_MID_W = 0.0                # ours: PF-syn in the ranks 2-25 neural rank of the final layer, per molecule\n'
                  '                              # (1 - PF_MID_W) z(neural_score_saner) + PF_MID_W z(pf_fz); 0 = midx exactly\n')
PF_MID_COMMENT = C1_MIDX_ANCHOR.split('\n')[1] + '\n'      # PF_MID_W's own line may be rewritten by PF_MID_W=...

# the final layer (the generator's cell 16 text: pinned midx cell 15 + the generator's MIDX splices)
HEAD_OLD = ('# cell 1, plus one hook of ours: _mid_neural_score (PF_MID_W > 0: PF-syn joins the ranks 2-25 neural rank).\n')
HEAD_NEW = ('# cell 1, plus hooks of ours: _mid_neural_score (PF_MID_W > 0: PF-syn joins the ranks 2-25 neural rank) and\n'
            '# MIDX_NEURAL_W (cell 1: the ranks 2-25 neural weight in _score_cand, midx\'s literal 0.75).\n')
SCORE_OLD = ('                    return 1.0 / (3.0 + br) + 0.75 / (3.0 + nr) + POP_W * (p_val / 20.0) + (SPEC_W if lib_hit '
             'else 0.0) * s_val + NP_W * np_val\n')
SCORE_NEW = ('                    # ours: MIDX_NEURAL_W (cell 1) in place of midx\'s literal 0.75 (0.75 = midx exactly)\n'
             '                    return 1.0 / (3.0 + br) + MIDX_NEURAL_W / (3.0 + nr) + POP_W * (p_val / 20.0) + (SPEC_W if '
             'lib_hit else 0.0) * s_val + NP_W * np_val\n')
PRINT_OLD = ("          f'library-hit molecules {rep.get(\"lib_hit_molecules\")}, POP_W {POP_W}, NP_W {NP_W}')\n")
PRINT_NEW = ("          f'library-hit molecules {rep.get(\"lib_hit_molecules\")}, POP_W {POP_W}, NP_W {NP_W}, '\n"
             "          f'MIDX_NEURAL_W {MIDX_NEURAL_W}')\n")
FINAL_HOOK = [(HEAD_OLD, HEAD_NEW), (SCORE_OLD, SCORE_NEW), (PRINT_OLD, PRINT_NEW)]
FINAL_ID = 'busyaprime-variant-m-merge'


def parse(name, text):
    """A generator argument value -> float in RANGE, or (None, why)."""
    try:
        v = float(text)
    except ValueError:
        return None, f'{name}={text!r}: a number in [{RANGE[0]}, {RANGE[1]}]'
    if not math.isfinite(v) or not RANGE[0] <= v <= RANGE[1]:
        return None, f'{name}={v!r}: a finite weight in [{RANGE[0]}, {RANGE[1]}]'
    return v, None


def custom_changed(w_top, w_mid):
    return (w_top, w_mid) != (W_TOP, W_MID)


def custom_line(w_top, w_mid):
    changed = [f'{n} {d!r} -> {w!r}' for n, w, d in (('w_top', w_top, W_TOP), ('w_mid', w_mid, W_MID)) if w != d]
    opts = ' '.join(f'{o}={w!r}' for o, w, d in (('CUSTOM_W_TOP', w_top, W_TOP), ('CUSTOM_W_MID', w_mid, W_MID)) if w != d)
    return (f'CUSTOM = dict(w_top={float(w_top)!r}, w_mid={float(w_mid)!r}, rescue=0)  # ours: {", ".join(changed)} '
            f'(scripts/make_pub_variant.py {opts})\n')


def midx_lines(w):
    return (f'MIDX_NEURAL_W = {float(w)!r}'.ljust(29) + ' # ours: neural weight w of the final layer\'s ranks 2-25 score\n'
            "                              # 1/(3 + b) + w/(3 + n) + prior (midx: the literal 0.75; the final layer reads this)\n")


def cell1_edits(w_top, w_mid, midx_w):
    """Splices of cell 1 (after the generator's CFG and the ELIG / ENG3 / SENS edits): the CUSTOM line when the weights
    differ from the template's; the MIDX_NEURAL_W line after PF_MID_W's comment when MIDX_NEURAL_W is given."""
    edits = []
    if custom_changed(w_top, w_mid):
        edits.append((CUSTOM_LINE, custom_line(w_top, w_mid)))
    if midx_w is not None:
        edits.append((PF_MID_COMMENT, PF_MID_COMMENT + midx_lines(midx_w)))
    return edits


def final_edits(midx_w):
    """Splices of the final-layer cell (the generator's cell 16 text) when MIDX_NEURAL_W is given."""
    return list(FINAL_HOOK) if midx_w is not None else []


def custom_of(c1):
    """(w_top, w_mid, rescue) of cell 1's single CUSTOM = dict(...) assignment (ast only)."""
    hits = [n for n in ast.parse(c1).body if isinstance(n, ast.Assign) and len(n.targets) == 1
            and getattr(n.targets[0], 'id', None) == 'CUSTOM']
    assert len(hits) == 1, f'CUSTOM: {len(hits)} assignments in cell 1'
    call = hits[0].value
    assert isinstance(call, ast.Call) and getattr(call.func, 'id', None) == 'dict' and not call.args, ast.dump(call)
    kw = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
    assert set(kw) == {'w_top', 'w_mid', 'rescue'}, kw
    return kw['w_top'], kw['w_mid'], kw['rescue']


def check_config(cfg, w_top, w_mid, midx_w):
    """None, or why the cell-1 configuration cannot carry these weights."""
    if (custom_changed(w_top, w_mid) or midx_w is not None) and cfg.get('VARIANT') != 'CUSTOM':
        return (f"CUSTOM_W_TOP / CUSTOM_W_MID / MIDX_NEURAL_W act on the CUSTOM variant (the final layer reads "
                f"variants/submission_CUSTOM.csv): VARIANT {cfg.get('VARIANT')!r} needs to be 'CUSTOM'")
    if midx_w is not None and cfg.get('FRAG_ON_V2') is not True:
        return 'MIDX_NEURAL_W acts in the final layer, which runs only with FRAG_ON_V2 = True'
    if midx_w is not None and cfg.get('MIDX_NEURAL_W') != midx_w:
        return f"cell 1 MIDX_NEURAL_W {cfg.get('MIDX_NEURAL_W')!r} != {midx_w!r}"
    return None


def _desc(w_top, w_mid, midx_w):
    parts = []
    if w_top != W_TOP:
        parts.append(f'w_top {w_top!r}')
    if w_mid != W_MID:
        parts.append(f'w_mid {w_mid!r}')
    if midx_w is not None:
        parts.append(f'final layer {midx_w!r}')
    return ', '.join(parts)


def note(w_top, w_mid, midx_w, final_cell):
    """Markdown cell 0 note (blockquote paragraph, right after the title)."""
    opts = ' '.join(f'{o}={w!r}' for o, w, on in (('CUSTOM_W_TOP', w_top, w_top != W_TOP),
                                                  ('CUSTOM_W_MID', w_mid, w_mid != W_MID),
                                                  ('MIDX_NEURAL_W', midx_w, midx_w is not None)) if on)
    lines = [f'> **Fusion weights (this build: scripts/make_pub_variant.py {opts}).** The neural weights of the',
             '> rank fusion (b the baseline rank, n the neural rank among the listed candidates):']
    lines.append(f'> - consensus (cell 1 CUSTOM, cell 9): rank 1 by 1/(3 + b) + w_top/(3 + n) with w_top {w_top!r}'
                 + (f' (template {W_TOP!r})' if w_top != W_TOP else ' (unchanged)') + ',')
    lines.append(f'>   ranks 2-25 by 1/(3 + b) + w_mid/(3 + n) with w_mid {w_mid!r}'
                 + (f' (template {W_MID!r})' if w_mid != W_MID else ' (unchanged)') + ';')
    if midx_w is not None:
        lines.append(f'> - final layer (cell {final_cell}): ranks 2-25 re-sorted by 1/(3 + b) + MIDX_NEURAL_W/(3 + n) + the '
                     'documentation')
        lines.append(f'>   prior with MIDX_NEURAL_W {midx_w!r} from cell 1 in place of midx\'s literal 0.75 (0.75 = midx '
                     'exactly).')
    else:
        lines.append(f'> - final layer (cell {final_cell}): unchanged, ranks 2-25 re-sorted by 1/(3 + b) + 0.75/(3 + n) + '
                     'the documentation')
        lines.append('>   prior (midx\'s literal; CUSTOM\'s w_mid then decides only exact ties of that score and the answer if the')
        lines.append('>   final layer fails).')
    lines.append('> Everything else is the build the notes below describe.')
    return '\n'.join(lines) + '\n>\n'


def record(w_top, w_mid, midx_w, final_cell):
    return dict(custom=dict(w_top=w_top, w_mid=w_mid, rescue=0), custom_template=dict(w_top=W_TOP, w_mid=W_MID),
                midx_neural_w=midx_w, midx_literal=MIDX_LITERAL,
                custom_rule='cell 1 CUSTOM -> build_consensus (cell 9) -> consensus.leader_safe_order: rank 1 by '
                            '1/(3 + b) + w_top/(3 + n), ranks 2-25 by 1/(3 + b) + w_mid/(3 + n)',
                final_layer=(f'cell {final_cell} _score_cand: 1/(3 + br) + MIDX_NEURAL_W/(3 + nr) + prior (cell 1 '
                             'MIDX_NEURAL_W)' if midx_w is not None else
                             f'cell {final_cell} _score_cand: midx literal 0.75 (MIDX_NEURAL_W not given)'),
                options=dict(CUSTOM_W_TOP=w_top, CUSTOM_W_MID=w_mid, MIDX_NEURAL_W=midx_w))


def apply(nb, w_top, w_mid, midx_w):
    """Markdown cell 0 (title suffix + note) and casmi.ours.weights, after every other edit of the generator (ELIG /
    ENG3 / SENS rewrite cell 0 and renumber cells). Only when a weight differs from the template / MIDX_NEURAL_W given."""
    if not (custom_changed(w_top, w_mid) or midx_w is not None):
        return None
    cells = nb['cells']
    ids = [c.get('id') for c in cells]
    assert ids.count(FINAL_ID) == 1, 'final-layer cell'
    final_cell = ids.index(FINAL_ID)
    c0 = cells[0]['source']
    assert cells[0]['cell_type'] == 'markdown' and c0.startswith('# ') and c0.count('\n\n') >= 1, c0[:80]
    i = c0.index('\n\n')
    assert '\n' not in c0[:i] and c0[i + 2:].startswith('> '), 'cell 0: a one-line title, then a blockquote'
    cells[0]['source'] = (c0[:i] + f' + fusion weights ({_desc(w_top, w_mid, midx_w)})\n\n'
                          + note(w_top, w_mid, midx_w, final_cell) + c0[i + 2:])
    rec = record(w_top, w_mid, midx_w, final_cell)
    nb['metadata']['casmi']['ours']['weights'] = rec
    return dict(record=rec, final_cell=final_cell)
