"""Engine 2 clone (docs/engine2_study.md option R): settings, column names and gate thresholds. Our own code.

Shared by the vendored engine (casmi/e2/engine.py reads N_ANALOG), the replay driver (scripts/e2_replay.py), the gates
(casmi/e2/gates.py) and step 3's row packer. Importing this module never imports the vendored Engine 2 files.
"""
import os

# ---- settings -------------------------------------------------------------------------------------------------------
# Serving / row setting for the analog list length. E2:pv.py L19 default 80 (prvsiyan's rows); pubm's eng_runner L14
# served 200 (the train / serve skew of docs/engine2_study.md section 1.8). Step 3's CV picks one; the serving runner
# sets E2_N_ANALOG to it and asserts it equals the n_analog stored in the row npz.
N_ANALOG = int(os.environ.get('E2_N_ANALOG', '80'))
N_ANALOG_ROWS = (80, 200)            # both are built by the replay; the first is the primary (unsuffixed) set
ELECTRON = 0.00054858                # scripts/dump_ret_rows.py L19 (query electron-mass fix for enveda-180 '+' rows)
FIX_LIB = 'enveda-180'               # scripts/dump_ret_rows.py L74
NPX_LIB = 'enveda-np-examples'       # scripts/train_ret_ranker.py L81

# ---- bank (scripts/build_e2_bank.py on pool v18c; kernel bryantyu/casmi26-e2-bank) ------------------------------------
POOL_MD5 = '8afda40aa9324130b800523c1e5f267b'
BANK_WHAT = 'Engine 2 clone candidate bank'
BANK_EXPECT = dict(rows=1160133, nbits=6930,
                   bits_sha256='a61e372fc41091bf5c7467e08134081266262665077cb9e2c56d73e4887f3013')
BANK_FILES = ('fp', 'mass', 'meta', 'bits')       # MANIFEST outputs keys checked by bankio.check_bank

# ---- replay inputs (scripts/e2_pack_inputs.py) ----------------------------------------------------------------------
INPUTS_WHAT = 'Engine 2 clone replay inputs (docs/engine2_study.md option R step 2)'

# ---- feature blocks (E2:pv.py L249-255; E2:casmi_engine.py L299-341) -------------------------------------------------
BASE_W, LIBX_W, ANAPOL_W, FRAGX_W = 31, 8, 5, 7
ANA_DEP_BASE = (5, 6, 7, 8, 9, 10, 11, 12, 13, 26, 28)     # base columns that depend on the analog list
FP_BASE = (15, 16, 17, 18, 19, 20, 25, 26, 27, 28, 29, 30)  # base columns that use the FP model (zero when zlog is None)

BASE_COLS = [f'b{i:02d}' for i in range(BASE_W)]
LIBX_COLS = [f'lx{i}' for i in range(LIBX_W)]
FRAGX_COLS = [f'fx{i}' for i in range(FRAGX_W)]


def anapol_cols(n, primary=True):
    return [f'ap{i}' + ('' if primary else f'_a{n}') for i in range(ANAPOL_W)]


def base_alt_cols(n):
    return [f'b{i:02d}_a{n}' for i in ANA_DEP_BASE]


def feature_cols(n_analog=N_ANALOG_ROWS):
    """The float32 feature columns of e2r_<tag>.parquet, in file order (67 for (80, 200))."""
    n0, rest = n_analog[0], n_analog[1:]
    cols = list(BASE_COLS)
    for n in rest:
        cols += base_alt_cols(n)
    cols += LIBX_COLS + anapol_cols(n0)
    for n in rest:
        cols += anapol_cols(n, primary=False)
    return cols + FRAGX_COLS


ID_COLS = {'qid': 'int32', 'pid': 'int32', 'key': 'str', 'label': 'int8', 'cls': 'int8', 'truth': 'str', 'subset': 'str',
           'reach': 'int8', 'art_ret': 'int8', 'art_e2': 'int8', 'in_ret': 'int8', 'in_train_pool': 'int8',
           'in_coco_pool': 'int8', 'pf_bern': 'float32', 'pf_src': 'int8'}

# cubres reproduction/legacy-v2/ranker_lineage/FEATURE_COLUMNS.json (source_sha256 716e01b3...), zero-based
BASE_NAMES = dict(enumerate([
    'library_similarity', 'library_rank_norm', 'library_max', 'library_minus_max', 'library_positive',
    'analog_cubic_max', 'analog_cubic_rank_norm', 'analog_cubic_max_over_candidates', 'analog_cubic_minus_max',
    'analog_linear_max', 'best_analog_tanimoto', 'top_analog_tanimoto', 'analog_cubic_weighted_mean',
    'top_analog_similarity', 'log_candidate_count', 'fp_raw_zscore', 'fp_raw_rank_norm', 'fp_raw_minus_max',
    'fp_length_corrected_zscore', 'fp_length_corrected_rank_norm', 'fp_raw_is_max', 'fragment_score',
    'fragment_rank_norm', 'fragment_minus_max', 'fragment_zscore', 'library_model_agreement', 'analog_model_agreement',
    'library_best_model_rank_agreement', 'analog_best_model_rank_agreement', 'library_best_agreement_times_library_max',
    'library_vs_negative_model_rank_correlation']))
LIBX_NAMES = ['lib_mean', 'lib_mean_rank', 'lib_mean_gap', 'libsh', 'libsh_rank', 'libsh_gap', 'log1p_lib_cnt',
              'log1p_lib_cnt_gap']
ANAPOL_NAMES = ['anapol_ap', 'anapol_ap_rank', 'anapol_ap_gap', 'anapol_ap6', 'anapol_top_sim']
FRAGX_NAMES = ['frag_ad', 'frag_ad_rank', 'frag_ad_gap', 'frag_mean', 'frag_mean_rank', 'frag_mean_gap', 'frag_pol']

# ---- gates (docs/engine2_study.md section 5 step 2; spec section 5) ---------------------------------------------------
GATES = dict(
    G1_control_min_frac=0.95, G1_control_lv=0.9, G1_control_n=25,
    G2a_identity_mret=0.99,
    G2b_identity_e2=0.985, G2b_edge_da=1e-5, G2b_floor_mass=200.0,
    G2c_truth_slack=1, G2c_joined_share=0.995,
    G3_lib_c1=0.90, G3_ana_c2_full=0.48, G3_ana_c2_tc=0.50, G3_frag_c2_tc=0.20,
    G4_cpu_s_per_query=3.0,
    G5_rel=1e-3, G5_abs=1e-4,
)
# channel columns reported in the parity tables (column -> what it is)
CHANNEL_COLS = {'b00': 'library lv', 'b05': 'analog ap (cubic)', 'b21': 'MetFrag frag', 'lx0': 'lib_mean',
                'lx3': 'libsh (adduct-shifted)', 'ap0': 'anapol ap', 'fx0': 'frag_ad', 'fx3': 'frag_mean',
                'b15': 'z(f.z)', 'pf_bern': 'PeakFormer f.z'}
REF_COLS = ('lib', 'an_best', 'frag', 'pf_bern')     # ret18cu channels on the ret18cu window (e2pf)
