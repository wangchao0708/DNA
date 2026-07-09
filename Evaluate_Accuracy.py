#!/usr/bin/env python3
"""
Model attribution evaluation with GPU-accelerated train, test, and ablation modes.

Modes:
  train: fit bias terms on a stratified training split and select the best method on validation data.
  test: evaluate fixed configurations on independent CSV files produced by the DNA attribution scripts.
  ablation: scan noise-count and step-count grids, then report validation accuracy tables.

The public release intentionally uses placeholder paths and evaluation parameters. Fill in the
CSV paths, noise counts, selected timesteps or sigmas, score type, and bias terms before running.
"""

import pandas as pd
import numpy as np
import time
import os

# ==================== CONFIG ====================
MODE = 'test'  # 'train' | 'test' | 'ablation'

VAL_RATIO = 0.4

CSV_PATHS = [
    'DNA_SD1_results.csv',
    'DNA_SD2_results.csv',
    'DNA_SD3_results.csv',
    'DNA_SDXL_results.csv',
    'DNA_FLUX1_results.csv',
    'DNA_FLUX2_results.csv',
]

N_MAX_STEPS = None
N_NOISE = None
RANDOM_SEED = None
N_IMAGES = None
METHODS_TO_RUN = ['D']

ABL_NOISE_GRID = []
ABL_STEPS_GRID = []
ABL_SEED = None

# Test-mode CSV names are aligned with the default outputs of the DNA attribution scripts.
# Fill the remaining fields with the selected parameters from your finalized experiment.
TEST_CONFIGS = [
    {'csv_path': 'DNA_SD1_results.csv', 'n_noise': None, 'score_type': 'zscore', 'timesteps': [], 'bias': []},
    {'csv_path': 'DNA_SD2_results.csv', 'n_noise': None, 'score_type': 'zscore', 'timesteps': [], 'bias': []},
    {'csv_path': 'DNA_SD3_results.csv', 'n_noise': None, 'score_type': 'zscore', 'timesteps': [], 'bias': []},
    {'csv_path': 'DNA_SDXL_results.csv', 'n_noise': None, 'score_type': 'zscore', 'timesteps': [], 'bias': []},
    {'csv_path': 'DNA_FLUX1_results.csv', 'n_noise': None, 'score_type': 'zscore', 'timesteps': [], 'bias': []},
    {'csv_path': 'DNA_FLUX2_results.csv', 'n_noise': None, 'score_type': 'zscore', 'timesteps': [], 'bias': []},
]
# ================================================

# -- GPU backend detection ------------------------------------------------------
USE_GPU = True
xp = np
_torch_device = None

try:
    import cupy as cp
    cp.array([1.0])
    xp = cp
    USE_GPU = True
    print("[GPU] CuPy CUDA")
except Exception:
    try:
        import torch
        if torch.cuda.is_available():
            USE_GPU = True
            _torch_device = torch.device('cuda')
            print(f"[GPU] PyTorch CUDA: {torch.cuda.get_device_name(0)}")
        else:
            print("[CPU] CUDA is unavailable; using NumPy")
            USE_GPU = False
    except ImportError:
        print("[CPU] NumPy mode")
        USE_GPU = False


def to_gpu(arr):
    if not USE_GPU:
        return arr
    if xp is not np:
        return xp.asarray(arr)
    import torch
    return torch.tensor(arr, device=_torch_device, dtype=torch.float32)


def to_cpu(arr):
    if not USE_GPU:
        return arr
    if xp is not np:
        return xp.asnumpy(arr)
    import torch
    if isinstance(arr, torch.Tensor):
        return arr.cpu().numpy()
    return arr


# -- Data loading with noise-count and image-count limits ------------------------------
def load_data(csv_path, n_noise=50, n_images=None):
    df = pd.read_csv(csv_path)
    t_cols = sorted([c for c in df.columns if c.startswith('mse_t')],
                    key=lambda x: int(x.replace('mse_t', '')))
    models = sorted(df['model'].unique())

    if 'noise_idx' in df.columns:
        df = df.sort_values(['source', 'img_idx', 'model', 'noise_idx'])
    df = df.groupby(['source', 'img_idx', 'model']).head(n_noise).reset_index(drop=True)

    df_avg = df.groupby(['source', 'img_idx', 'model'])[t_cols].mean().reset_index()
    records = []
    for (src, idx), grp in df_avg.groupby(['source', 'img_idx']):
        g = grp.set_index('model')
        row = {'source': src, 'img_idx': idx}
        for m in models:
            for tc in t_cols:
                row[f'{m}_{tc}'] = float(g.loc[m, tc])
        records.append(row)
    wide = pd.DataFrame(records)

    # -- Limit image count by taking the first n_images samples per source, sorted by img_idx --
    if n_images is not None:
        wide = wide.sort_values(['source', 'img_idx'])
        wide = wide.groupby('source').head(n_images).reset_index(drop=True)

    n, n_m, n_t = len(wide), len(models), len(t_cols)
    MSE = np.zeros((n, n_m, n_t), dtype=np.float32)
    for k, m in enumerate(models):
        for j, tc in enumerate(t_cols):
            MSE[:, k, j] = wide[f'{m}_{tc}'].values
    y_labels = wide['source'].values
    label2idx = {m: i for i, m in enumerate(models)}
    y = np.array([label2idx[s] for s in y_labels])
    t_values = np.array([int(tc.replace('mse_t', '')) for tc in t_cols])
    return MSE, y, models, t_cols, t_values


# -- Stratified train/validation split ----------------------------------------
def stratified_split(y, val_ratio=0.4, random_seed=42):
    """
    Return (train_idx, val_idx).
    train: 60%   val: 40%
    """
    rng = np.random.default_rng(random_seed)
    train_idx, val_idx = [], []
    for cls in np.unique(y):
        ci = np.where(y == cls)[0]
        rng.shuffle(ci)
        n_val = int(len(ci) * val_ratio)
        val_idx.extend(ci[:n_val].tolist())
        train_idx.extend(ci[n_val:].tolist())
    return np.array(train_idx), np.array(val_idx)


# -- z-score normalization ----------------------------------------------------
def zscore_mse(M):
    out = np.zeros_like(M)
    for j in range(M.shape[2]):
        mu = M[:, :, j].mean(axis=1, keepdims=True)
        sig = M[:, :, j].std(axis=1, keepdims=True)
        sig = np.where(sig < 1e-15, 1.0, sig)
        out[:, :, j] = (M[:, :, j] - mu) / sig
    return out


# -- rank aggregation ---------------------------------------------------------
def rank_aggregate(M):
    R = np.zeros_like(M)
    for j in range(M.shape[2]):
        R[:, :, j] = M[:, :, j].argsort(axis=1).argsort(axis=1)
    return R


# -- GPU-vectorized bias search ----------------------------------------------
def grid_search_bias_gpu(scored_np, y_np, n_steps=21, search_range=None,
                         fix_idx=0, n_refine=2):
    n, n_m = scored_np.shape
    free_idx = [i for i in range(n_m) if i != fix_idx]
    n_free = len(free_idx)
    if n_free == 0:
        return float((scored_np.argmin(1) == y_np).mean()), np.zeros(n_m, dtype=np.float32)
    if search_range is None:
        ss = np.sort(scored_np, axis=1)
        if ss.shape[1] > 1:
            gaps = ss[:, 1] - ss[:, 0]
            search_range = float(np.percentile(gaps, 95)) * 3
        else:
            search_range = 1.0

    def _search(vals_list):
        grids = np.meshgrid(*vals_list, indexing='ij')
        flat = [g.ravel() for g in grids]
        nc = flat[0].shape[0]
        bg = np.zeros((nc, n_m), dtype=np.float32)
        for k, fi in enumerate(free_idx):
            bg[:, fi] = flat[k]
        sg = to_gpu(scored_np)
        bgpu = to_gpu(bg)
        yg = to_gpu(y_np.astype(np.int32))
        if USE_GPU and xp is not np:
            c = sg[xp.newaxis] + bgpu[:, xp.newaxis]
            p = xp.argmin(c, axis=2)
            a = (p == yg[xp.newaxis]).mean(axis=1)
            bi = int(xp.argmax(a))
            ba = float(a[bi])
        elif USE_GPU and _torch_device is not None:
            import torch
            c = sg.unsqueeze(0) + bgpu.unsqueeze(1)
            p = c.argmin(dim=2)
            a = (p == yg.long().unsqueeze(0)).float().mean(dim=1)
            bi = int(a.argmax())
            ba = float(a[bi])
        else:
            ba = 0.0
            bi = 0
            al = []
            for s in range(0, nc, 512):
                b_ = bg[s:s + 512]
                c = scored_np[np.newaxis] + b_[:, np.newaxis]
                p = c.argmin(axis=2)
                al.append((p == y_np[np.newaxis]).mean(axis=1))
            aa = np.concatenate(al)
            bi = int(aa.argmax())
            ba = float(aa[bi])
        return ba, bg[bi].copy()

    vals = np.linspace(-search_range, search_range, n_steps, dtype=np.float32)
    best_acc, best_b = _search([vals] * n_free)
    step = (vals[1] - vals[0]) if len(vals) > 1 else search_range
    for _ in range(n_refine):
        fv = [np.linspace(best_b[fi] - step, best_b[fi] + step,
                          n_steps, dtype=np.float32) for fi in free_idx]
        ar, br = _search(fv)
        if ar > best_acc:
            best_acc, best_b = ar, br
        step /= max(1, n_steps // 2)
    return best_acc, best_b


# -- eval_with_bias: fit on train, evaluate on train and validation -------------------
def eval_with_bias(scored_tr, scored_va, y_tr, y_va, n_steps=21):
    """
    Fit bias on train and evaluate separately on train and validation.
    Returns: (train_acc_noBias, val_acc_noBias, train_acc_bias, val_acc_bias, biases).
    """
    nob_tr = float((scored_tr.argmin(1) == y_tr).mean())
    nob_va = float((scored_va.argmin(1) == y_va).mean())
    acc_tr, biases = grid_search_bias_gpu(scored_tr, y_tr, n_steps=n_steps)
    acc_va = float(((scored_va + biases).argmin(1) == y_va).mean())
    return nob_tr, nob_va, float(acc_tr), acc_va, biases


# -- Key methods used in comparison tables --------------------------------------
KEY_METHODS = [
    'Baseline',
    'A_noBias',        'A_bias',
    'B_noBias',        'B_bias',
    'C_best_noBias',   'C_best_bias',
    'D_best_noBias',   'D_best_bias',
    'E_best_noBias',   'E_best_bias',
    'F_noBias',        'F_bias',
    'G_noBias',        'G_bias',
    'H_best_noBias',   'H_best_bias',
    'EH_best_noBias',  'EH_best_bias',
    'Overall_best',
]


# -- Method-filter helper ------------------------------------------------------
ALL_METHOD_TAGS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'EH']


def _resolve_methods(methods_to_run):
    """Normalize METHODS_TO_RUN: ['all'] -> all methods; ['A','H'] -> {'A','H'}."""
    if methods_to_run is None or len(methods_to_run) == 0:
        return set(ALL_METHOD_TAGS)
    if 'all' in [m.lower() for m in methods_to_run]:
        return set(ALL_METHOD_TAGS)
    s = set()
    for m in methods_to_run:
        mu = m.upper()
        if mu in ALL_METHOD_TAGS:
            s.add(mu)
        else:
            print(f"  [WARN] Unknown method '{m}',ignored (available: {ALL_METHOD_TAGS})")
    return s


# -- Print reusable test configuration ----------------------------------------------
def _format_test_config_entry(cfg, n_noise, csv_path=None,
                               method_name=None, comment=None):
    """
    Generate the text for one TEST_CONFIGS entry, including indentation and trailing comma.
    When csv_path is None, keep the 'CHANGE_TO_YOUR_TEST_CSV_PATH' placeholder.
    """
    lines = []
    if comment:
        lines.append(f"    # {comment}")
    lines.append("    {")
    cp = csv_path if csv_path else 'CHANGE_TO_YOUR_TEST_CSV_PATH'
    lines.append(f"        'csv_path': '{cp}',")
    lines.append(f"        'n_noise': {n_noise},")
    lines.append(f"        'score_type': '{cfg.get('score_type', 'avg')}',")
    if 'timesteps' in cfg:
        lines.append(f"        'timesteps': {cfg['timesteps']},")
    if 'low_steps' in cfg:
        lines.append(f"        'low_steps': {cfg['low_steps']},")
        lines.append(f"        'high_steps': {cfg['high_steps']},")
        lines.append(f"        'weight_low': {cfg['weight_low']},")
    bias_str = ', '.join(f"{v:.6f}" for v in cfg.get('bias', []))
    lines.append(f"        'bias': [{bias_str}],")
    lines.append("    },")
    return '\n'.join(lines)


def _print_test_config(cfg, n_noise):
    """Print one configuration entry for backward compatibility"""
    print(_format_test_config_entry(cfg, n_noise))


# =======================================================================
#  Core search: run the full method sweep for a given (n_noise, n_max_steps) pair and return metrics.
#  Used by train and ablation modes; verbose controls detailed logging.
# =======================================================================
def run_full_search(MSE, y, models, t_arr, seed, N, verbose=True,
                    methods_to_run=None):
    """
    Run a 60/40 split and selected method search on the given MSE and labels.
    methods_to_run: set or None; None means all methods.
    Returns: key (method -> (train_acc, val_acc)), info (method -> config dict),
          best_method_name selected on validation.
    """
    enabled = _resolve_methods(methods_to_run) if methods_to_run is not None \
              else set(ALL_METHOD_TAGS)

    n, n_m, n_t = MSE.shape
    tri, vai = stratified_split(y, val_ratio=VAL_RATIO, random_seed=seed)
    Mtr = MSE[tri].astype(np.float32)
    Mva = MSE[vai].astype(np.float32)
    ytr, yva = y[tri], y[vai]

    if verbose:
        print(f"  Train:{len(tri)}  Val:{len(vai)}  "
              f"class distribution (train): {dict(zip(models, np.bincount(ytr, minlength=n_m)))}")

    # -- Single-step accuracy based on the training split --
    pa = np.array([(Mtr[:, :, j].argmin(1) == ytr).mean() for j in range(n_t)])
    to_order = np.argsort(-pa)

    if verbose:
        print(f"\n  single-step accuracy Top-{min(10, n_t)}:")
        for rank in range(min(10, n_t)):
            j = to_order[rank]
            print(f"    #{rank+1}: t={t_arr[j]:>4d}  acc={pa[j]:.4f}")

    def topk(idx, k):
        return idx[np.argsort(-pa[idx])][:min(k, len(idx))]

    Mz_tr = zscore_mse(Mtr);  Mz_va = zscore_mse(Mva)
    Mr_tr = rank_aggregate(Mtr); Mr_va = rank_aggregate(Mva)
    Neff = min(N, n_t)
    tn = to_order[:Neff]
    tn_steps = sorted(t_arr[tn].tolist())

    key = {}
    info = {}
    _bv, _bn = 0.0, ''  # select the best method on validation

    def rec(nm, tr, va):
        nonlocal _bv, _bn
        key[nm] = (float(tr), float(va))
        if float(va) > _bv:
            _bv, _bn = float(va), nm

    # -- Baseline --
    bl_tr = float((Mtr.mean(2).argmin(1) == ytr).mean())
    bl_va = float((Mva.mean(2).argmin(1) == yva).mean())
    rec('Baseline', bl_tr, bl_va)
    info['Baseline'] = {'score_type': 'avg',
                        'timesteps': sorted(t_arr.tolist()),
                        'bias': np.zeros(n_m).tolist()}
    if verbose:
        print(f"\n  Baseline (all {n_t} steps, no bias): Tr={bl_tr:.4f}  Va={bl_va:.4f}")

    # -- A: avg MSE (top-N) + bias --
    if 'A' in enabled:
        if verbose: print(f"  [A] avg top-{Neff} + bias")
        s1 = Mtr[:, :, tn].mean(2); s2 = Mva[:, :, tn].mean(2)
        a, b, c, d, biases = eval_with_bias(s1, s2, ytr, yva, 25)
        rec('A_noBias', a, b); rec('A_bias', c, d)
        info['A_noBias'] = {'score_type': 'avg', 'timesteps': tn_steps,
                            'bias': np.zeros(n_m).tolist()}
        info['A_bias'] = {'score_type': 'avg', 'timesteps': tn_steps,
                          'bias': biases.tolist()}
        if verbose:
            print(f"      noBias Tr/Va: {a:.4f}/{b:.4f}   +bias Tr/Va: {c:.4f}/{d:.4f}")

    # -- B: z-score (top-N) + bias --
    if 'B' in enabled:
        if verbose: print(f"  [B] zscore top-{Neff} + bias")
        s1 = Mz_tr[:, :, tn].mean(2); s2 = Mz_va[:, :, tn].mean(2)
        a, b, c, d, biases = eval_with_bias(s1, s2, ytr, yva, 25)
        rec('B_noBias', a, b); rec('B_bias', c, d)
        info['B_noBias'] = {'score_type': 'zscore', 'timesteps': tn_steps,
                            'bias': np.zeros(n_m).tolist()}
        info['B_bias'] = {'score_type': 'zscore', 'timesteps': tn_steps,
                          'bias': biases.tolist()}
        if verbose:
            print(f"      noBias Tr/Va: {a:.4f}/{b:.4f}   +bias Tr/Va: {c:.4f}/{d:.4f}")

    # -- C: timestepsrange subset (cap N) + bias --
    if 'C' in enabled:
        if verbose: print(f"  [C] range subset (cap {N}) + bias")
        bc = {}
        for mt in [100, 150, 200, 250, 300, 350, 400, 500]:
            mi = np.where(t_arr <= mt)[0]
            if len(mi) == 0:
                continue
            sel = topk(mi, N)
            s1 = Mtr[:, :, sel].mean(2); s2 = Mva[:, :, sel].mean(2)
            a, b, c, d, biases = eval_with_bias(s1, s2, ytr, yva, 21)
            if not bc or d > bc['va']:
                bc = dict(va=d, tr=c, nt=a, nv=b,
                          tag=f't<={mt}({len(sel)} steps)',
                          steps=sorted(t_arr[sel].tolist()),
                          biases=biases.tolist())
        for mn in [30, 50, 70, 90]:
            for mx in [150, 200, 250, 300]:
                mi = np.where((t_arr >= mn) & (t_arr <= mx))[0]
                if len(mi) == 0:
                    continue
                sel = topk(mi, N)
                s1 = Mtr[:, :, sel].mean(2); s2 = Mva[:, :, sel].mean(2)
                a, b, c, d, biases = eval_with_bias(s1, s2, ytr, yva, 21)
                if not bc or d > bc['va']:
                    bc = dict(va=d, tr=c, nt=a, nv=b,
                              tag=f'[{mn},{mx}]({len(sel)} steps)',
                              steps=sorted(t_arr[sel].tolist()),
                              biases=biases.tolist())
        if bc:
            rec('C_best_noBias', bc['nt'], bc['nv'])
            rec('C_best_bias', bc['tr'], bc['va'])
            info['C_best_noBias'] = {'score_type': 'avg', 'timesteps': bc['steps'],
                                      'bias': np.zeros(n_m).tolist()}
            info['C_best_bias'] = {'score_type': 'avg', 'timesteps': bc['steps'],
                                    'bias': bc['biases']}
            if verbose:
                print(f"      Best: {bc['tag']}  nB: {bc['nt']:.4f}/{bc['nv']:.4f}"
                      f"  +B: {bc['tr']:.4f}/{bc['va']:.4f}")

    # -- D: z-score + range subset --
    if 'D' in enabled:
        if verbose: print(f"  [D] zscore + range subset (cap {N}) + bias")
        bd = {}
        for mt in [100, 150, 200, 250, 300, 500]:
            mi = np.where(t_arr <= mt)[0]
            if len(mi) == 0:
                continue
            sel = topk(mi, N)
            s1 = Mz_tr[:, :, sel].mean(2); s2 = Mz_va[:, :, sel].mean(2)
            a, b, c, d, biases = eval_with_bias(s1, s2, ytr, yva, 21)
            if not bd or d > bd['va']:
                bd = dict(va=d, tr=c, nt=a, nv=b,
                          tag=f'z_t<={mt}({len(sel)} steps)',
                          steps=sorted(t_arr[sel].tolist()),
                          biases=biases.tolist())
        for mn in [30, 50, 70, 90]:
            for mx in [150, 200, 250, 300]:
                mi = np.where((t_arr >= mn) & (t_arr <= mx))[0]
                if len(mi) == 0:
                    continue
                sel = topk(mi, N)
                s1 = Mz_tr[:, :, sel].mean(2); s2 = Mz_va[:, :, sel].mean(2)
                a, b, c, d, biases = eval_with_bias(s1, s2, ytr, yva, 21)
                if not bd or d > bd['va']:
                    bd = dict(va=d, tr=c, nt=a, nv=b,
                              tag=f'z_[{mn},{mx}]({len(sel)} steps)',
                              steps=sorted(t_arr[sel].tolist()),
                              biases=biases.tolist())
        if bd:
            rec('D_best_noBias', bd['nt'], bd['nv'])
            rec('D_best_bias', bd['tr'], bd['va'])
            info['D_best_noBias'] = {'score_type': 'zscore', 'timesteps': bd['steps'],
                                      'bias': np.zeros(n_m).tolist()}
            info['D_best_bias'] = {'score_type': 'zscore', 'timesteps': bd['steps'],
                                    'bias': bd['biases']}
            if verbose:
                print(f"      Best: {bd['tag']}  nB: {bd['nt']:.4f}/{bd['nv']:.4f}"
                      f"  +B: {bd['tr']:.4f}/{bd['va']:.4f}")

    # -- E: two-group weighting --
    if 'E' in enabled:
        if verbose: print(f"  [E] two-group weighting (global top-{Neff} split) + bias")
        be = {}
        for sp in [100, 150, 200, 250]:
            lo = tn[t_arr[tn] <= sp]
            hi = tn[t_arr[tn] > sp]
            if len(lo) == 0 or len(hi) == 0:
                continue
            Ml1 = Mtr[:, :, lo].mean(2); Mh1 = Mtr[:, :, hi].mean(2)
            Ml2 = Mva[:, :, lo].mean(2); Mh2 = Mva[:, :, hi].mean(2)
            for wp in range(0, 105, 5):
                wl = wp / 100.0
                s1 = wl * Ml1 + (1 - wl) * Mh1
                s2 = wl * Ml2 + (1 - wl) * Mh2
                a, b, c, d, biases = eval_with_bias(s1, s2, ytr, yva, 17)
                if not be or d > be['va']:
                    be = dict(va=d, tr=c, nt=a, nv=b, tag=f'sp{sp}_w{wp}',
                              lo_steps=sorted(t_arr[lo].tolist()),
                              hi_steps=sorted(t_arr[hi].tolist()),
                              weight_low=wl, biases=biases.tolist())
        if be:
            rec('E_best_noBias', be['nt'], be['nv'])
            rec('E_best_bias', be['tr'], be['va'])
            info['E_best_noBias'] = {'score_type': 'weighted',
                                      'low_steps': be['lo_steps'],
                                      'high_steps': be['hi_steps'],
                                      'weight_low': be['weight_low'],
                                      'bias': np.zeros(n_m).tolist()}
            info['E_best_bias'] = {'score_type': 'weighted',
                                    'low_steps': be['lo_steps'],
                                    'high_steps': be['hi_steps'],
                                    'weight_low': be['weight_low'],
                                    'bias': be['biases']}
            if verbose:
                print(f"      Best: {be['tag']}  nB: {be['nt']:.4f}/{be['nv']:.4f}"
                      f"  +B: {be['tr']:.4f}/{be['va']:.4f}")

    # -- F: Median --
    if 'F' in enabled:
        if verbose: print(f"  [F] median top-{Neff} + bias")
        s1 = np.median(Mtr[:, :, tn], 2); s2 = np.median(Mva[:, :, tn], 2)
        a, b, c, d, biases = eval_with_bias(s1, s2, ytr, yva, 25)
        rec('F_noBias', a, b); rec('F_bias', c, d)
        info['F_noBias'] = {'score_type': 'median', 'timesteps': tn_steps,
                            'bias': np.zeros(n_m).tolist()}
        info['F_bias'] = {'score_type': 'median', 'timesteps': tn_steps,
                          'bias': biases.tolist()}
        if verbose:
            print(f"      noBias Tr/Va: {a:.4f}/{b:.4f}   +bias Tr/Va: {c:.4f}/{d:.4f}")

    # -- G: Rank --
    if 'G' in enabled:
        if verbose: print(f"  [G] rank top-{Neff} + bias (including subsets)")
        s1 = Mr_tr[:, :, tn].mean(2); s2 = Mr_va[:, :, tn].mean(2)
        a, b, c, d, biases_g = eval_with_bias(s1, s2, ytr, yva, 25)
        g_nob, g_b, g_tag = (a, b), (c, d), f'top-{Neff}'
        g_steps = list(tn_steps)
        g_biases = biases_g.tolist()
        for mt in [100, 150, 200, 250, 300]:
            mi = np.where(t_arr <= mt)[0]
            if len(mi) == 0:
                continue
            sel = topk(mi, N)
            s1 = Mr_tr[:, :, sel].mean(2); s2 = Mr_va[:, :, sel].mean(2)
            a2, b2, c2, d2, biases_g2 = eval_with_bias(s1, s2, ytr, yva, 21)
            if d2 > g_b[1]:
                g_nob, g_b, g_tag = (a2, b2), (c2, d2), f'rank_t<={mt}({len(sel)} steps)'
                g_steps = sorted(t_arr[sel].tolist())
                g_biases = biases_g2.tolist()
        rec('G_noBias', g_nob[0], g_nob[1])
        rec('G_bias', g_b[0], g_b[1])
        info['G_noBias'] = {'score_type': 'rank', 'timesteps': g_steps,
                            'bias': np.zeros(n_m).tolist()}
        info['G_bias'] = {'score_type': 'rank', 'timesteps': g_steps,
                          'bias': g_biases}
        if verbose:
            print(f"      Best: {g_tag}  nB: {g_nob[0]:.4f}/{g_nob[1]:.4f}"
                  f"  +B: {g_b[0]:.4f}/{g_b[1]:.4f}")

    # -- H: Top-K --
    if 'H' in enabled:
        if verbose: print(f"  [H] top-K (K=3..{Neff}) + bias")
        bh = {}
        for K in range(3, Neff + 1):
            sel = to_order[:K]
            s1 = Mtr[:, :, sel].mean(2); s2 = Mva[:, :, sel].mean(2)
            a, b, c, d, biases_h = eval_with_bias(s1, s2, ytr, yva, 21)
            if not bh or d > bh['va']:
                bh = dict(va=d, tr=c, nt=a, nv=b, K=K,
                          steps=sorted(t_arr[to_order[:K]].tolist()),
                          biases=biases_h.tolist())
        if bh:
            rec('H_best_noBias', bh['nt'], bh['nv'])
            rec('H_best_bias', bh['tr'], bh['va'])
            info['H_best_noBias'] = {'score_type': 'avg', 'timesteps': bh['steps'],
                                      'bias': np.zeros(n_m).tolist()}
            info['H_best_bias'] = {'score_type': 'avg', 'timesteps': bh['steps'],
                                    'bias': bh['biases']}
            if verbose:
                print(f"      Best K={bh['K']}  nB: {bh['nt']:.4f}/{bh['nv']:.4f}"
                      f"  +B: {bh['tr']:.4f}/{bh['va']:.4f}")

    # -- EH: two-group Top-K --
    if 'EH' in enabled:
        if verbose: print(f"  [EH] two-group Top-K (nL+nH<={N}) + bias ...")
        beh = {}
        for sp in [100, 150, 200, 250]:
            li = np.where(t_arr <= sp)[0]
            hi = np.where(t_arr > sp)[0]
            if len(li) == 0 or len(hi) == 0:
                continue
            ls = li[np.argsort(-pa[li])]
            hs = hi[np.argsort(-pa[hi])]
            lo_tr = {k: Mtr[:, :, ls[:k]].mean(2) for k in range(1, min(N, len(ls)) + 1)}
            lo_va = {k: Mva[:, :, ls[:k]].mean(2) for k in range(1, min(N, len(ls)) + 1)}
            hi_tr = {k: Mtr[:, :, hs[:k]].mean(2) for k in range(1, min(N, len(hs)) + 1)}
            hi_va = {k: Mva[:, :, hs[:k]].mean(2) for k in range(1, min(N, len(hs)) + 1)}
            mxL = min(N - 1, len(ls))
            for nL in range(1, mxL + 1):
                mxH = min(N - nL, len(hs))
                for nH in range(1, mxH + 1):
                    Ml1 = lo_tr[nL]; Mh1 = hi_tr[nH]
                    Ml2 = lo_va[nL]; Mh2 = hi_va[nH]
                    for wp in range(0, 105, 10):
                        wl = wp / 100.0
                        s1 = wl * Ml1 + (1 - wl) * Mh1
                        s2 = wl * Ml2 + (1 - wl) * Mh2
                        a, b, c, d, biases_eh = eval_with_bias(s1, s2, ytr, yva, 17)
                        if not beh or d > beh['va']:
                            beh = dict(va=d, tr=c, nt=a, nv=b,
                                       tag=f'sp{sp}_nL{nL}_nH{nH}_w{wp}',
                                       total=nL + nH,
                                       lo_steps=sorted(t_arr[ls[:nL]].tolist()),
                                       hi_steps=sorted(t_arr[hs[:nH]].tolist()),
                                       weight_low=wl,
                                       biases=biases_eh.tolist())
        if beh:
            rec('EH_best_noBias', beh['nt'], beh['nv'])
            rec('EH_best_bias', beh['tr'], beh['va'])
            info['EH_best_noBias'] = {'score_type': 'weighted',
                                       'low_steps': beh['lo_steps'],
                                       'high_steps': beh['hi_steps'],
                                       'weight_low': beh['weight_low'],
                                       'bias': np.zeros(n_m).tolist()}
            info['EH_best_bias'] = {'score_type': 'weighted',
                                     'low_steps': beh['lo_steps'],
                                     'high_steps': beh['hi_steps'],
                                     'weight_low': beh['weight_low'],
                                     'bias': beh['biases']}
            if verbose:
                print(f"      Best: {beh['tag']} ({beh['total']} steps)  "
                      f"nB: {beh['nt']:.4f}/{beh['nv']:.4f}  "
                      f"+B: {beh['tr']:.4f}/{beh['va']:.4f}")

    key['Overall_best'] = key[_bn]
    return key, info, _bn


# =======================================================================
#  Train mode: detailed search for one parameter combination
# =======================================================================
def process_one_file(csv_path, seed, N, n_noise):
    fname = os.path.basename(csv_path).replace('.csv', '')
    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"  File: {fname}")
    print(f"  N_MAX={N}  N_NOISE={n_noise}  seed={seed}  val_ratio={VAL_RATIO}")
    print(f"{'='*70}")

    MSE, y, models, t_cols, t_arr = load_data(csv_path, n_noise, N_IMAGES)
    n, n_m, n_t = MSE.shape
    print(f"  model({n_m}): {models}")
    print(f"  images:{n}  timesteps:{n_t}  range:[{t_arr.min()},{t_arr.max()}]")

    key, info, best_name = run_full_search(MSE, y, models, t_arr, seed, N,
                                            verbose=True,
                                            methods_to_run=METHODS_TO_RUN)
    best_cfg = info.get(best_name, {})

    print(f"\n  >>> BEST (on validation): {best_name}  "
          f"Tr={key[best_name][0]:.4f}  Va={key[best_name][1]:.4f}"
          f"  [fileelapsed {time.time()-t0:.1f}s]")

    print(f"\n  {'-'*60}")
    print(f"  Best configuration details:")
    print(f"    Method:       {best_name}")
    print(f"    noise count:   {n_noise}")
    print(f"    score type:   {best_cfg.get('score_type', 'N/A')}")
    if 'timesteps' in best_cfg:
        print(f"    number of timesteps: {len(best_cfg['timesteps'])}")
        print(f"    timesteps:     {best_cfg['timesteps']}")
    if 'low_steps' in best_cfg:
        print(f"    low-step group ({len(best_cfg['low_steps'])} steps): {best_cfg['low_steps']}")
        print(f"    high-step group ({len(best_cfg['high_steps'])} steps): {best_cfg['high_steps']}")
        print(f"    low-step weight:   {best_cfg['weight_low']}")
    bias_str = [f"{v:.6f}" for v in best_cfg.get('bias', [])]
    print(f"    Bias:       [{', '.join(bias_str)}]")
    print(f"    model order:   {list(models)}")

    print(f"\n  -- Reusable TEST_CONFIGS entry; update csv_path as needed --")
    _print_test_config(best_cfg, n_noise)
    print(f"  {'-'*60}")

    return key, fname, best_name, info, list(models)


# =======================================================================
#  Test mode: fixed configuration without splitting
# =======================================================================
def process_test_file(config):
    csv_path = config['csv_path']
    n_noise = config['n_noise']
    score_type = config['score_type']
    bias = np.array(config['bias'], dtype=np.float32)

    fname = os.path.basename(csv_path).replace('.csv', '')
    missing = []
    if n_noise is None:
        missing.append('n_noise')
    if len(bias) == 0:
        missing.append('bias')
    if score_type == 'weighted':
        for field in ('low_steps', 'high_steps', 'weight_low'):
            if field not in config or config[field] in (None, []):
                missing.append(field)
    elif not config.get('timesteps'):
        missing.append('timesteps')

    if missing:
        print(f"\n  [ERROR] TEST_CONFIGS entry for {fname} is incomplete: {', '.join(missing)}")
        return None, None, fname

    print(f"\n{'='*70}")
    print(f"  [TEST] File: {fname}")
    print(f"  noise count: {n_noise}  score type: {score_type}")
    print(f"{'='*70}")

    MSE, y, models, t_cols, t_arr = load_data(csv_path, n_noise)
    n, n_m, n_t = MSE.shape
    print(f"  model({n_m}): {models}")
    print(f"  images:{n}  timesteps:{n_t}  range:[{t_arr.min()},{t_arr.max()}]")
    print(f"  class distribution: {dict(zip(models, np.bincount(y, minlength=n_m)))}")

    if len(bias) != n_m:
        print(f"  [ERROR] Bias length ({len(bias)}) does not match the number of models ({n_m})!")
        return None, None, fname

    M = MSE.astype(np.float32)

    if score_type == 'weighted':
        lo_vals = config['low_steps']
        hi_vals = config['high_steps']
        wl = config['weight_low']
        lo_idx, hi_idx = [], []
        for t in lo_vals:
            idx = np.where(t_arr == t)[0]
            if len(idx) == 0:
                print(f"  [ERROR] timesteps {t} does not exist! available: {sorted(t_arr.tolist())}")
                return None, None, fname
            lo_idx.append(idx[0])
        for t in hi_vals:
            idx = np.where(t_arr == t)[0]
            if len(idx) == 0:
                print(f"  [ERROR] timesteps {t} does not exist! available: {sorted(t_arr.tolist())}")
                return None, None, fname
            hi_idx.append(idx[0])
        lo_idx = np.array(lo_idx)
        hi_idx = np.array(hi_idx)
        scored = wl * M[:, :, lo_idx].mean(2) + (1 - wl) * M[:, :, hi_idx].mean(2)
        total_steps = len(lo_vals) + len(hi_vals)
        print(f"  low-step group ({len(lo_vals)} steps): {lo_vals}")
        print(f"  high-step group ({len(hi_vals)} steps): {hi_vals}")
        print(f"  low-step weight: {wl}  total steps: {total_steps}")
    else:
        ts_vals = config['timesteps']
        ts_idx = []
        for t in ts_vals:
            idx = np.where(t_arr == t)[0]
            if len(idx) == 0:
                print(f"  [ERROR] timesteps {t} does not exist! available: {sorted(t_arr.tolist())}")
                return None, None, fname
            ts_idx.append(idx[0])
        ts_idx = np.array(ts_idx)
        print(f"  timesteps ({len(ts_vals)} steps): {ts_vals}")

        if score_type == 'zscore':
            M = zscore_mse(M)
        elif score_type == 'rank':
            M = rank_aggregate(M)

        if score_type == 'median':
            scored = np.median(M[:, :, ts_idx], axis=2)
        else:
            scored = M[:, :, ts_idx].mean(axis=2)

    print(f"  Bias: {bias.tolist()}")

    pred_nb = scored.argmin(axis=1)
    acc_nb = float((pred_nb == y).mean())
    pred_b = (scored + bias).argmin(axis=1)
    acc_b = float((pred_b == y).mean())

    print(f"\n  -- Outputs --")
    print(f"  Accuracy (no bias): {acc_nb:.4f}  ({int((pred_nb == y).sum())}/{n})")
    print(f"  Accuracy (with bias): {acc_b:.4f}  ({int((pred_b == y).sum())}/{n})")

    print(f"\n  -- Per-class accuracy (with bias) --")
    for ci in range(n_m):
        mask = (y == ci)
        cnt = int(mask.sum())
        if cnt > 0:
            correct = int((pred_b[mask] == ci).sum())
            cls_acc = correct / cnt
            print(f"    {models[ci]}: {cls_acc:.4f}  ({correct}/{cnt})")

    print(f"\n  -- Confusion matrix (with bias, rows=true, columns=predicted) --")
    cm_header = f"  {'':>15}" + "".join(f"{m:>12}" for m in models)
    print(cm_header)
    for i in range(n_m):
        row = f"  {models[i]:>15}"
        for j in range(n_m):
            cnt = int(((y == i) & (pred_b == j)).sum())
            row += f"{cnt:>12}"
        print(row)

    return acc_nb, acc_b, fname


# =======================================================================
#  Ablation mode: scan the (N_NOISE, N_MAX_STEPS) grid
# =======================================================================
def process_ablation_file(csv_path, seed, noise_grid, steps_grid):
    """
    Scan the (n_noise, n_max_steps) grid for one file.
    Returns:
      train_grid[i,j] = train acc (best method for noise_grid[i] and steps_grid[j])
      val_grid[i,j]   = val acc
      best_method[i,j] = best method name
      best_combo_info = dict, complete information for the highest-validation combination in this file
                        {'n_noise', 'n_steps', 'method', 'config', 'tr_acc', 'va_acc'}
    """
    fname = os.path.basename(csv_path).replace('.csv', '')
    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"  [ABLATION] File: {fname}")
    print(f"  N_NOISE grid: {noise_grid}")
    print(f"  N_MAX_STEPS grid: {steps_grid}")
    print(f"  total {len(noise_grid)} x {len(steps_grid)} = "
          f"{len(noise_grid) * len(steps_grid)} combinations")
    print(f"{'='*70}")

    nN = len(noise_grid)
    nS = len(steps_grid)
    train_grid = np.full((nN, nS), np.nan, dtype=np.float64)
    val_grid = np.full((nN, nS), np.nan, dtype=np.float64)
    best_method = [['' for _ in range(nS)] for _ in range(nN)]

    best_combo_info = None  # Track the global best combination for this file

    cache_noise = None
    cache_data = None

    for i, nn in enumerate(noise_grid):
        if cache_noise != nn:
            print(f"\n  [load] n_noise={nn} ...", end=' ', flush=True)
            tld = time.time()
            MSE, y, models, t_cols, t_arr = load_data(csv_path, nn, N_IMAGES)
            cache_noise = nn
            cache_data = (MSE, y, models, t_cols, t_arr)
            print(f"shape={MSE.shape}  [{time.time()-tld:.1f}s]")
        else:
            MSE, y, models, t_cols, t_arr = cache_data

        for j, ns in enumerate(steps_grid):
            tcomb = time.time()
            key, info, bn = run_full_search(MSE, y, models, t_arr,
                                             seed, ns, verbose=False,
                                             methods_to_run=METHODS_TO_RUN)
            tr_acc, va_acc = key[bn]
            train_grid[i, j] = tr_acc
            val_grid[i, j] = va_acc
            best_method[i][j] = bn
            print(f"    n_noise={nn:>3d}  n_steps={ns:>3d}  "
                  f"best={bn:<18}  Tr={tr_acc:.4f}  Va={va_acc:.4f}  "
                  f"[{time.time()-tcomb:.1f}s]")

            # Track the highest-validation combination for this file
            if best_combo_info is None or va_acc > best_combo_info['va_acc']:
                best_combo_info = {
                    'n_noise': nn,
                    'n_steps': ns,
                    'method': bn,
                    'config': info.get(bn, {}),
                    'tr_acc': float(tr_acc),
                    'va_acc': float(va_acc),
                }

    print(f"\n  [fileelapsed {time.time()-t0:.1f}s]")

    _print_ablation_table(fname, noise_grid, steps_grid,
                          train_grid, val_grid, best_method)

    return train_grid, val_grid, best_method, fname, list(models), best_combo_info


def _print_ablation_table(fname, noise_grid, steps_grid,
                          train_grid, val_grid, best_method):
    """Print the two-dimensional ablation table for one file, centered on validation accuracy with train accuracy included."""
    cw = 9
    axis_label = "N_NOISE\\N_STEPS"
    print(f"\n  +- {fname} - Val Accuracy (columns: N_MAX_STEPS, rows: N_NOISE) -")
    hdr = f"  | {axis_label:>16}" + "".join(f"{s:>{cw}d}" for s in steps_grid)
    print(hdr)
    print("  | " + "-" * (16 + cw * len(steps_grid)))
    for i, nn in enumerate(noise_grid):
        row = f"  | {nn:>16d}"
        for j in range(len(steps_grid)):
            v = val_grid[i, j]
            row += f"{v:>{cw}.4f}" if not np.isnan(v) else f"{'N/A':>{cw}}"
        print(row)

    print(f"\n  +- {fname} - Train Accuracy -")
    print(hdr)
    print("  | " + "-" * (16 + cw * len(steps_grid)))
    for i, nn in enumerate(noise_grid):
        row = f"  | {nn:>16d}"
        for j in range(len(steps_grid)):
            v = train_grid[i, j]
            row += f"{v:>{cw}.4f}" if not np.isnan(v) else f"{'N/A':>{cw}}"
        print(row)

    # Find the best combination based on validation accuracy
    bi, bj = np.unravel_index(np.nanargmax(val_grid), val_grid.shape)
    print(f"\n  +- best combination: n_noise={noise_grid[bi]}  n_steps={steps_grid[bj]}  "
          f"Tr={train_grid[bi,bj]:.4f}  Va={val_grid[bi,bj]:.4f}  "
          f"method={best_method[bi][bj]}")


def _print_ablation_summary(noise_grid, steps_grid, all_train, all_val,
                            all_best_method, fnames):
    """Cross-file summary: average validation accuracy for each combination."""
    nN = len(noise_grid)
    nS = len(steps_grid)
    avg_val = np.full((nN, nS), np.nan)
    avg_train = np.full((nN, nS), np.nan)
    for i in range(nN):
        for j in range(nS):
            tv = [all_val[fn][i, j] for fn in fnames]
            tt = [all_train[fn][i, j] for fn in fnames]
            avg_val[i, j] = np.nanmean(tv)
            avg_train[i, j] = np.nanmean(tt)

    cw = 9
    print(f"\n\n{'#'*80}")
    print(f"  ===== Ablation summary: six files (N_NOISE x N_MAX_STEPS) =====")
    print(f"{'#'*80}")

    # One validation table per file
    for fn in fnames:
        print(f"\n  -- {fn} - Val Acc --")
        axis_label = "N_NOISE\\N_STEPS"
        hdr = f"  {axis_label:>16}" + "".join(f"{s:>{cw}d}" for s in steps_grid)
        print(hdr)
        print("  " + "-" * (16 + cw * len(steps_grid)))
        for i, nn in enumerate(noise_grid):
            row = f"  {nn:>16d}"
            for j in range(len(steps_grid)):
                v = all_val[fn][i, j]
                row += f"{v:>{cw}.4f}" if not np.isnan(v) else f"{'N/A':>{cw}}"
            print(row)
        bi, bj = np.unravel_index(np.nanargmax(all_val[fn]), all_val[fn].shape)
        print(f"    Best: n_noise={noise_grid[bi]} n_steps={steps_grid[bj]} "
              f"Tr={all_train[fn][bi,bj]:.4f} Va={all_val[fn][bi,bj]:.4f}")

    # One train table per file
    for fn in fnames:
        print(f"\n  -- {fn} - Train Acc --")
        axis_label = "N_NOISE\\N_STEPS"
        hdr = f"  {axis_label:>16}" + "".join(f"{s:>{cw}d}" for s in steps_grid)
        print(hdr)
        print("  " + "-" * (16 + cw * len(steps_grid)))
        for i, nn in enumerate(noise_grid):
            row = f"  {nn:>16d}"
            for j in range(len(steps_grid)):
                v = all_train[fn][i, j]
                row += f"{v:>{cw}.4f}" if not np.isnan(v) else f"{'N/A':>{cw}}"
            print(row)

    # Cross-file average validation accuracy
    print(f"\n  ===== Cross-file average validation accuracy =====")
    axis_label = "N_NOISE\\N_STEPS"
    hdr = f"  {axis_label:>16}" + "".join(f"{s:>{cw}d}" for s in steps_grid)
    print(hdr)
    print("  " + "-" * (16 + cw * len(steps_grid)))
    for i, nn in enumerate(noise_grid):
        row = f"  {nn:>16d}"
        for j in range(len(steps_grid)):
            v = avg_val[i, j]
            row += f"{v:>{cw}.4f}" if not np.isnan(v) else f"{'N/A':>{cw}}"
        print(row)
    bi, bj = np.unravel_index(np.nanargmax(avg_val), avg_val.shape)
    print(f"\n  >>> Best average validation result: n_noise={noise_grid[bi]}  n_steps={steps_grid[bj]}  "
          f"avg_Tr={avg_train[bi,bj]:.4f}  avg_Va={avg_val[bi,bj]:.4f}")

    # -- Combined comparison table: one row per combination and one column per file. --
    print(f"\n\n  ===== Comparison (Train/Val Acc, one row per combination) =====")
    fcw = 14   # each cell contains "0.9300/0.9400"
    h = f"  {'(noise,steps)':>16}" + "".join(f"{fn[:13]:>{fcw}}" for fn in fnames) + f"{'avg':>{fcw}}"
    print(h)
    print("  " + "-" * (16 + fcw * (len(fnames) + 1)))
    for i, nn in enumerate(noise_grid):
        for j, ns in enumerate(steps_grid):
            tag = f"({nn},{ns})"
            row = f"  {tag:>16}"
            for fn in fnames:
                tv = all_train[fn][i, j]
                vv = all_val[fn][i, j]
                if not (np.isnan(tv) or np.isnan(vv)):
                    cell = f"{tv:.4f}/{vv:.4f}"
                else:
                    cell = "N/A"
                row += f"{cell:>{fcw}}"
            avg_cell = f"{avg_train[i,j]:.4f}/{avg_val[i,j]:.4f}" \
                       if not (np.isnan(avg_train[i,j]) or np.isnan(avg_val[i,j])) \
                       else "N/A"
            row += f"{avg_cell:>{fcw}}"
            print(row)


# =======================================================================
#  Train-mode comparison
# =======================================================================
def print_comparison(all_keys, fnames, best_names):
    nf = len(fnames)
    labels = [f'F{i+1}' for i in range(nf)]
    mw = 20
    cw = 8

    print(f"\n\n{'#'*80}")
    print("  ===== multi-file comparison =====")
    print(f"  N_NOISE={N_NOISE}  N_MAX_STEPS={N_MAX_STEPS}  SEED={RANDOM_SEED}  "
          f"VAL_RATIO={VAL_RATIO}")
    print(f"{'#'*80}")
    print(f"\n  file labels:")
    for i, fn in enumerate(fnames):
        bm = best_names.get(fn, '?')
        print(f"    F{i+1} = {fn}")
        print(f"         best method (on val): {bm}")

    hdr = f"  {'Method':<{mw}}" + "".join(f"{lb:>{cw}}" for lb in labels)
    sep = "  " + "-" * (mw + cw * nf)

    print(f"\n  ===== Train Accuracy =====")
    print(hdr); print(sep)
    for m in KEY_METHODS:
        row = f"  {m:<{mw}}"
        for fn in fnames:
            v = all_keys.get(fn, {}).get(m)
            row += f"{v[0]:>{cw}.4f}" if v else f"{'N/A':>{cw}}"
        print(row)

    print(f"\n  ===== Val Accuracy =====")
    print(hdr); print(sep)
    for m in KEY_METHODS:
        row = f"  {m:<{mw}}"
        best_in_row = False
        for fn in fnames:
            v = all_keys.get(fn, {}).get(m)
            row += f"{v[1]:>{cw}.4f}" if v else f"{'N/A':>{cw}}"
            if best_names.get(fn) == m:
                best_in_row = True
        if best_in_row:
            row += "  <<<best"
        print(row)

    pcw = 15
    print(f"\n  ===== combined (Tr/Va) =====")
    h2 = f"  {'Method':<{mw}}" + "".join(f"{lb:^{pcw}}" for lb in labels)
    print(h2)
    print("  " + "-" * (mw + pcw * nf))
    for m in KEY_METHODS:
        row = f"  {m:<{mw}}"
        for fn in fnames:
            v = all_keys.get(fn, {}).get(m)
            if v:
                cell = f"{v[0]:.4f}/{v[1]:.4f}"
                row += f"{cell:^{pcw}}"
            else:
                row += f"{'N/A':^{pcw}}"
        print(row)

    print(f"\n  {'='*60}")
    print("  Best method per file based on validation:")
    print(f"  {'file':<8} {'best method':<35} {'Tr':>7} {'Va':>7} {'delta_Va':>8}")
    print(f"  {'-'*58}")
    for i, fn in enumerate(fnames):
        bm = best_names[fn]
        v = all_keys[fn].get('Overall_best', (0, 0))
        bl = all_keys[fn].get('Baseline', (0, 0))
        delta = v[1] - bl[1]
        print(f"  F{i+1:<6} {bm:<35} {v[0]:>7.4f} {v[1]:>7.4f} {delta:>+8.4f}")


# =======================================================================
#  main
# =======================================================================
def main():
    t_start = time.time()

    if MODE == 'train':
        print("=" * 70)
        print("  Model attribution - multi-file comparison [train mode]")
        print(f"  N_MAX_STEPS={N_MAX_STEPS}  N_NOISE={N_NOISE}  "
              f"RANDOM_SEED={RANDOM_SEED}  VAL_RATIO={VAL_RATIO}")
        print(f"  N_IMAGES={N_IMAGES}  METHODS_TO_RUN={METHODS_TO_RUN}")
        print(f"  GPU: {'Yes' if USE_GPU else 'No'}")
        print(f"  number of files: {len(CSV_PATHS)}")
        print("=" * 70)

        for path in CSV_PATHS:
            if not os.path.exists(path):
                print(f"\n  [ERROR] filedoes not exist: {path}")
                return

        all_keys = {}; fnames = []; best_names = {}
        all_info = {}; fname_to_path = {}
        for path in CSV_PATHS:
            key, fn, bn, info_dict, models = process_one_file(
                path, RANDOM_SEED, N_MAX_STEPS, N_NOISE)
            all_keys[fn] = key
            fnames.append(fn)
            best_names[fn] = bn
            all_info[fn] = info_dict
            fname_to_path[fn] = path

        print_comparison(all_keys, fnames, best_names)

        # -- Summary output: reusable TEST_CONFIGS block --
        # Print to terminal and write to a timestamped file to avoid overwrites
        cfg_lines = []
        cfg_lines.append("# " + "=" * 78)
        cfg_lines.append("# Auto-generated TEST_CONFIGS")
        cfg_lines.append(f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        cfg_lines.append(f"# Source script: train mode")
        cfg_lines.append(f"# N_NOISE={N_NOISE}  N_MAX_STEPS={N_MAX_STEPS}  "
                         f"VAL_RATIO={VAL_RATIO}  RANDOM_SEED={RANDOM_SEED}")
        cfg_lines.append(f"# METHODS_TO_RUN={METHODS_TO_RUN}  N_IMAGES={N_IMAGES}")
        cfg_lines.append("# Note: csv_path defaults to the training CSV; replace it with the corresponding independent test CSV path.")
        cfg_lines.append("# " + "=" * 78)
        cfg_lines.append("")
        cfg_lines.append("TEST_CONFIGS = [")
        for fn in fnames:
            bn = best_names[fn]
            best_cfg = all_info[fn].get(bn, {})
            tr_acc, va_acc = all_keys[fn].get(bn, (0.0, 0.0))
            comment = (f"{fn}  | best={bn}  "
                       f"Tr={tr_acc:.4f}  Va={va_acc:.4f}  "
                       f"(N_NOISE={N_NOISE}, N_MAX_STEPS={N_MAX_STEPS})")
            entry = _format_test_config_entry(
                best_cfg, N_NOISE,
                csv_path=fname_to_path[fn],
                comment=comment)
            cfg_lines.append(entry)
        cfg_lines.append("]")

        # Print to terminal
        print(f"\n\n{'#'*80}")
        print("  ===== Reusable TEST_CONFIGS block (one entry per file) =====")
        print(f"  Note: csv_path defaults to the training CSV; replace it with the corresponding independent test CSV path.")
        print(f"{'#'*80}\n")
        for ln in cfg_lines:
            print(ln)

        # Write file
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'test_configs_out')
        os.makedirs(out_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(
            out_dir,
            f'test_configs_N{N_NOISE}_S{N_MAX_STEPS}_{timestamp}.py')
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(cfg_lines) + '\n')
            print(f"\n  >>> Saved TEST_CONFIGS to file: {out_path}")
        except Exception as e:
            print(f"\n  [WARN] Failed to write file: {e}")
        print(f"  >>> Update each csv_path to the independent test-set path, "
              f"then set MODE='test' and rerun this script.")

    elif MODE == 'test':
        print("=" * 70)
        print("  Model attribution [test mode]")
        print(f"  GPU: {'Yes' if USE_GPU else 'No'}")
        print(f"  number of test configurations: {len(TEST_CONFIGS)}")
        print("=" * 70)

        if not TEST_CONFIGS:
            print("\n  [ERROR] TEST_CONFIGS is empty!")
            return

        results = []
        for cfg in TEST_CONFIGS:
            if not os.path.exists(cfg['csv_path']):
                print(f"\n  [ERROR] filedoes not exist: {cfg['csv_path']}")
                continue
            acc_nb, acc_b, fname = process_test_file(cfg)
            if acc_nb is not None:
                results.append((fname, acc_nb, acc_b))

        if results:
            print(f"\n{'='*70}")
            print("  === test summary ===")
            print(f"  {'file':<40} {'no bias':>10} {'with bias':>10}")
            print(f"  {'-'*60}")
            for fname, acc_nb, acc_b in results:
                print(f"  {fname:<40} {acc_nb:>10.4f} {acc_b:>10.4f}")

    elif MODE == 'ablation':
        print("=" * 70)
        print("  Model attribution [ablation mode]")
        print(f"  N_NOISE grid:    {ABL_NOISE_GRID}")
        print(f"  N_MAX_STEPS grid: {ABL_STEPS_GRID}")
        print(f"  SEED={ABL_SEED}  VAL_RATIO={VAL_RATIO}")
        print(f"  N_IMAGES={N_IMAGES}  METHODS_TO_RUN={METHODS_TO_RUN}")
        print(f"  GPU: {'Yes' if USE_GPU else 'No'}")
        print(f"  number of files: {len(CSV_PATHS)}")
        print(f"  total combinations: {len(CSV_PATHS)} x {len(ABL_NOISE_GRID)} x "
              f"{len(ABL_STEPS_GRID)} = "
              f"{len(CSV_PATHS) * len(ABL_NOISE_GRID) * len(ABL_STEPS_GRID)}")
        print("=" * 70)

        for path in CSV_PATHS:
            if not os.path.exists(path):
                print(f"\n  [ERROR] filedoes not exist: {path}")
                return

        all_train = {}; all_val = {}; all_best = {}
        all_combo_info = {}; fname_to_path = {}
        fnames = []
        for path in CSV_PATHS:
            tr_g, va_g, bm, fn, models, combo_info = process_ablation_file(
                path, ABL_SEED, ABL_NOISE_GRID, ABL_STEPS_GRID)
            all_train[fn] = tr_g
            all_val[fn] = va_g
            all_best[fn] = bm
            all_combo_info[fn] = combo_info
            fname_to_path[fn] = path
            fnames.append(fn)

        _print_ablation_summary(ABL_NOISE_GRID, ABL_STEPS_GRID,
                                 all_train, all_val, all_best, fnames)

        # -- Summary output: TEST_CONFIGS for each file best combination, printed and saved --
        cfg_lines = []
        cfg_lines.append("# " + "=" * 78)
        cfg_lines.append("# Auto-generated TEST_CONFIGS (from ablation mode)")
        cfg_lines.append(f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        cfg_lines.append(f"# Each file uses the highest-validation ablation-grid combination as the best configuration")
        cfg_lines.append(f"# ABL_NOISE_GRID={ABL_NOISE_GRID}")
        cfg_lines.append(f"# ABL_STEPS_GRID={ABL_STEPS_GRID}")
        cfg_lines.append(f"# METHODS_TO_RUN={METHODS_TO_RUN}  "
                         f"VAL_RATIO={VAL_RATIO}  SEED={ABL_SEED}  N_IMAGES={N_IMAGES}")
        cfg_lines.append("# Note: csv_path defaults to the training CSV; replace it with the corresponding independent test CSV path.")
        cfg_lines.append("# " + "=" * 78)
        cfg_lines.append("")
        cfg_lines.append("TEST_CONFIGS = [")
        for fn in fnames:
            ci = all_combo_info[fn]
            if ci is None:
                cfg_lines.append(f"    # {fn}  | no valid combination; skipped")
                continue
            comment = (f"{fn}  | best={ci['method']}  "
                       f"Tr={ci['tr_acc']:.4f}  Va={ci['va_acc']:.4f}  "
                       f"(N_NOISE={ci['n_noise']}, N_MAX_STEPS={ci['n_steps']})")
            entry = _format_test_config_entry(
                ci['config'], ci['n_noise'],
                csv_path=fname_to_path[fn],
                comment=comment)
            cfg_lines.append(entry)
        cfg_lines.append("]")

        # Print to terminal
        print(f"\n\n{'#'*80}")
        print("  ===== Reusable TEST_CONFIGS block (ablation mode: best combination per file) =====")
        print(f"  Note: csv_path defaults to the training CSV; replace it with the corresponding independent test CSV path.")
        print(f"{'#'*80}\n")
        for ln in cfg_lines:
            print(ln)

        # Save to file
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'test_configs_out')
        os.makedirs(out_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(
            out_dir,
            f'test_configs_ablation_{timestamp}.py')
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(cfg_lines) + '\n')
            print(f"\n  >>> Saved TEST_CONFIGS to file: {out_path}")
        except Exception as e:
            print(f"\n  [WARN] Failed to write file: {e}")
        print(f"  >>> Update each csv_path to the independent test-set path, "
              f"then set MODE='test' and rerun this script.")

    else:
        print(f"  [ERROR] unknown mode: {MODE} (set it to 'train' / 'test' / 'ablation')")

    print(f"\n  Total elapsed: {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
