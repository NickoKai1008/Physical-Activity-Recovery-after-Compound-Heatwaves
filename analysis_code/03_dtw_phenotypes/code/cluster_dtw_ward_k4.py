"""
OPTIMIZED LAG-RESPONSE CLUSTERING — publication-grade variant.

Design decisions (v2):

  1. PRIMARY distance = DTW on z-scored sig-filtered beta (same feature as
     Scheme B in `dtw_cluster_multi_schemeB.py`).
     -> keeps the full 7-dim lag structure; DTW is robust to small time-shifts.

  2. The submission phenotype definition is locked to k=4 and Ward linkage,
     reproducing the archived 18/14/11/20 city partition.

  3. The k-search and alternative-linkage diagnostics use a composite ranking
     of three validity indices:
     silhouette (higher=better), Calinski-Harabasz (higher=better),
     Davies-Bouldin (lower=better). These diagnostics quantify assignment
     sensitivity around the locked submission classification.

  4. An auxiliary shape-feature panel summarizes 8 interpretable features per
     city after clustering and supports interpretation of each phenotype.

Output: analysis_code/03_dtw_phenotypes/output/model
"""
from __future__ import annotations

import os
from pathlib import Path
# Belt-and-braces against Windows GBK consoles (conda run re-spawns cmd and
# silently drops stdout.reconfigure). Setting the env var and rebinding
# stdout/stderr both make Python write UTF-8 regardless of the host shell.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import warnings
import numpy as np
# NOTE: console prints additionally use plain ASCII (b for beta, -> for arrow,
# +/- for plus-minus) so even if a pipe strips the UTF-8 setting above we do
# not crash. Unicode is only kept inside matplotlib figure titles where the
# font renderer handles it correctly.
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import PowerNorm
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    calinski_harabasz_score, davies_bouldin_score,
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from dtaidistance import dtw

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family": "Arial", "font.size": 11,
    "axes.linewidth": 0.8, "figure.dpi": 300,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
})

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_DIR = SCRIPT_DIR.parent
OUT = Path(os.environ.get("DTW_OUT", MODULE_DIR / "output" / "model"))
OUT.mkdir(parents=True, exist_ok=True)

# Lock the submission phenotype definition. Environment overrides are retained
# only for the explicitly labelled linkage and k-sensitivity analyses.
FORCE_K = int(os.environ.get("DTW_FORCE_K", "4"))
_force_linkage_env = os.environ.get("DTW_FORCE_LINKAGE", "ward")
FORCE_LINKAGE = None if _force_linkage_env.lower() in ("", "none", "auto") else _force_linkage_env

PALETTE = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488",
    "#F39B7F", "#8491B4", "#91D1C2", "#DC0000",
]

MIN_CLUSTER_SIZE = 6            # soft constraint (strict min=10 infeasible here)
K_RANGE = range(3, 7)           # {3,4,5,6}
LINKAGE_METHODS = ("ward", "average", "complete")
# Same Sakoe-Chiba band as the 7-day scripts. This keeps "one comparable DTW
# logic" when only the lag horizon changes from 7 to 12 days.
DTW_WINDOW = 3

# ------------------------------------------------------------------------------
# Outlier handling  (Cleveland |max β|=14 and Newark =10.6 have magnitudes that
# dominate DTW distances and do NOT share a common shape with each other).
# ------------------------------------------------------------------------------
#   OUTLIER_METHOD : how we FLAG high-magnitude cities
#       "percentile"  |β|_max > the OUTLIER_PARAM-th percentile (default 95)
#       "tukey"       |β|_max > Q3 + OUTLIER_PARAM*IQR (set PARAM=3 for extreme)
#       "mad"         |β|_max > median + OUTLIER_PARAM*1.4826*MAD
#       "abs"         |β|_max > OUTLIER_PARAM  (legacy hard threshold)
#       "none"        no outlier handling
#
#   OUTLIER_HANDLING : what to DO with flagged cities
#       "assign_nearest"  cluster on core; each outlier is reassigned to the
#                         core cluster whose centroid has the highest Pearson
#                         correlation with that outlier's z-scored shape.
#                         -> every cluster ≥ real size, no n=2 artefacts,
#                            outliers shown as diamonds on the map.
#                         THIS IS THE RECOMMENDED PUBLICATION MODE.
#       "separate"        cluster on core only; outliers form a grey cluster.
#       "winsorise"       clip raw β to ±OUTLIER_PARAM and include all cities.
#       "keep"            do nothing (may produce degenerate partitions).
OUTLIER_METHOD   = "percentile"    # {"percentile", "tukey", "mad", "abs", "none"}
OUTLIER_PARAM    = 95.0            # 95th percentile  (4 cities: Cleveland, Newark, Corpus Christi, Miami)
OUTLIER_ABS_FLOOR = 8.0            # 12d guardrail: catch high-magnitude degeneracy even if P95 shifts
OUTLIER_HANDLING = "assign_nearest"  # {"assign_nearest", "separate", "winsorise", "keep"}

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA
# ═══════════════════════════════════════════════════════════════════════════════
input_path = Path(os.environ.get("DTW_INPUT", MODULE_DIR.parents[1] / "analysis" / "02_ppml_post_event" / "data" / "city_lag_estimates_12day.csv"))
df = pd.read_csv(input_path).dropna(subset=["estimate"])
kc = df[df["city"] == "Kansas City"]
if kc["n_obs"].nunique() > 1:
    df = df.drop(kc[kc["n_obs"] != kc["n_obs"].max()].index).reset_index(drop=True)

cities   = sorted(df["city"].unique())
n_cities = len(cities)
lags     = np.array(sorted(df["lag"].unique()), dtype=int)
n_lags   = len(lags)
print(f"[INFO] {n_cities} cities x {n_lags} lags ({list(lags)})")
print(f"[INFO] DTW_WINDOW = {DTW_WINDOW} (fixed to match 7-day scripts)")

beta_wide = df.pivot(index="city", columns="lag", values="estimate").loc[cities, lags]
se_wide   = df.pivot(index="city", columns="lag", values="std.error").loc[cities, lags]
pval_wide = df.pivot(index="city", columns="lag", values="p.value").loc[cities, lags]
beta = beta_wide.values
se   = se_wide.values
pval = pval_wide.values

# -------------------------------------------------------------------------
# Data-driven outlier identification on |β|_max
# -------------------------------------------------------------------------
maxabs = np.abs(beta).max(axis=1)
if OUTLIER_METHOD == "percentile":
    thr = float(np.percentile(maxabs, OUTLIER_PARAM))
    rule_desc = f"|b|_max > P{OUTLIER_PARAM:.0f} ({thr:.2f})"
elif OUTLIER_METHOD == "tukey":
    q1, q3 = np.percentile(maxabs, [25, 75])
    iqr = q3 - q1
    thr = float(q3 + OUTLIER_PARAM * iqr)
    rule_desc = f"|b|_max > Q3+{OUTLIER_PARAM}*IQR ({thr:.2f})"
elif OUTLIER_METHOD == "mad":
    med = float(np.median(maxabs))
    mad = float(np.median(np.abs(maxabs - med)))
    thr = med + OUTLIER_PARAM * 1.4826 * mad
    rule_desc = f"|b|_max > median+{OUTLIER_PARAM}*1.4826*MAD ({thr:.2f})"
elif OUTLIER_METHOD == "abs":
    thr = float(OUTLIER_PARAM)
    rule_desc = f"|b|_max > {thr:.2f} (fixed)"
else:
    thr = np.inf
    rule_desc = "no outlier rule"

if OUTLIER_ABS_FLOOR is not None:
    floor = float(OUTLIER_ABS_FLOOR)
    extreme_mask = (maxabs > thr) | (maxabs >= floor)
    rule_desc = f"{rule_desc} OR |b|_max >= {floor:.2f}"
else:
    extreme_mask = maxabs > thr
extreme_cities = [cities[i] for i, m in enumerate(extreme_mask) if m]
print(f"[OUTLIER] rule = {rule_desc}")
print(f"[OUTLIER] flagged {int(extreme_mask.sum())} cities: {extreme_cities}")
print(f"[OUTLIER] handling = '{OUTLIER_HANDLING}'")

# For the 'winsorise' mode we clip the magnitudes.  For every other mode we
# keep β as-is (handling happens AFTER clustering).
if OUTLIER_HANDLING == "winsorise" and extreme_mask.any():
    beta_use = np.clip(beta, -thr, thr)
    n_clipped = int(np.sum(np.abs(beta) > thr))
    print(f"[OUTLIER] winsorised {n_clipped} b entries to +/-{thr:.2f}")
else:
    beta_use = beta

# ═══════════════════════════════════════════════════════════════════════════════
# 2a. PRIMARY FEATURE: z-scored sig-filtered beta  (same as Scheme B)
# ═══════════════════════════════════════════════════════════════════════════════
def zscore_rows(mat):
    out = np.zeros_like(mat, dtype=float)
    for i in range(mat.shape[0]):
        s = np.std(mat[i])
        out[i] = (mat[i] - np.mean(mat[i])) / s if s > 1e-6 else 0.0
    return np.nan_to_num(out)

feat_primary = zscore_rows(np.where(pval < 0.05, beta_use, 0.0))

# ═══════════════════════════════════════════════════════════════════════════════
# 2b. SHAPE FEATURES  (for post-hoc interpretation only)
# ═══════════════════════════════════════════════════════════════════════════════
def shape_features(beta_row: np.ndarray, pval_row: np.ndarray, lags: np.ndarray):
    """Return an 8-dim shape feature vector for one city."""
    abs_b   = np.abs(beta_row)
    peak_i  = int(np.argmax(abs_b))
    peak_lag = int(lags[peak_i])
    peak_val = float(beta_row[peak_i])
    # early/late split: first half vs second half of the lag profile.
    # Matches original 7-lag behaviour (mid_lag=3) and adapts to 12 lags (mid_lag=6).
    mid_lag = int(lags[len(lags) // 2 - 1]) if len(lags) >= 2 else int(lags[0])
    early   = float(np.sum(beta_row[lags <= mid_lag]))
    late    = float(np.sum(beta_row[lags >  mid_lag]))
    ratio_el = early / (abs(late) + 1e-6)
    # monotone score: +1 per increasing adjacency, -1 per decreasing
    diffs = np.diff(beta_row)
    monotone = float(np.sum(np.sign(diffs)))
    # OLS slope of beta on lag
    x = lags.astype(float)
    y = beta_row.astype(float)
    xm = x - x.mean()
    ym = y - y.mean()
    decay_slope = float(np.sum(xm * ym) / (np.sum(xm ** 2) + 1e-12))
    n_sig = int(np.sum(pval_row < 0.05))
    return np.array([
        peak_lag, peak_val, early, late, ratio_el,
        monotone, decay_slope, n_sig,
    ], dtype=float)

FEAT_NAMES = [
    "peak_lag", "peak_val", "early_auc", "late_auc",
    "ratio_early_late", "monotone", "decay_slope", "n_sig",
]

raw_shape = np.stack([shape_features(beta_use[i], pval[i], lags) for i in range(n_cities)])
raw_shape = np.nan_to_num(raw_shape, nan=0.0, posinf=0.0, neginf=0.0)
# 5/95 winsorise so extreme betas (Cleveland, Newark) don't dominate the
# auxiliary visualisation. Standardise by median/IQR (RobustScaler).
for j in range(raw_shape.shape[1]):
    lo, hi = np.percentile(raw_shape[:, j], [5, 95])
    raw_shape[:, j] = np.clip(raw_shape[:, j], lo, hi)
shape_std = RobustScaler().fit_transform(raw_shape)

pd.DataFrame(raw_shape, index=cities, columns=FEAT_NAMES).to_csv(
    os.path.join(OUT, "shape_features_raw.csv"))
pd.DataFrame(shape_std, index=cities, columns=FEAT_NAMES).to_csv(
    os.path.join(OUT, "shape_features_standardised.csv"))

# ═══════════════════════════════════════════════════════════════════════════════
# 2c. DTW DISTANCE ON PRIMARY FEATURE
# ═══════════════════════════════════════════════════════════════════════════════
def dtw_distance_matrix(feat_arr: np.ndarray, window: int = DTW_WINDOW) -> np.ndarray:
    """DTW without aggressive pruning (pruning can return +inf on dissimilar pairs)."""
    n = feat_arr.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw.distance(feat_arr[i], feat_arr[j], window=window)
            D[i, j] = D[j, i] = d
    # Replace any residual inf/nan with the finite-max x10 (a safe sentinel).
    finite = D[np.isfinite(D)]
    cap = finite.max() * 10 if finite.size > 0 and finite.max() > 0 else 1.0
    D = np.where(np.isfinite(D), D, cap)
    rng = np.random.RandomState(42)
    jit = rng.uniform(0, 1e-8, D.shape)
    jit = (jit + jit.T) / 2
    np.fill_diagonal(jit, 0)
    return D + jit

# ═══════════════════════════════════════════════════════════════════════════════
# 2d. SPLIT core vs extreme (only if outlier handling is assign_nearest/separate)
# ═══════════════════════════════════════════════════════════════════════════════
cluster_on_subset = OUTLIER_HANDLING in ("assign_nearest", "separate") and extreme_mask.any()
if cluster_on_subset:
    core_idx    = np.where(~extreme_mask)[0]
    extreme_idx = np.where(extreme_mask)[0]
    print(f"[INFO] Clustering on {len(core_idx)} core cities; "
          f"{len(extreme_idx)} flagged cities will be handled post-hoc by "
          f"'{OUTLIER_HANDLING}'.")
else:
    core_idx    = np.arange(n_cities)
    extreme_idx = np.array([], dtype=int)

print("[INFO] Computing DTW distance matrix ...")
D_core = dtw_distance_matrix(feat_primary[core_idx])
finite_core = D_core[(D_core > 0) & np.isfinite(D_core)]
print(f"[INFO] DTW range: [{finite_core.min():.4f}, {finite_core.max():.4f}]   "
      f"(n_pairs={finite_core.size})")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. CONSTRAINED HIERARCHICAL CLUSTERING WITH MULTI-CRITERION k SELECTION
# ═══════════════════════════════════════════════════════════════════════════════
def score_partition(D: np.ndarray, feat_for_euclid: np.ndarray, labels: np.ndarray):
    """silhouette on DTW (precomputed), CH/DB on standardised raw feature (euclidean)."""
    if len(set(labels)) < 2:
        return (-1.0, -1.0, np.inf)
    try:
        sil = silhouette_score(D, labels, metric="precomputed")
    except Exception:
        sil = -1.0
    try:
        ch = calinski_harabasz_score(feat_for_euclid, labels)
    except Exception:
        ch = -1.0
    try:
        db = davies_bouldin_score(feat_for_euclid, labels)
    except Exception:
        db = np.inf
    return sil, ch, db

def search_best(D: np.ndarray,
                feat_for_euclid: np.ndarray,
                methods=LINKAGE_METHODS,
                k_range=K_RANGE,
                min_size=MIN_CLUSTER_SIZE):
    """
    Grid-search (linkage method, k) on a precomputed distance matrix D,
    subject to min_size constraint. Rank by composite (silhouette + CH - DB).
    Auto-fallback: if min_size=10 yields nothing, try 8, 6, 4, 2.
    """
    condensed = squareform(D, checks=False)
    trees = {}
    for m in methods:
        try:
            trees[m] = linkage(condensed, method=m)
        except Exception as e:
            print(f"[WARN] linkage({m}) failed: {e}")

    diag_rows = []
    for m in trees:
        for k in k_range:
            lab = fcluster(trees[m], t=k, criterion="maxclust")
            _, counts = np.unique(lab, return_counts=True)
            diag_rows.append(dict(method=m, k=int(k),
                                  min_size=int(counts.min()),
                                  max_size=int(counts.max())))
    diag = pd.DataFrame(diag_rows)
    print("\n[diagnostic] min-cluster-size at each (method, k):")
    print(diag.pivot(index="method", columns="k", values="min_size").to_string())

    fallbacks = [min_size, 8, 6, 4, 2]
    effective_min = None
    rows = []
    for ms in fallbacks:
        rows = []
        for m in trees:
            for k in k_range:
                lab = fcluster(trees[m], t=k, criterion="maxclust")
                _, counts = np.unique(lab, return_counts=True)
                if len(counts) < 2 or counts.min() < ms:
                    continue
                sil, ch, db = score_partition(D, feat_for_euclid, lab)
                rows.append(dict(method=m, k=int(k),
                                 min_size=int(counts.min()),
                                 max_size=int(counts.max()),
                                 silhouette=sil, calinski_harabasz=ch,
                                 davies_bouldin=db, labels=lab))
        if rows:
            effective_min = ms
            break
    if not rows:
        raise RuntimeError(f"No partition found even with min_size=2 in k_range={list(k_range)}.")
    if effective_min != min_size:
        print(f"[WARN] min_size={min_size} infeasible; fell back to min_size={effective_min}.")
    else:
        print(f"[INFO] min_size satisfied at requested value = {effective_min}.")
    cand = pd.DataFrame(rows)
    cand.attrs["effective_min_size"] = effective_min

    def z(col, higher_better=True):
        v = cand[col].values.astype(float)
        if np.std(v) < 1e-12:
            return np.zeros_like(v)
        zv = (v - v.mean()) / v.std()
        return zv if higher_better else -zv

    cand["composite"] = z("silhouette") + z("calinski_harabasz") + z("davies_bouldin", False)
    cand_sorted = cand.sort_values("composite", ascending=False).reset_index(drop=True)
    best = cand_sorted.iloc[0].to_dict()
    best["Z"] = trees[best["method"]]
    return best, cand_sorted, trees

_active_methods = (FORCE_LINKAGE,) if FORCE_LINKAGE else LINKAGE_METHODS
if FORCE_LINKAGE:
    print(f"\n[FORCE_LINKAGE] grid search restricted to method='{FORCE_LINKAGE}'")
best, cand_table, trees = search_best(
    D_core, feat_primary[core_idx], methods=_active_methods
)

if FORCE_K is not None:
    sub = cand_table[cand_table["k"] == int(FORCE_K)].copy()
    if FORCE_LINKAGE is not None and (sub["method"] == FORCE_LINKAGE).any():
        sub = sub[sub["method"] == FORCE_LINKAGE]
    if len(sub) == 0:
        raise RuntimeError(f"[FORCE_K] No candidate at k={FORCE_K} "
                           f"(linkage={FORCE_LINKAGE}) in cand_table. "
                           f"Available:\n{cand_table[['method','k']].to_string()}")
    winner = sub.sort_values("composite", ascending=False).iloc[0].to_dict()
    winner["Z"] = trees[winner["method"]]
    print(f"\n[FORCE_K] Overriding automatic winner.")
    print(f"  auto-best : method={best['method']}  k={best['k']}  "
          f"sil={best['silhouette']:.3f}  composite={best['composite']:+.3f}")
    print(f"  forced    : method={winner['method']}  k={winner['k']}  "
          f"sil={winner['silhouette']:.3f}  composite={winner['composite']:+.3f}")
    best = winner

core_labels = best["labels"].astype(int)
k_core      = best["k"]
method      = best["method"]
Z           = best["Z"]

labels = np.zeros(n_cities, dtype=int)
labels[core_idx] = core_labels

outlier_assignments = {}     # city -> (target_cluster_id, r_best)
extreme_cluster_id  = None

if extreme_idx.size == 0:
    k = k_core

elif OUTLIER_HANDLING == "assign_nearest":
    # Each outlier goes into the core cluster whose centroid (computed from
    # z-scored raw β — shape only, magnitude removed) is most Pearson-correlated
    # with the outlier's own z-scored shape.
    def _zrow(v):
        v = np.asarray(v, dtype=float)
        s = v.std()
        return (v - v.mean()) / s if s > 1e-8 else np.zeros_like(v)

    core_z = np.stack([_zrow(beta[i]) for i in core_idx])
    centroids_z = np.stack([
        core_z[core_labels == cid].mean(axis=0) for cid in sorted(set(core_labels))
    ])
    cid_list = sorted(set(core_labels))
    print("\n[OUTLIER] shape-based reassignment:")
    for oi in extreme_idx:
        z_o = _zrow(beta[oi])
        if np.std(z_o) < 1e-8:
            labels[oi] = cid_list[0]
            outlier_assignments[cities[oi]] = (cid_list[0], float("nan"))
            print(f"  {cities[oi]:<18s} -> C{cid_list[0]}  (flat shape, default)")
            continue
        rs = np.array([
            float(np.corrcoef(z_o, centroids_z[j])[0, 1]) for j in range(len(cid_list))
        ])
        rs = np.nan_to_num(rs, nan=-1.0)
        j_best = int(np.argmax(rs))
        target = cid_list[j_best]
        labels[oi] = target
        outlier_assignments[cities[oi]] = (target, float(rs[j_best]))
        print(f"  {cities[oi]:<18s} |b|max={maxabs[oi]:5.2f}  "
              f"-> C{target}  (best r={rs[j_best]:+.2f}, all={np.round(rs,2).tolist()})")
    k = k_core

elif OUTLIER_HANDLING == "separate":
    extreme_cluster_id = k_core + 1
    labels[extreme_idx] = extreme_cluster_id
    k = extreme_cluster_id
    print(f"[INFO] Flagged cities ({[cities[i] for i in extreme_idx]}) "
          f"assigned to separate grey cluster C{extreme_cluster_id}.")

else:
    k = k_core

print("\n" + "=" * 72)
print(f"[BEST] linkage={method}, k={k}, min_size={best['min_size']}, max_size={best['max_size']}")
print(f"       silhouette={best['silhouette']:.3f}   "
      f"calinski-harabasz={best['calinski_harabasz']:.1f}   "
      f"davies-bouldin={best['davies_bouldin']:.3f}")
print("=" * 72)
print("\n[k search table] (filtered by min_size >= {})".format(MIN_CLUSTER_SIZE))
print(cand_table.drop(columns=["labels"]).to_string(index=False))

cand_table.drop(columns=["labels"]).to_csv(os.path.join(OUT, "k_search_table.csv"), index=False)

# cluster assignment table
clust_df = pd.DataFrame({"city": cities, "cluster": labels})
clust_df["is_outlier"] = extreme_mask.astype(int)
clust_df["beta_max_abs"] = maxabs
clust_df["outlier_assigned_by"] = [
    f"{OUTLIER_HANDLING} (r={outlier_assignments[c][1]:+.2f})"
    if c in outlier_assignments else ""
    for c in cities
]
for j, name in enumerate(FEAT_NAMES):
    clust_df[name] = raw_shape[:, j]
for i, lag_i in enumerate(lags):
    clust_df[f"beta_lag{lag_i}"] = beta[:, i]
    clust_df[f"pval_lag{lag_i}"] = pval[:, i]
clust_df.to_csv(os.path.join(OUT, "city_cluster_optimized.csv"), index=False)
print(f"[SAVED] city_cluster_optimized.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. PER-CLUSTER QUALITY METRICS
# ═══════════════════════════════════════════════════════════════════════════════
sil_core = silhouette_samples(D_core, core_labels, metric="precomputed")
extreme_idx_set = set(extreme_idx.tolist())
rows = []
for cid in sorted(set(labels)):
    idx = np.where(labels == cid)[0]
    beta_c = beta[idx]
    centroid_beta = beta_c.mean(axis=0)
    corrs = [np.corrcoef(beta_c[i], centroid_beta)[0, 1]
             for i in range(len(idx))
             if np.std(beta_c[i]) > 1e-8 and np.std(centroid_beta) > 1e-8]
    mean_corr = float(np.mean(corrs)) if corrs else 0.0
    rmse = float(np.sqrt(np.mean((beta_c - centroid_beta) ** 2)))
    n_outliers_in_cluster = sum(1 for i in idx if i in extreme_idx_set)

    if cid == extreme_cluster_id:
        mean_sil = float("nan")
        tag = f"separate outlier cluster ({rule_desc})"
    else:
        core_member_rows = np.where(np.isin(core_idx, idx))[0]
        mean_sil = (float(np.mean(sil_core[core_member_rows]))
                    if core_member_rows.size > 0 else float("nan"))
        tag = (f"includes {n_outliers_in_cluster} outlier(s)"
               if n_outliers_in_cluster > 0 else "")
    rows.append(dict(
        cluster=int(cid), n=len(idx), n_outliers=n_outliers_in_cluster, tag=tag,
        silhouette=round(mean_sil, 4) if not np.isnan(mean_sil) else None,
        mean_corr_to_centroid=round(mean_corr, 4),
        rmse_to_centroid=round(rmse, 4),
    ))
qm = pd.DataFrame(rows)
qm.to_csv(os.path.join(OUT, "cluster_quality_metrics_optimized.csv"), index=False)

print("\n--- Per-cluster quality metrics ---")
print(qm.to_string(index=False))

mask_core_qm = qm["cluster"] != (extreme_cluster_id if extreme_cluster_id else -1)
core_qm = qm[mask_core_qm]
N = core_qm["n"].sum()
if N > 0:
    w_sil  = (core_qm["silhouette"]            * core_qm["n"]).sum() / N
    w_corr = (core_qm["mean_corr_to_centroid"] * core_qm["n"]).sum() / N
    w_rmse = (core_qm["rmse_to_centroid"]      * core_qm["n"]).sum() / N
    print(f"\n[WEIGHTED (n={N})] sil={w_sil:+.3f}  "
          f"r_to_centroid={w_corr:.3f}  rmse_to_centroid={w_rmse:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. FIGURE: dendrogram (top) + cluster response curves (bottom)
# ═══════════════════════════════════════════════════════════════════════════════
clust_ids = sorted(set(labels))
ccolors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(clust_ids)}
if extreme_cluster_id is not None:
    ccolors[extreme_cluster_id] = "#888888"

n_bc = min(len(clust_ids), 4)
n_br = int(np.ceil(len(clust_ids) / n_bc))

fig = plt.figure(figsize=(max(16, 4.5 * n_bc), 5.5 + 4 * n_br))
gs = gridspec.GridSpec(1 + n_br, n_bc, height_ratios=[1.4] + [1] * n_br,
                       hspace=0.40, wspace=0.30)

ax_top = fig.add_subplot(gs[0, :])
core_cities_arr = np.array([cities[i] for i in core_idx])
ct = Z[-(k_core - 1), 2] if k_core > 1 else 0
dendrogram(Z, labels=core_cities_arr, leaf_rotation=90, leaf_font_size=6,
           ax=ax_top, above_threshold_color="#AAAAAA", color_threshold=ct)
cmap_city = {cities[i]: ccolors[labels[i]] for i in range(n_cities)}
for lbl in ax_top.get_xticklabels():
    lbl.set_color(cmap_city.get(lbl.get_text(), "#333"))
    lbl.set_fontweight("bold")
ax_top.set_ylabel(f"{method.capitalize()} DTW distance")
ax_top.set_title(
    f"Optimized scheme  (DTW + {method} linkage, k={k_core}, "
    f"sil={best['silhouette']:.3f}, CH={best['calinski_harabasz']:.1f}, "
    f"DB={best['davies_bouldin']:.3f})  "
    f"| {len(extreme_idx)} outlier(s) by {rule_desc}  "
    f"-> {OUTLIER_HANDLING}",
    fontsize=11, fontweight="bold",
)

# Cluster response curves — plotted on the NORMALISED feature that was
# actually fed to DTW (z-scored, significance-filtered, optional winsorise),
# so the visual similarity inside each cluster matches the clustering
# criterion. Magnitude is deliberately removed; for magnitude-aware raw
# beta curves see fig_nature_composite.png (panel c).
feat_plot = np.clip(
    np.nan_to_num(feat_primary, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0,
)
for ci, cid in enumerate(clust_ids):
    r = 1 + ci // n_bc
    c = ci % n_bc
    ax = fig.add_subplot(gs[r, c])
    idx = np.where(labels == cid)[0]
    col = ccolors[cid]
    for i in idx:
        ax.plot(lags, feat_plot[i], color=col, alpha=0.22, linewidth=0.8)
    mean_f = feat_plot[idx].mean(axis=0)
    se_f   = feat_plot[idx].std(axis=0) / max(np.sqrt(len(idx)), 1)
    ax.fill_between(lags, mean_f - 1.96 * se_f, mean_f + 1.96 * se_f,
                    color=col, alpha=0.15)
    ax.plot(lags, mean_f, color=col, linewidth=2.5, zorder=5)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xticks(lags); ax.set_xlabel("Lag (days)")
    if c == 0:
        ax.set_ylabel("Normalised value (z-score, sig-filtered)")
    m = qm.iloc[clust_ids.index(cid)]
    sil_str = f"sil={m['silhouette']:.2f}" if m['silhouette'] is not None else "sil=n/a"
    title = (f"C{cid} (n={m['n']})  {sil_str}  "
             f"r={m['mean_corr_to_centroid']:.2f}  RMSE={m['rmse_to_centroid']:.2f}")
    if m["tag"]:
        title += f"  [{m['tag']}]"
    ax.set_title(title, fontsize=9, fontweight="bold", color=col)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.savefig(os.path.join(OUT, "scheme_optimized_clusters.png"))
fig.savefig(os.path.join(OUT, "scheme_optimized_clusters.pdf"))
plt.close(fig)
print("[SAVED] scheme_optimized_clusters.png/.pdf  (normalised feature view)")

# ═══════════════════════════════════════════════════════════════════════════════
# 5b. NATURE COMPOSITE  (a) DTW distance matrix   (b) silhouette bar
#                       (c-*) cluster response curves   (last) dendrogram
# ═══════════════════════════════════════════════════════════════════════════════
# For the heatmap we need a full-63-city distance matrix (so outliers are
# visible in their assigned rows). If we clustered on the core subset, recompute
# the full DTW here; otherwise reuse D_core (already 63x63).
if cluster_on_subset:
    D_full_for_plot = dtw_distance_matrix(feat_primary)
else:
    D_full_for_plot = D_core

nat_n_bc = min(len(clust_ids), 4)
nat_n_br = int(np.ceil(len(clust_ids) / nat_n_bc))

fig_nat = plt.figure(figsize=(22, 6 + 4.5 * nat_n_br + 5))
gs_nat = gridspec.GridSpec(
    1 + nat_n_br + 1, max(nat_n_bc, 4),
    height_ratios=[1.8] + [1] * nat_n_br + [1.2],
    hspace=0.42, wspace=0.30,
)

# ---- (a) DTW distance heatmap sorted by cluster label ----
ax_a = fig_nat.add_subplot(gs_nat[0, :max(nat_n_bc, 4) - 1])
order_p = np.argsort(labels, kind="stable")
ord_names = [cities[i] for i in order_p]
D_plot = np.nan_to_num(D_full_for_plot.astype(np.float64),
                        nan=0.0, posinf=0.0, neginf=0.0)
ord_D = D_plot[np.ix_(order_p, order_p)]
pos = ord_D[ord_D > 0]
vmax_p = max(float(np.percentile(pos, 99)), 1e-6) if len(pos) > 0 else 1.0
ord_D = np.clip(ord_D, 0.0, vmax_p)
np.fill_diagonal(ord_D, np.nan)
ax_a.imshow(ord_D, cmap="inferno_r", aspect="auto",
            norm=PowerNorm(gamma=0.4, vmin=1e-9, vmax=vmax_p),
            interpolation="nearest")
ax_a.set_xticks(range(n_cities))
ax_a.set_yticks(range(n_cities))
ax_a.set_xticklabels(ord_names, rotation=90, fontsize=5.5)
ax_a.set_yticklabels(ord_names, fontsize=5.5)
for ti, oi in enumerate(order_p):
    col_t = ccolors[labels[oi]]
    ax_a.get_xticklabels()[ti].set_color(col_t)
    ax_a.get_yticklabels()[ti].set_color(col_t)
prev_l = labels[order_p[0]]
for pos_i, oi in enumerate(order_p):
    if labels[oi] != prev_l:
        ax_a.axhline(pos_i - 0.5, color="black", linewidth=0.8)
        ax_a.axvline(pos_i - 0.5, color="black", linewidth=0.8)
        prev_l = labels[oi]
ax_a.set_title("a  DTW distance matrix (sorted by cluster)",
               fontsize=13, fontweight="bold", loc="left")

# ---- (b) Silhouette bar chart (core only) ----
ax_b = fig_nat.add_subplot(gs_nat[0, -1])
y_lo = 0
for cid in clust_ids:
    if cid == extreme_cluster_id:
        members_in_cid = np.where(labels == cid)[0]
        sv = np.zeros(len(members_in_cid))
    else:
        member_rows = np.where(np.isin(core_idx, np.where(labels == cid)[0]))[0]
        sv = np.sort(sil_core[member_rows]) if member_rows.size > 0 else np.array([])
    ax_b.barh(range(y_lo, y_lo + len(sv)), sv, height=1.0,
              color=ccolors[cid], edgecolor="none")
    y_lo += len(sv)
ax_b.axvline(best["silhouette"], color="k", linestyle="--", linewidth=0.8,
             label=f"mean={best['silhouette']:.3f}")
ax_b.set_xlabel("Silhouette coeff.")
ax_b.set_title("b  Silhouette", fontweight="bold", loc="left")
ax_b.set_yticks([])
ax_b.legend(frameon=False, fontsize=8, loc="lower right")
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)

# ---- (c...) Cluster response curves (raw β with significance markers) ----
for ci, cid in enumerate(clust_ids):
    r = 1 + ci // nat_n_bc
    c_col = ci % nat_n_bc
    ax = fig_nat.add_subplot(gs_nat[r, c_col])
    midx = np.where(labels == cid)[0]
    col = ccolors[cid]
    for idx_city in midx:
        city_name = cities[idx_city]
        city_df = (df[df["city"] == city_name]
                   .drop_duplicates(subset="lag")
                   .set_index("lag")
                   .reindex(lags))
        betas_i = city_df["estimate"].values
        pvals_i = city_df["p.value"].values
        ax.plot(lags, betas_i, color=col, alpha=0.25, linewidth=0.8)
        sig_m = (pvals_i < 0.05) & np.isfinite(betas_i)
        ax.scatter(lags[sig_m], betas_i[sig_m], color=col, s=14,
                   zorder=4, edgecolors="white", linewidth=0.3)
    sub_names = [cities[i] for i in midx]
    sub = df[df["city"].isin(sub_names)]
    mean_b = sub.groupby("lag")["estimate"].mean().reindex(lags)
    ax.plot(mean_b.index, mean_b.values, color=col, linewidth=2.5, zorder=5)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xticks(lags); ax.set_xlabel("Lag (days)")
    if c_col == 0:
        ax.set_ylabel("PPML coeff.")
    letter = chr(ord("c") + ci)
    ax.set_title(f"{letter}", fontsize=13, fontweight="bold", loc="left")
    m = qm.iloc[clust_ids.index(cid)]
    sil_s = f"sil={m['silhouette']:.2f}" if m['silhouette'] is not None else "sil=n/a"
    hdr = f"Cluster {cid} (n={m['n']})  {sil_s}  r={m['mean_corr_to_centroid']:.2f}"
    if m["tag"]:
        hdr += f"  [{m['tag']}]"
    ax.text(0.5, 1.01, hdr, transform=ax.transAxes,
            ha="center", fontsize=9, color=col, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ---- (last) Dendrogram (core cities) ----
last_row = 1 + nat_n_br
ax_d = fig_nat.add_subplot(gs_nat[last_row, :])
ct_d = Z[-(k_core - 1), 2] if k_core > 1 else 0
dendrogram(Z, labels=core_cities_arr, leaf_rotation=90, leaf_font_size=6,
           ax=ax_d, above_threshold_color="#AAAAAA", color_threshold=ct_d)
for lbl_x in ax_d.get_xticklabels():
    lbl_x.set_color(cmap_city.get(lbl_x.get_text(), "#333"))
    lbl_x.set_fontweight("bold")
ax_d.set_ylabel(f"{method.capitalize()} DTW distance")
ltr_last = chr(ord("c") + len(clust_ids))
if OUTLIER_HANDLING == "assign_nearest" and extreme_idx.size > 0:
    outlier_note = (f"{len(extreme_idx)} outliers reassigned by shape corr: "
                    + ", ".join(f"{cities[i]}→C{labels[i]}" for i in extreme_idx))
elif OUTLIER_HANDLING == "separate" and extreme_idx.size > 0:
    outlier_note = f"{len(extreme_idx)} outliers in grey cluster C{extreme_cluster_id}"
elif OUTLIER_HANDLING == "winsorise" and extreme_mask.any():
    outlier_note = f"winsorised β to ±{thr:.2f}"
else:
    outlier_note = "no outlier handling"
subtitle = (f"{ltr_last}  Hierarchy (DTW + {method} linkage, k={k_core}; "
            f"outlier rule: {rule_desc}; {outlier_note})")
ax_d.set_title(subtitle, fontweight="bold", loc="left", fontsize=10)
ax_d.spines["top"].set_visible(False)
ax_d.spines["right"].set_visible(False)

fig_nat.savefig(os.path.join(OUT, "fig_nature_composite.png"))
fig_nat.savefig(os.path.join(OUT, "fig_nature_composite.pdf"))
plt.close(fig_nat)
print("[SAVED] fig_nature_composite.png/.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. FEATURE CONTRIBUTION BAR (why-each-cluster-is-what-it-is, for reviewers)
# ═══════════════════════════════════════════════════════════════════════════════
feat_contrib = np.stack([shape_std[labels == cid].mean(axis=0) for cid in clust_ids])
fig2, ax2 = plt.subplots(figsize=(12, 0.55 * len(FEAT_NAMES) + 2))
im = ax2.imshow(feat_contrib.T, cmap="RdBu_r", aspect="auto",
                vmin=-abs(feat_contrib).max(), vmax=abs(feat_contrib).max())
ax2.set_xticks(range(len(clust_ids)))
ax2.set_xticklabels([f"C{c}\n(n={(labels==c).sum()})" for c in clust_ids],
                    fontweight="bold")
ax2.set_yticks(range(len(FEAT_NAMES))); ax2.set_yticklabels(FEAT_NAMES)
for i in range(len(FEAT_NAMES)):
    for j in range(len(clust_ids)):
        ax2.text(j, i, f"{feat_contrib[j, i]:+.2f}",
                 ha="center", va="center", fontsize=8,
                 color="white" if abs(feat_contrib[j, i]) > 1.0 else "black")
ax2.set_title("Standardised shape-feature centroid per cluster",
              fontweight="bold", fontsize=12)
plt.colorbar(im, ax=ax2, fraction=0.035, pad=0.02, label="z-score")
fig2.tight_layout()
fig2.savefig(os.path.join(OUT, "fig_feature_contribution.png"))
fig2.savefig(os.path.join(OUT, "fig_feature_contribution.pdf"))
plt.close(fig2)
print("[SAVED] fig_feature_contribution.png/.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. US MAP
# ═══════════════════════════════════════════════════════════════════════════════
CITY_COORDS = {
    "Abilene": (32.449, -99.733),           "Amarillo": (35.222, -101.831),
    "Arlington": (32.736, -97.108),         "Atlanta": (33.749, -84.388),
    "Austin": (30.267, -97.743),            "Bakersfield": (35.373, -119.019),
    "Baltimore": (39.290, -76.612),         "Boston": (42.361, -71.058),
    "Cape Coral": (26.563, -81.949),        "Chandler": (33.306, -111.841),
    "Charleston": (32.777, -79.931),        "Charlotte": (35.227, -80.843),
    "Chicago": (41.878, -87.630),           "Cincinnati": (39.100, -84.512),
    "Clearwater": (27.966, -82.800),        "Cleveland": (41.499, -81.694),
    "Columbia": (34.000, -81.035),          "Columbus": (39.961, -82.999),
    "Corpus Christi": (27.800, -97.396),    "Dallas": (32.777, -96.797),
    "Detroit": (42.331, -83.046),           "Fort Worth": (32.755, -97.331),
    "Fresno": (36.737, -119.787),           "Gilbert": (33.352, -111.789),
    "Henderson": (36.040, -114.982),        "Houston": (29.760, -95.370),
    "Indianapolis": (39.768, -86.158),      "Jacksonville": (30.332, -81.656),
    "Kansas City": (39.100, -94.578),       "Las Vegas": (36.169, -115.140),
    "Los Angeles": (34.052, -118.244),      "Louisville": (38.253, -85.759),
    "Lubbock": (33.577, -101.855),          "Mesa": (33.415, -111.832),
    "Miami": (25.762, -80.192),             "Milwaukee": (43.039, -87.907),
    "Minneapolis": (44.977, -93.265),       "Miramar": (25.988, -80.235),
    "Nashville": (36.163, -86.781),         "New York": (40.713, -74.006),
    "Newark": (40.736, -74.172),            "Oklahoma City": (35.468, -97.516),
    "Orlando": (28.538, -81.379),           "Overland Park": (38.982, -94.671),
    "Palm Bay": (28.034, -80.588),          "Philadelphia": (39.953, -75.164),
    "Phoenix": (33.449, -112.074),          "Raleigh": (35.780, -78.639),
    "Richmond": (37.541, -77.434),          "Riverside": (33.953, -117.396),
    "Sacramento": (38.582, -121.494),       "San Antonio": (29.424, -98.494),
    "San Bernardino": (34.108, -117.289),   "San Jose": (37.339, -121.895),
    "Scottsdale": (33.494, -111.926),       "St. Louis": (38.627, -90.199),
    "St. Petersburg": (27.771, -82.638),    "Tallahassee": (30.438, -84.281),
    "Tampa": (27.951, -82.457),             "Tucson": (32.222, -110.975),
    "Virginia Beach": (36.853, -75.978),    "Visalia": (36.330, -119.292),
    "Washington": (38.907, -77.037),
}
MISSING_CITIES_NO_PPML = {
    "Aurora": (39.729, -104.832), "Denver": (39.739, -104.990),
    "Hollywood": (26.011, -80.149), "Long Beach": (33.770, -118.194),
    "Oakland": (37.805, -122.271), "Pittsburgh": (40.441, -79.995),
    "Portland": (45.523, -122.676), "Salt Lake City": (40.760, -111.891),
    "San Diego": (32.716, -117.161), "San Francisco": (37.775, -122.419),
    "Santa Ana": (33.746, -117.868), "Seattle": (47.606, -122.332),
}

has_geo = False
try:
    import geopandas as gpd
    us_local = os.path.join(OUT, "us-states.json")
    if not os.path.exists(us_local):
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json",
            us_local)
    us = gpd.read_file(us_local)
    has_geo = True
except Exception as e:
    print(f"[WARN] geopandas: {e}")

fig_m, ax_m = plt.subplots(figsize=(15, 9))
if has_geo:
    us.plot(ax=ax_m, color="#F0F0F0", edgecolor="#888", linewidth=0.5)
city_clust = {cities[i]: int(labels[i]) for i in range(n_cities)}
legend_h = []
for cid in clust_ids:
    n_m = (labels == cid).sum()
    lbl = (f"C{cid}: extreme (n={n_m})"
           if cid == extreme_cluster_id else f"Cluster {cid} (n={n_m})")
    legend_h.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=ccolors[cid],
                           markersize=10, label=lbl))
for cn in cities:
    if cn not in CITY_COORDS:
        continue
    lat, lon = CITY_COORDS[cn]
    cid = city_clust[cn]
    col = ccolors[cid]
    is_ext = cn in extreme_cities
    ax_m.scatter(lon, lat, c=col, s=130, zorder=5,
                 edgecolors="black" if is_ext else "white",
                 linewidth=1.2 if is_ext else 0.6,
                 marker="D" if is_ext else "o")
    ax_m.annotate(cn, (lon, lat), xytext=(lon + 0.45, lat + 0.35),
                  fontsize=5.5, color=col, fontweight="bold",
                  arrowprops=dict(arrowstyle="-", color="#AAA", linewidth=0.3))
wc_col = "#BBBBBB"
for wc_name, (wc_lat, wc_lon) in MISSING_CITIES_NO_PPML.items():
    ax_m.scatter(wc_lon, wc_lat, c=wc_col, s=90, zorder=4,
                 edgecolors="white", linewidth=0.5, marker="s")
    ax_m.annotate(wc_name, (wc_lon, wc_lat),
                  xytext=(wc_lon + 0.45, wc_lat + 0.35),
                  fontsize=5, color="#999999", fontstyle="italic",
                  arrowprops=dict(arrowstyle="-", color="#CCC", linewidth=0.3))
legend_h.append(Line2D([0], [0], marker="s", color="w", markerfacecolor=wc_col,
                       markersize=9,
                       label=f"Outside PPML estimable subset (n={len(MISSING_CITIES_NO_PPML)})"))
ax_m.set_xlim(-125, -66); ax_m.set_ylim(24, 50)
ax_m.set_xlabel("Longitude"); ax_m.set_ylabel("Latitude")
ax_m.set_title(f"Optimized lag-response clusters  "
               f"(DTW + {method} linkage, k={k_core}; outliers = diamonds, "
               f"{rule_desc})",
               fontsize=12, fontweight="bold", pad=12)
ax_m.legend(handles=legend_h, loc="lower left", frameon=True,
            fontsize=9, framealpha=0.9, edgecolor="#CCC")
ax_m.spines["top"].set_visible(False)
ax_m.spines["right"].set_visible(False)
ax_m.set_aspect("equal")
fig_m.tight_layout()
fig_m.savefig(os.path.join(OUT, "fig_optimized_map.png"))
fig_m.savefig(os.path.join(OUT, "fig_optimized_map.pdf"))
plt.close(fig_m)
print("[SAVED] fig_optimized_map.png/.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. HUMAN-READABLE CITY LIST  (csv long form + txt grouped)
#      - one row per (cluster, cities) with n and is_outlier flag
#      - plus the 12 "no-heatwave" cities at the bottom (not clustered)
# ═══════════════════════════════════════════════════════════════════════════════
extreme_set = set(cities[i] for i in extreme_idx.tolist())
long_rows = []
txt_lines = [
    f"### Optimized scheme  (DTW + {method} linkage, k={k_core})",
    f"    Outlier rule : {rule_desc}",
    f"    Handling     : {OUTLIER_HANDLING}",
    f"    Total        : {n_cities} clustered + {len(MISSING_CITIES_NO_PPML)} no-heatwave",
    "",
]
for cid in sorted(set(labels)):
    idx = np.where(labels == cid)[0]
    members_plain    = sorted([cities[i] for i in idx if cities[i] not in extreme_set])
    members_outliers = sorted([cities[i] for i in idx if cities[i] in extreme_set])
    all_members = members_plain + [f"{c}*" for c in members_outliers]
    long_rows.append(dict(
        cluster=int(cid), n=len(idx),
        n_outliers=len(members_outliers),
        cities_core="; ".join(members_plain),
        cities_outlier_reassigned="; ".join(members_outliers),
    ))
    tag = (" [includes {} outlier(s) reassigned by shape]".format(len(members_outliers))
           if members_outliers else "")
    txt_lines.append(f"Cluster {cid}  (n={len(idx)}){tag}:")
    txt_lines.append("    " + ", ".join(all_members))
    txt_lines.append("")
no_hw = sorted(MISSING_CITIES_NO_PPML.keys())
txt_lines.append(f"[not clustered - no heatwave events]  (n={len(no_hw)}):")
txt_lines.append("    " + ", ".join(no_hw))
txt_lines.append("")
txt_lines.append("Note: cities marked with '*' were flagged as magnitude outliers "
                 "(|b|_max above the data-driven threshold) and reassigned to the core "
                 "cluster whose centroid shape (z-scored) is most Pearson-correlated "
                 "with their own shape.")

pd.DataFrame(long_rows).to_csv(
    os.path.join(OUT, "cluster_city_list_optimized.csv"), index=False, encoding="utf-8"
)
with open(os.path.join(OUT, "cluster_city_list_optimized.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(txt_lines))
print("[SAVED] cluster_city_list_optimized.csv  /  cluster_city_list_optimized.txt")

print(f"\n[DONE] All outputs -> {OUT}")
