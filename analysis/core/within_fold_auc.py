"""AUC using only positive-negative comparisons within held-out folds."""
import numpy as np
from sklearn.metrics import roc_auc_score


def summarize(labels, predictions, folds):
    """Return aggregate point estimates; predictions are aligned full-cohort arrays.

    Fold AUCs weight tied scores by one half. The overall within-fold estimate
    weights each fold by its number of positive-negative pairs.
    """
    y = np.asarray(labels)
    q = np.asarray(predictions, dtype=float)
    if y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be a one-dimensional binary array")
    if q.ndim == 1:
        q = q[None, :]
    if q.ndim != 2 or q.shape[1] != len(y) or not len(q):
        raise ValueError("Predictions must have shape (models, participants)")
    if not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
        raise ValueError("Probabilities must be finite and between zero and one")
    tests = [np.asarray(f["test"]) for f in folds]
    if not tests or any(t.ndim != 1 or t.dtype.kind not in "iu" for t in tests):
        raise ValueError("Fold test indices must be one-dimensional integer arrays")
    seen = np.concatenate(tests)
    if not np.array_equal(np.sort(seen), np.arange(len(y))):
        raise ValueError("Test folds must partition every participant exactly once")
    rows = []
    for fold, ids in zip(folds, tests):
        n_positive = int(y[ids].sum())
        n_negative = len(ids) - n_positive
        if not n_positive or not n_negative:
            raise ValueError("Every test fold must contain both classes")
        rows.append({"fold": int(fold["fold"]), "n_positive": n_positive,
                     "n_negative": n_negative,
                     "auc": [float(roc_auc_score(y[ids], p[ids])) for p in q]})
    auc = np.array([r["auc"] for r in rows])
    weights = np.array([r["n_positive"] * r["n_negative"] for r in rows])
    pooled = [float(roc_auc_score(y, p)) for p in q]
    weighted = np.average(auc, axis=0, weights=weights).tolist()
    for row in rows:
        row["delta"] = row["auc"][0] - row["auc"][1] if len(q) == 2 else None
    return {"pooled_auc": pooled, "macro_within_fold_auc": auc.mean(axis=0).tolist(),
            "pair_weighted_within_fold_auc": weighted,
            "pooled_delta": pooled[0] - pooled[1] if len(q) == 2 else None,
            "within_fold_delta": weighted[0] - weighted[1] if len(q) == 2 else None,
            "folds": rows}


def sensitivity(labels, predictions, jobs, original_folds):
    """Select the reported comparisons from the core report's loaded predictions."""
    arms = ["linked_basic", "average_basic"]
    specs = [("main_ensemble", [f"fixed30/eegnet/{a}/ensemble3/split20261109/final" for a in arms], original_folds)]
    for split in [20261110, 20261111]:
        folds = [j for j in jobs.values() if j["mode"] == "split30" and
                 j["architecture"] == "eegnet" and j["arm"] == "linked_basic" and
                 j["seed"] == 1109 and j["split_seed"] == split]
        specs.append((f"alternative_partition_{split}", [f"split30/eegnet/{a}/seed1109/split{split}/final" for a in arms], folds))
    for epoch in [30, 100]:
        specs.append((f"long_trajectory_{epoch}", [f"long100/eegnet/{a}/seed1109/split20261109/{epoch}" for a in arms], original_folds))
    specs.append(("spectral", [f"spectral/spectral_{a}" for a in arms], original_folds))
    comparisons, unavailable = {}, {}
    for name, keys, folds in specs:
        if not all(k in predictions for k in keys) or not folds:
            unavailable[name] = "Complete saved predictions and corresponding folds required"
            continue
        values = []
        for key in keys:
            ids, scores = predictions[key]
            ids = np.asarray(ids)
            if not np.array_equal(np.sort(ids), np.arange(len(labels))):
                raise ValueError("Sensitivity requires complete full-cohort predictions")
            q = np.empty(len(labels)); q[ids] = scores
            values.append(q)
        comparisons[name] = summarize(labels, values, sorted(folds, key=lambda f: f["fold"]))
    return {"scope": "Saved-probability point estimates; no training or confidence intervals. Folds share training observations and are not independent replicates.",
            "model_order": ["linked_mastoid", "common_average"],
            "weighting": "Number of positive-negative pairs in each held-out fold",
            "comparisons": comparisons, "unavailable": unavailable}
