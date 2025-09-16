
from __future__ import annotations
import numpy as np

def psnr(a: np.ndarray, b: np.ndarray, data_range: int = 255) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0: return float("inf")
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(mse)

def miou_from_ids(y_true: np.ndarray, y_pred: np.ndarray, K: int | None = None) -> float:
    y_true = y_true.astype(np.int64); y_pred = y_pred.astype(np.int64)
    if K is None:
        K = max(int(y_true.max())+1, int(y_pred.max())+1)
    ious = []
    for c in range(K):
        t = (y_true == c); p = (y_pred == c)
        inter = np.logical_and(t, p).sum()
        union = np.logical_or(t, p).sum()
        if union == 0: continue
        ious.append(inter / union)
    if not ious: return 0.0
    return float(np.mean(ious))

def f1_binary_edge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    t = (y_true > 0); p = (y_pred > 0)
    tp = np.logical_and(t, p).sum()
    fp = np.logical_and(~t, p).sum()
    fn = np.logical_and(t, ~p).sum()
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)
    if (prec + rec) == 0: return 0.0
    return 2 * prec * rec / (prec + rec)
