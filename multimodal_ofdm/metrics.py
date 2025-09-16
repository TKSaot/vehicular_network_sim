
# multimodal_ofdm/metrics.py
import numpy as np

def psnr(a: np.ndarray, b: np.ndarray, data_range: int = 255) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(data_range) - 10 * np.log10(mse))

def ssim(a: np.ndarray, b: np.ndarray, data_range: int = 255,
         K1: float = 0.01, K2: float = 0.03) -> float:
    """
    簡易 SSIM：画素全体での平均・分散・共分散から 1 値を出す（窓畳み込みなし）．
    依存ライブラリを増やさず“良し悪し”を定量化する目的に十分．
    """
    x = a.astype(np.float64)
    y = b.astype(np.float64)
    L = float(data_range)
    c1 = (K1 * L) ** 2
    c2 = (K2 * L) ** 2
    mx = x.mean()
    my = y.mean()
    vx = x.var()
    vy = y.var()
    vxy = np.mean((x - mx) * (y - my))
    num = (2 * mx * my + c1) * (2 * vxy + c2)
    den = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(num / den) if den != 0 else 1.0

def f1_binary_edge(t: np.ndarray, p: np.ndarray) -> float:
    """
    t, p: 0/255 または 0/1 の 2値画像．
    """
    t = (t > 0).astype(np.uint8)
    p = (p > 0).astype(np.uint8)
    tp = np.logical_and(t, p).sum()
    fp = np.logical_and(1 - t, p).sum()
    fn = np.logical_and(t, 1 - p).sum()
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    if (prec + rec) == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))

def miou_from_ids(gt: np.ndarray, pred: np.ndarray, K: int) -> float:
    """
    0..K-1 のクラス ID マップから mIoU を計算．
    """
    gt = gt.reshape(-1)
    pred = pred.reshape(-1)
    ious = []
    for c in range(K):
        g = (gt == c)
        p = (pred == c)
        inter = np.logical_and(g, p).sum()
        union = np.logical_or(g, p).sum()
        if union == 0:
            continue
        ious.append(inter / union)
    if not ious:
        return 0.0
    return float(np.mean(ious))
