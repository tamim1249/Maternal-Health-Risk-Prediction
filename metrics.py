"""Discrimination, safety and calibration metrics."""
from .config import *

def ece_score(y, P, n_bins=15):
    conf, acc = P.max(1), (P.argmax(1) == y).astype(float)
    bins, e = np.linspace(0, 1, n_bins + 1), 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e

def compute_metrics(y, P, K, high):
    y = np.asarray(y); pred = P.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=list(range(K)), average="macro", zero_division=0)
    pc, rc, _, _ = precision_recall_fscore_support(y, pred, labels=list(range(K)), average=None, zero_division=0)
    try: aucv = roc_auc_score(y, P, multi_class="ovr", average="macro", labels=list(range(K)))
    except Exception: aucv = np.nan
    Y = np.eye(K)[y]
    return dict(Accuracy=accuracy_score(y, pred), BalancedAcc=balanced_accuracy_score(y, pred),
                MacroP=p, MacroR=r, MacroF1=f, MCC=matthews_corrcoef(y, pred),
                Kappa=cohen_kappa_score(y, pred), QWK=cohen_kappa_score(y, pred, weights="quadratic"),
                AUC=aucv, HighRecall=rc[high], HighPrecision=pc[high],
                UnderTriage=float(np.mean(pred < y)),               # severity under-estimation rate
                ECE=ece_score(y, P), Brier=float(np.mean(np.sum((P - Y) ** 2, 1))),
                LogLoss=log_loss(y, np.clip(P, 1e-12, 1), labels=list(range(K))))

METRIC_COLS = ["Accuracy", "BalancedAcc", "MacroP", "MacroR", "MacroF1", "MCC", "Kappa", "QWK", "AUC",
               "HighRecall", "HighPrecision", "UnderTriage", "ECE", "Brier", "LogLoss"]

