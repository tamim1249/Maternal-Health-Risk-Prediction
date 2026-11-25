"""Global configuration, shared imports and I/O helpers for OCH-MRNet experiments."""
import os, time, math, json, random, warnings, itertools, copy
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from scipy.optimize import minimize, minimize_scalar
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, StackingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
                             matthews_corrcoef, cohen_kappa_score, roc_auc_score, log_loss,
                             confusion_matrix, roc_curve, precision_recall_curve, auc, f1_score, recall_score)
warnings.filterwarnings("ignore")

CFG = dict(
    SEED=42,
    OUT="outputs",
    MATERNAL_PATH="data/Maternal Health Risk Data Set.csv",
    FETAL_PATH="data/fetal_health.csv",
    N_SPLITS=5,
    N_REPEATS={"maternal": 5, "fetal": 3},   # Protocol B (leakage-free) repeats
    N_REPEATS_A=3,                           # Protocol A (naive, literature-style) repeats
    RUN_FETAL=True,
    RUN_ABLATION=True,
    ABL_REPEATS={"maternal": 2, "fetal": 1},   # repeats on which retrained ablations run
    INNER_FOLDS={"maternal": 4, "fetal": 3},   # inner CV for out-of-fold stacking
    HIGH_COST=2.0,                           # extra cost on the highest-risk class
    CONF_ALPHAS=[0.05, 0.10, 0.15, 0.20],
    MAIN_ALPHA=0.10,
    NOISE_LEVELS=[0.0, 0.05, 0.10, 0.20, 0.30],
    FT=dict(d_token=32, n_layers=3, n_heads=4, dropout=0.15, lr=1e-3, wd=1e-4,
            batch=64, max_epochs=400, patience=40),
    FAST_MODE=False,                         # True = quick dry run (1 repeat, fewer epochs)
    SHOW_FIGURES=False,                      # True inside notebooks
)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "axes.titleweight": "bold"})
PROP = "OCH-MRNet (Proposed)"
PROP_COLOR, BASE_COLOR = "#C0392B", "#5D6D7E"

def set_seed(s):
    random.seed(s); np.random.seed(s)
    try:
        import torch; torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    except Exception: pass

def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(CFG["FIG"], f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(CFG["FIG"], f"{name}.pdf"), bbox_inches="tight")
    if CFG["SHOW_FIGURES"]: plt.show()
    plt.close(fig)
    print(f"[saved figure] {name}")

def save_table(df, name, index=False):
    df.to_csv(os.path.join(CFG["TAB"], f"{name}.csv"), index=index)
    try:
        df.to_latex(os.path.join(CFG["TAB"], f"{name}.tex"), index=index, escape=True)
    except Exception as e:
        print("  (latex export skipped:", e, ")")
    print(f"[saved table] {name}  shape={df.shape}")


def configure(**overrides):
    """Update CFG in place (keys as in CFG, case-insensitive), create output folders and seed everything."""
    for k, v in overrides.items():
        if v is not None: CFG[k.upper()] = v
    if CFG["FAST_MODE"]:
        CFG["N_REPEATS"] = {"maternal": 1, "fetal": 1}; CFG["N_REPEATS_A"] = 1
        CFG["ABL_REPEATS"] = {"maternal": 1, "fetal": 1}
        CFG["FT"].update(max_epochs=60, patience=10)
    CFG["FIG"] = os.path.join(CFG["OUT"], "figures"); CFG["TAB"] = os.path.join(CFG["OUT"], "tables")
    os.makedirs(CFG["FIG"], exist_ok=True); os.makedirs(CFG["TAB"], exist_ok=True)
    set_seed(CFG["SEED"])
    return CFG
