"""Baseline classifiers (16) and their hyperparameters."""
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from .config import *
from .features import *
from .ft_transformer import FTClassifier

class CatWrap(BaseEstimator, ClassifierMixin):
    """Thin sklearn wrapper (avoids CatBoost/sklearn tag issues in Pipeline/Stacking)."""
    def __init__(self, iterations=800, depth=6, learning_rate=0.05, random_state=42):
        self.iterations, self.depth, self.learning_rate, self.random_state = iterations, depth, learning_rate, random_state
    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        self.model_ = CatBoostClassifier(iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
                                         loss_function="MultiClass", random_seed=self.random_state, verbose=0,
                                         allow_writing_files=False, thread_count=-1)
        self.model_.fit(X, y, sample_weight=sample_weight); return self
    def predict_proba(self, X): return self.model_.predict_proba(X)
    def predict(self, X): return self.classes_[self.predict_proba(X).argmax(1)]

def make_baselines(ds):
    S = CFG["SEED"]
    xgb = lambda: XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                                objective="multi:softprob", eval_metric="mlogloss", random_state=S, n_jobs=-1, verbosity=0)
    P = lambda clf, kf=False: Pipeline([("prep", ClinicalPreprocessor(ds, use_kf=kf, scale=True)), ("clf", clf)])
    return {
        "Logistic Regression": lambda: P(LogisticRegression(max_iter=3000, C=1.0)),
        "KNN":                 lambda: P(KNeighborsClassifier(n_neighbors=7, weights="distance")),
        "SVM (RBF)":           lambda: P(SVC(C=10, gamma="scale", probability=True, random_state=S)),
        "Decision Tree":       lambda: P(DecisionTreeClassifier(random_state=S)),
        "Naive Bayes":         lambda: P(GaussianNB()),
        "Random Forest":       lambda: P(RandomForestClassifier(n_estimators=500, random_state=S, n_jobs=-1)),
        "Extra Trees":         lambda: P(ExtraTreesClassifier(n_estimators=500, random_state=S, n_jobs=-1)),
        "AdaBoost":            lambda: P(AdaBoostClassifier(n_estimators=300, learning_rate=0.5, random_state=S)),
        "Gradient Boosting":   lambda: P(GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=S)),
        "XGBoost":             lambda: P(xgb()),
        "LightGBM":            lambda: P(LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, random_state=S, verbose=-1, n_jobs=-1)),
        "CatBoost":            lambda: P(CatWrap(random_state=S)),
        "MLP":                 lambda: P(MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=2000, early_stopping=True, random_state=S)),
        "FT-Transformer":      lambda: P(FTClassifier(ordinal=False, seed=S)),
        "Stacking (Lit.)":     lambda: P(StackingClassifier(
                                    estimators=[("rf", RandomForestClassifier(n_estimators=300, random_state=S, n_jobs=-1)),
                                                ("xgb", xgb()),
                                                ("gb", GradientBoostingClassifier(n_estimators=200, random_state=S))],
                                    final_estimator=xgb(), cv=StratifiedKFold(5, shuffle=True, random_state=S),
                                    stack_method="predict_proba", n_jobs=1)),
        "CatBoost + KF":       lambda: P(CatWrap(random_state=S), kf=True),
    }

def hyperparam_table():
    """Table T03: hyperparameters of every model."""
    return pd.DataFrame([
        ("Logistic Regression", "C=1.0, max_iter=3000, L2"), ("KNN", "k=7, distance weighting"),
        ("SVM (RBF)", "C=10, gamma=scale, Platt probabilities"), ("Decision Tree", "unpruned, gini"),
        ("Naive Bayes", "Gaussian"), ("Random Forest", "500 trees"), ("Extra Trees", "500 trees"),
        ("AdaBoost", "300 estimators, lr=0.5"), ("Gradient Boosting", "300 est., lr=0.05, depth=3"),
        ("XGBoost", "400 est., depth=6, lr=0.05, subsample=0.9, colsample=0.9"),
        ("LightGBM", "400 est., lr=0.05, num_leaves=31"), ("CatBoost", "800 it., depth=6, lr=0.05"),
        ("MLP", "128-64, ReLU, Adam, early stopping"),
        ("FT-Transformer", f"d={CFG['FT']['d_token']}, layers={CFG['FT']['n_layers']}, heads={CFG['FT']['n_heads']}, dropout={CFG['FT']['dropout']}, AdamW lr={CFG['FT']['lr']}, early stop (patience {CFG['FT']['patience']})"),
        ("Stacking (Lit.)", "RF(300)+XGB+GB(200) -> XGB meta, 5-fold inner CV"),
        ("CatBoost + KF", "CatBoost + clinical knowledge features"),
        (PROP, f"Branches: FT (ordinal CORAL) + CatBoost (early-stopped, then refit) + GB(300, lr 0.05, depth 3); inner {CFG['INNER_FOLDS']} group-CV OOF stacking -> multinomial LR meta (C=1); cross-conformal (LAC + Mondrian); high-class cost x{CFG['HIGH_COST']}"),
    ], columns=["Model", "Hyperparameters"])
