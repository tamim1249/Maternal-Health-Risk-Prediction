"""OCH-MRNet: cost-sensitive ordinal hybrid with out-of-fold stacking and cross-conformal triage sets."""
from catboost import CatBoostClassifier
from .config import *
from .features import *
from .ft_transformer import train_ft, ft_predict

def nll(p, y): return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))

def make_weights(y, K, high_cost, balanced=True):
    if not balanced: return np.ones(len(y))
    cnt = np.bincount(y, minlength=K).astype(float); w = len(y) / (K * np.maximum(cnt, 1))
    w[K - 1] *= high_cost; sw = w[y]
    return sw / sw.mean()

def conformal_q(scores, alpha):
    n = len(scores)
    if n == 0 or math.ceil((n + 1) * (1 - alpha)) > n: return 1.0
    return float(np.quantile(scores, math.ceil((n + 1) * (1 - alpha)) / n, method="higher"))

META_SETS = {"full": ["ft", "cat", "gb"], "tree_only": ["cat", "gb"], "no_gb": ["ft", "cat"]}
def meta_X(P, keys): return np.hstack([np.log(np.clip(P[k], 1e-6, 1)) for k in keys])
def new_meta(): return LogisticRegression(max_iter=5000, C=1.0)

class OCHMRNet:
    """Knowledge features -> three heterogeneous branches [FT-Transformer (ordinal CORAL), CatBoost, Gradient Boosting]
       trained with inner group-CV -> cost-sensitive out-of-fold stacking meta-learner -> branches refit on the
       WHOLE training fold (no data held out) -> cross-conformal prediction sets (marginal + class-conditional)."""
    def __init__(self, dataset, K, use_kf=True, ordinal=True, cost=True, seed=42):
        self.dataset, self.K, self.use_kf, self.ordinal, self.cost, self.seed = dataset, K, use_kf, ordinal, cost, seed

    def _fit_branches(self, Za, ya, sw, Zv=None, yv=None):
        ft = train_ft(Za, ya, Zv, yv, self.K, self.ordinal, sw, self.seed,
                      fixed_epochs=None if Zv is not None else self.ft_epochs_)
        kw = dict(depth=6, learning_rate=0.03, loss_function="MultiClass", random_seed=self.seed, verbose=0,
                  allow_writing_files=False, thread_count=-1)
        if Zv is not None:
            cat = CatBoostClassifier(iterations=2000, early_stopping_rounds=100, **kw)
            cat.fit(Za, ya, sample_weight=sw, eval_set=(Zv, yv))
        else:
            cat = CatBoostClassifier(iterations=self.cat_iters_, **kw); cat.fit(Za, ya, sample_weight=sw)
        gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                                        random_state=self.seed).fit(Za, ya, sample_weight=sw)
        return dict(ft=ft, cat=cat, gb=gb)

    def _probs(self, M, Z):
        return dict(ft=ft_predict(M["ft"], Z), cat=M["cat"].predict_proba(Z), gb=M["gb"].predict_proba(Z))

    def fit(self, X, y, groups):
        y = np.asarray(y); groups = np.asarray(groups); t0 = time.time()
        self.prep_ = ClinicalPreprocessor(self.dataset, self.use_kf, True).fit(X)
        Z = self.prep_.transform(X)
        sw = make_weights(y, self.K, CFG["HIGH_COST"], balanced=self.cost)
        # 1) inner group-CV -> out-of-fold branch probabilities for EVERY training row
        oof = {k: np.zeros((len(y), self.K)) for k in ["ft", "cat", "gb"]}; eps, its = [], []
        inner = StratifiedGroupKFold(CFG["INNER_FOLDS"][self.dataset], shuffle=True, random_state=self.seed)
        for a, v in inner.split(Z, y, groups):
            M = self._fit_branches(Z[a], y[a], sw[a], Z[v], y[v]); P = self._probs(M, Z[v])
            for k in oof: oof[k][v] = P[k]
            eps.append(M["ft"].best_epoch_ + 1); its.append(max(50, M["cat"].get_best_iteration() + 1))
        self.ft_epochs_, self.cat_iters_ = int(np.median(eps)), int(np.median(its))
        # 2) class-balanced stacking meta-learners on OOF log-probabilities
        msw = make_weights(y, self.K, 1.0, balanced=self.cost)          # meta: class-balanced only (cost already in branches)
        self.metas_ = {n: new_meta().fit(meta_X(oof, ks), y, sample_weight=msw) for n, ks in META_SETS.items()}
        # 3) cross-conformal scores: meta re-fitted by cross-fitting so every score is out-of-sample
        Pcv = np.zeros((len(y), self.K)); F = meta_X(oof, META_SETS["full"])
        for a, v in StratifiedGroupKFold(5, shuffle=True, random_state=self.seed + 7).split(F, y, groups):
            Pcv[v] = new_meta().fit(F[a], y[a], sample_weight=msw[a]).predict_proba(F[v])
        self.cal_scores_ = 1 - Pcv[np.arange(len(y)), y]; self.cal_y_ = y
        # 4) refit all branches on the whole training fold
        self.M_ = self._fit_branches(Z, y, sw)
        self.fit_time_ = time.time() - t0
        return self

    def variants(self, X):
        Z = self.prep_.transform(X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.prep_.cols_))
        P = self._probs(self.M_, Z)
        out = {n: self.metas_[n].predict_proba(meta_X(P, ks)) for n, ks in META_SETS.items()}
        out["avg"] = (P["ft"] + P["cat"] + P["gb"]) / 3; out["ft_only"] = P["ft"]
        return out

    def predict_proba(self, X): return self.variants(X)["full"]

    def predict_sets(self, P, alpha, mondrian=False):
        if not mondrian:
            return (1 - P) <= conformal_q(self.cal_scores_, alpha)
        qs = np.array([conformal_q(self.cal_scores_[self.cal_y_ == k], alpha) for k in range(self.K)])
        return (1 - P) <= qs[None, :]

    def n_params(self):
        return int(self.M_["ft"].n_params_), int(self.M_["cat"].tree_count_)


