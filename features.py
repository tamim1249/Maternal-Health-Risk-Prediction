"""Clinical knowledge features and the fold-safe preprocessor."""
from .config import *

PLAUSIBLE = {"maternal": {"HeartRate": (40, 200), "SystolicBP": (60, 250), "DiastolicBP": (30, 150)},
             "fetal": {}}

def kf_maternal(X):
    X = X.copy()
    X["MAP"] = (X["SystolicBP"] + 2*X["DiastolicBP"]) / 3
    X["PulsePressure"] = X["SystolicBP"] - X["DiastolicBP"]
    X["ShockIndex"] = X["HeartRate"] / X["SystolicBP"]
    X["BP_Ratio"] = X["SystolicBP"] / X["DiastolicBP"]
    X["HTN"] = ((X["SystolicBP"] >= 140) | (X["DiastolicBP"] >= 90)).astype(float)
    X["SevereHTN"] = ((X["SystolicBP"] >= 160) | (X["DiastolicBP"] >= 110)).astype(float)
    X["HyperGly"] = (X["BS"] >= 7.8).astype(float)
    X["DiabRange"] = (X["BS"] >= 11.1).astype(float)
    X["Fever"] = (X["BodyTemp"] >= 100.4).astype(float)
    X["AdvAge"] = (X["Age"] >= 35).astype(float)
    X["Teen"] = (X["Age"] < 20).astype(float)
    X["ClinicalFlagCount"] = X[["HTN", "HyperGly", "Fever", "AdvAge", "Teen"]].sum(1)
    return X

def kf_fetal(X):
    X = X.copy()
    X["TotalDecel"] = X["light_decelerations"] + X["severe_decelerations"] + X["prolongued_decelerations"]
    X["SevereOrProlonged"] = ((X["severe_decelerations"] + X["prolongued_decelerations"]) > 0).astype(float)
    X["AccelPresent"] = (X["accelerations"] > 0).astype(float)
    X["BaselineAbn"] = ((X["baseline_value"] < 110) | (X["baseline_value"] > 160)).astype(float)
    X["HighASTV"] = (X["abnormal_short_term_variability"] > 60).astype(float)
    X["HighALTV"] = (X["percentage_of_time_with_abnormal_long_term_variability"] >= 20).astype(float)
    X["DecelPerContraction"] = X["TotalDecel"] / (X["uterine_contractions"] + 1e-3)
    X["HistSkew"] = X["histogram_mean"] - X["histogram_median"]
    X["ClinicalFlagCount"] = X[["SevereOrProlonged", "BaselineAbn", "HighASTV", "HighALTV"]].sum(1) + (1 - X["AccelPresent"])
    return X

KF_FN = {"maternal": kf_maternal, "fetal": kf_fetal}

class ClinicalPreprocessor(BaseEstimator, TransformerMixin):
    """Fold-safe: plausibility check -> median impute (train stats) -> optional knowledge features -> optional scaling."""
    def __init__(self, dataset="maternal", use_kf=False, scale=True):
        self.dataset, self.use_kf, self.scale = dataset, use_kf, scale
    def _clean(self, X):
        X = X.copy()
        for c, (lo, hi) in PLAUSIBLE[self.dataset].items():
            if c in X: X.loc[(X[c] < lo) | (X[c] > hi), c] = np.nan
        return X
    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=getattr(self, "cols_", None) or list(X.columns))
        self.cols_ = list(X.columns)
        Xc = self._clean(X); self.medians_ = Xc.median()
        Xc = Xc.fillna(self.medians_)
        if self.use_kf: Xc = KF_FN[self.dataset](Xc)
        self.feature_names_out_ = list(Xc.columns)
        self.scaler_ = StandardScaler().fit(Xc.values) if self.scale else None
        return self
    def transform(self, X):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.cols_)
        Xc = self._clean(X[self.cols_]).fillna(self.medians_)
        if self.use_kf: Xc = KF_FN[self.dataset](Xc)
        A = Xc.values.astype(np.float32)
        return self.scaler_.transform(A).astype(np.float32) if self.scale else A

