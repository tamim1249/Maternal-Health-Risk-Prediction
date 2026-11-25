"""Dataset loading and duplicate / label-conflict audit."""
from .config import *

def load_maternal(path):
    df = pd.read_csv(path); df.columns = [c.strip() for c in df.columns]
    df["RiskLevel"] = df["RiskLevel"].astype(str).str.strip().str.lower()
    y = df["RiskLevel"].map({"low risk": 0, "mid risk": 1, "high risk": 2})
    assert y.notna().all(), "Unexpected RiskLevel labels"
    return df.drop(columns="RiskLevel").astype(float), y.astype(int)

def load_fetal(path):
    df = pd.read_csv(path); df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    y = df["fetal_health"].astype(int) - 1
    return df.drop(columns="fetal_health").astype(float), y

def build_dataset(name, X, y, class_names):
    full = X.copy(); full["__y"] = y.values
    dup_mask = full.duplicated()
    d = full.loc[~dup_mask].reset_index(drop=True)
    Xd, yd = d.drop(columns="__y"), d["__y"].astype(int).values
    groups = pd.factorize(pd.util.hash_pandas_object(Xd, index=False))[0]   # identical feature vectors -> same group
    n_label_per_vec = pd.Series(yd).groupby(groups).nunique()
    conflict_groups = int((n_label_per_vec > 1).sum())
    rows_in_conflict = int(pd.Series(groups).isin(n_label_per_vec[n_label_per_vec > 1].index).sum())
    best_per_vec = pd.Series(yd).groupby(groups).agg(lambda s_: s_.value_counts().iloc[0])
    ceiling = float(best_per_vec.sum() / len(yd))      # max accuracy any model can reach on dedup data
    audit = dict(Dataset=name, Raw_rows=len(X), Exact_duplicates=int(dup_mask.sum()),
                 Duplicate_pct=round(dup_mask.mean()*100, 2), Unique_rows=len(Xd),
                 Unique_feature_vectors=int(len(np.unique(groups))),
                 Conflicting_label_groups=conflict_groups, Rows_in_conflict_groups=rows_in_conflict,
                 Label_noise_accuracy_ceiling=round(ceiling, 4))
    for k, c in enumerate(class_names):
        audit[f"Raw_{c}"] = int((y.values == k).sum()); audit[f"Dedup_{c}"] = int((yd == k).sum())
    return dict(name=name, X=X.reset_index(drop=True), y=y.values, Xd=Xd, yd=yd, gd=groups,
                classes=class_names, K=len(class_names), high=len(class_names)-1, audit=audit)


def prepare_data():
    """Load both datasets, run the duplicate/label-conflict audit and save Tables T01/T02."""
    DATA = {}
    Xm, ym = load_maternal(CFG["MATERNAL_PATH"])
    DATA["maternal"] = build_dataset("Maternal Health Risk", Xm, ym, ["Low", "Mid", "High"])
    if CFG["RUN_FETAL"]:
        Xf, yf = load_fetal(CFG["FETAL_PATH"])
        DATA["fetal"] = build_dataset("Fetal Health (CTG)", Xf, yf, ["Normal", "Suspect", "Pathological"])

    audit_df = pd.DataFrame([D["audit"] for D in DATA.values()]).T.reset_index()
    audit_df.columns = ["Item"] + [D["name"] for D in DATA.values()]
    save_table(audit_df, "T02_duplicate_audit")
    print(audit_df.to_string(index=False))

    # Table 1: dataset description
    desc_rows = []
    for key, D in DATA.items():
        for c in D["Xd"].columns:
            s = D["Xd"][c]
            desc_rows.append(dict(Dataset=D["name"], Feature=c, Mean=round(s.mean(), 3), SD=round(s.std(), 3),
                                  Min=s.min(), Median=s.median(), Max=s.max(), Unique=s.nunique()))
    save_table(pd.DataFrame(desc_rows), "T01_dataset_description")
    print("Implausible HeartRate (<40 bpm) rows in maternal raw data:", int((Xm["HeartRate"] < 40).sum()))
    return DATA
