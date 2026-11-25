"""Cross-validation engine: Protocol A (naive) and Protocol B (deduplicated, group-stratified)."""
from .config import *
from .features import *
from .baselines import make_baselines
from .model import OCHMRNet
from .metrics import compute_metrics

ROBUST_MODELS = ["Random Forest", "XGBoost", "CatBoost", "Stacking (Lit.)", "FT-Transformer", PROP]
ABL_MAP = {"full": "OCH-MRNet (Full)", "avg": "w/o Stacking meta-learner (avg)", "tree_only": "w/o FT branch (CatBoost+GB)",
           "no_gb": "w/o GB branch (FT+CatBoost)", "ft_only": "w/o Tree branches (FT only)"}

def overlap_rate(Xtr, ytr, Xte, yte):
    a = pd.util.hash_pandas_object(Xtr.assign(__y=ytr), index=False)
    b = pd.util.hash_pandas_object(Xte.assign(__y=yte), index=False)
    return float(b.isin(set(a)).mean())

def add_noise(X, std, level, rng):
    if level == 0: return X
    Xn = X.copy()
    for c in X.columns:
        if X[c].nunique() > 2: Xn[c] = X[c] + rng.normal(0, level * std[c], len(X))
    return Xn

def run_protocol(DATA, key, protocol):
    """Run all baselines + OCH-MRNet (and ablations, conformal, robustness) under Protocol A or B."""
    D = DATA[key]; K, high = D["K"], D["high"]
    if protocol == "B": X, y, groups, n_rep = D["Xd"], D["yd"], D["gd"], CFG["N_REPEATS"][key]
    else:               X, y, n_rep = D["X"], D["y"], CFG["N_REPEATS_A"]; groups = np.arange(len(y))
    baselines = make_baselines(key)
    res, conf_rows, rob_rows, oof, set_log = [], [], [], {}, []
    complexity = {}
    total = n_rep * CFG["N_SPLITS"]; t_start = time.time()
    for r in range(n_rep):
        skf = StratifiedGroupKFold(n_splits=CFG["N_SPLITS"], shuffle=True, random_state=CFG["SEED"] + r)
        for f, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y, groups)):
            Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
            ov = overlap_rate(Xtr, ytr, Xte, yte)
            std = Xtr.std(); rng_seed = CFG["SEED"] + 1000 * r + f
            meta = dict(Dataset=D["name"], Protocol=protocol, Repeat=r, Fold=f, TrainTestOverlap=ov)

            def record(name, P, fit_t, inf_t, extra=None):
                row = {**meta, "Model": name, **compute_metrics(yte, P, K, high), "FitTime_s": fit_t,
                       "InferMs_per_sample": 1000 * inf_t / len(yte)}
                if extra: row.update(extra)
                res.append(row)
                if r == 0:
                    oof.setdefault(name, np.full((len(y), K), np.nan))[te] = P

            # ---- baselines
            for name, factory in baselines.items():
                set_seed(CFG["SEED"]); m = factory(); t0 = time.time(); m.fit(Xtr, ytr); ft_ = time.time() - t0
                t0 = time.time(); P = m.predict_proba(Xte); it_ = time.time() - t0
                record(name, P, ft_, it_)
                if r == 0 and protocol == "B" and name in ROBUST_MODELS:
                    for lv in CFG["NOISE_LEVELS"]:
                        Pn = m.predict_proba(add_noise(Xte, std, lv, np.random.default_rng(rng_seed)))
                        rob_rows.append(dict(Dataset=D["name"], Model=name, Fold=f, Noise=lv,
                                             MacroF1=f1_score(yte, Pn.argmax(1), average="macro"),
                                             HighRecall=recall_score(yte, Pn.argmax(1), labels=[high], average="macro", zero_division=0)))
            # ---- proposed
            set_seed(CFG["SEED"]); m = OCHMRNet(key, K, seed=CFG["SEED"] + f).fit(Xtr, ytr, groups[tr])
            t0 = time.time(); V = m.variants(Xte); it_ = time.time() - t0
            record(PROP, V["full"], m.fit_time_, it_)
            if r == 0:
                oof.setdefault("OCH-MRNet (avg fusion)", np.full((len(y), K), np.nan))[te] = V["avg"]
            complexity.setdefault("ft_params", []).append(m.n_params()[0]); complexity.setdefault("cat_trees", []).append(m.n_params()[1])
            if protocol == "B":
                for vk, vname in ABL_MAP.items():
                    res.append({**meta, "Model": vname, "Ablation": True, **compute_metrics(yte, V[vk], K, high)})
                if CFG["RUN_ABLATION"] and r < CFG["ABL_REPEATS"][key]:
                    for vname, kw in [("w/o Knowledge features", dict(use_kf=False)),
                                      ("w/o Ordinal loss (CE)", dict(ordinal=False)),
                                      ("w/o Cost-sensitive weighting", dict(cost=False))]:
                        set_seed(CFG["SEED"]); ma = OCHMRNet(key, K, seed=CFG["SEED"] + f, **kw).fit(Xtr, ytr, groups[tr])
                        res.append({**meta, "Model": vname, "Ablation": True, **compute_metrics(yte, ma.predict_proba(Xte), K, high)})
                # conformal prediction
                for alpha in CFG["CONF_ALPHAS"]:
                    for mond in [False, True]:
                        S = m.predict_sets(V["full"], alpha, mond); sz = S.sum(1)
                        cov = S[np.arange(len(yte)), yte]
                        conf_rows.append(dict(Dataset=D["name"], Repeat=r, Fold=f, Alpha=alpha,
                                              Method="Class-conditional (Mondrian)" if mond else "Marginal (LAC)",
                                              Coverage=cov.mean(), HighClassCoverage=cov[yte == high].mean() if (yte == high).any() else np.nan,
                                              AvgSetSize=sz.mean(), Singleton=(sz == 1).mean(), Empty=(sz == 0).mean(),
                                              Abstain=(sz != 1).mean(),
                                              SingletonAcc=(cov[sz == 1].mean() if (sz == 1).any() else np.nan)))
                        if r == 0 and alpha == CFG["MAIN_ALPHA"] and mond:
                            for i in range(len(yte)):
                                set_log.append(dict(Dataset=D["name"], TrueClass=D["classes"][yte[i]],
                                                    PredSet="{" + ",".join(D["classes"][k] for k in np.where(S[i])[0]) + "}"))
                # robustness for proposed
                if r == 0:
                    for lv in CFG["NOISE_LEVELS"]:
                        Pn = m.predict_proba(add_noise(Xte, std, lv, np.random.default_rng(rng_seed)))
                        rob_rows.append(dict(Dataset=D["name"], Model=PROP, Fold=f, Noise=lv,
                                             MacroF1=f1_score(yte, Pn.argmax(1), average="macro"),
                                             HighRecall=recall_score(yte, Pn.argmax(1), labels=[high], average="macro", zero_division=0)))
            done = r * CFG["N_SPLITS"] + f + 1; el = time.time() - t_start
            last = pd.DataFrame(res)
            last = last[(last.Repeat == r) & (last.Fold == f) & (last.Model == PROP)].iloc[0]
            print(f"[{key}|{protocol}] fold {done}/{total} | overlap={ov:.2%} | {PROP} F1={last.MacroF1:.4f} "
                  f"HighRec={last.HighRecall:.4f} | elapsed {el/60:.1f} min, ETA {el/done*(total-done)/60:.1f} min")
    out = dict(res=pd.DataFrame(res), conf=pd.DataFrame(conf_rows), rob=pd.DataFrame(rob_rows), oof=oof,
               y=y, X=X, sets=pd.DataFrame(set_log), complexity=complexity)
    out["res"].to_csv(os.path.join(CFG["TAB"], f"raw_results_{key}_{protocol}.csv"), index=False)
    return out


