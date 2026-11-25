"""Result tables, statistical tests, figures, SHAP and counterfactual explanations."""
from .config import *
from .features import *
from .model import OCHMRNet
from .metrics import METRIC_COLS, ece_score
from .experiment import ABL_MAP

def generate_report(DATA, RUNS):
    """Build every table (T01-T14) and figure (Fig02-Fig15), SHAP, counterfactuals and the final summary."""
    audit_df = pd.DataFrame([D["audit"] for D in DATA.values()]).T.reset_index()
    audit_df.columns = ["Item"] + [D["name"] for D in DATA.values()]

    def mean_std_table(df, models=None, cols=METRIC_COLS, digits=4):
        g = df.groupby("Model")[cols]; m, s = g.mean(), g.std()
        out = pd.DataFrame({c: [f"{m.loc[i, c]:.{digits}f} ± {s.loc[i, c]:.{digits}f}" for i in m.index] for c in cols}, index=m.index)
        out["_sort"] = m["MacroF1"]
        if models is not None: out = out.loc[[x for x in models if x in out.index]]
        return out.sort_values("_sort", ascending=False).drop(columns="_sort").reset_index()

    MAIN_TABLES, MEANS = {}, {}
    for (key, prot), R in RUNS.items():
        if prot != "B": continue
        df = R["res"][R["res"].get("Ablation", pd.Series(False, index=R["res"].index)).fillna(False) == False]
        tbl = mean_std_table(df); MAIN_TABLES[key] = tbl
        MEANS[key] = df.groupby("Model")[METRIC_COLS].mean()
        save_table(tbl, f"T05_main_results_{key}")
        print(f"\n==== MAIN RESULTS ({key}, Protocol B) ====\n", tbl[["Model", "Accuracy", "MacroF1", "MCC", "QWK", "AUC", "HighRecall", "ECE"]].to_string(index=False))

    # ---- Leakage table (maternal: Protocol A vs B)
    A = RUNS[("maternal", "A")]["res"]; B = RUNS[("maternal", "B")]["res"]
    B = B[B.get("Ablation", pd.Series(False, index=B.index)).fillna(False) == False]
    leak = pd.DataFrame({
        "Acc_A (naive)": A.groupby("Model").Accuracy.mean(), "Acc_B (dedup+group)": B.groupby("Model").Accuracy.mean(),
        "F1_A (naive)": A.groupby("Model").MacroF1.mean(), "F1_B (dedup+group)": B.groupby("Model").MacroF1.mean()})
    leak["ΔAcc (inflation, pp)"] = (leak["Acc_A (naive)"] - leak["Acc_B (dedup+group)"]) * 100
    leak["ΔF1 (inflation, pp)"] = (leak["F1_A (naive)"] - leak["F1_B (dedup+group)"]) * 100
    leak = leak.sort_values("F1_A (naive)", ascending=False).round(4).reset_index()
    save_table(leak, "T04_leakage_protocolA_vs_B")
    print("\nMean train-test exact overlap — Protocol A: {:.2%} | Protocol B: {:.2%}".format(A.TrainTestOverlap.mean(), B.TrainTestOverlap.mean()))
    print(leak.to_string(index=False))

    # ---- Ablation
    ABL_ORDER = list(ABL_MAP.values()) + ["w/o Knowledge features", "w/o Ordinal loss (CE)", "w/o Cost-sensitive weighting"]
    ABL_TABLES = {}
    for (key, prot), R in RUNS.items():
        if prot != "B": continue
        ab = R["res"][R["res"].Model.isin(ABL_ORDER) & (R["res"].Repeat < CFG["ABL_REPEATS"][key])]   # same folds for every variant
        t = ab.groupby("Model")[["Accuracy", "MacroF1", "MCC", "QWK", "HighRecall", "UnderTriage", "ECE", "Brier"]].agg(["mean", "std"])
        t.columns = [f"{a}_{b}" for a, b in t.columns]; t = t.loc[[m for m in ABL_ORDER if m in t.index]]
        full = t.loc["OCH-MRNet (Full)"]
        t["ΔMacroF1 vs Full (pp)"] = (t["MacroF1_mean"] - full["MacroF1_mean"]) * 100
        t["ΔHighRecall vs Full (pp)"] = (t["HighRecall_mean"] - full["HighRecall_mean"]) * 100
        ABL_TABLES[key] = t.round(4).reset_index()
        save_table(ABL_TABLES[key], f"T06_ablation_{key}")
        print(f"\n==== ABLATION ({key}) ====\n", ABL_TABLES[key][["Model", "MacroF1_mean", "HighRecall_mean", "ECE_mean", "ΔMacroF1 vs Full (pp)"]].to_string(index=False))

    # ---- Calibration + conformal
    for (key, prot), R in RUNS.items():
        if prot != "B": continue
        cal = MEANS[key][["ECE", "Brier", "LogLoss"]].copy()
        cal.loc["OCH-MRNet (avg fusion, no meta-learner)"] = R["res"][R["res"].Model == "w/o Stacking meta-learner (avg)"][["ECE", "Brier", "LogLoss"]].mean()
        save_table(cal.sort_values("ECE").round(4).reset_index(), f"T07a_calibration_{key}")
        cf = R["conf"].groupby(["Method", "Alpha"])[["Coverage", "HighClassCoverage", "AvgSetSize", "Singleton", "Empty", "Abstain", "SingletonAcc"]].mean().reset_index()
        cf.insert(2, "Target coverage", 1 - cf.Alpha)
        save_table(cf.round(4), f"T07b_conformal_{key}")
        print(f"\n==== CONFORMAL ({key}) ====\n", cf.round(3).to_string(index=False))
        if len(R["sets"]):
            tri = pd.crosstab(R["sets"].PredSet, R["sets"].TrueClass)
            save_table(tri.reset_index(), f"T07c_triage_sets_{key}")

    # ---- Complexity
    comp_rows = []
    for (key, prot), R in RUNS.items():
        if prot != "B": continue
        df = R["res"][R["res"].get("Ablation", pd.Series(False, index=R["res"].index)).fillna(False) == False]
        c = df.groupby("Model")[["FitTime_s", "InferMs_per_sample"]].mean().round(4)
        c.insert(0, "Dataset", DATA[key]["name"])
        c["Params / trees"] = ""
        c.loc[PROP, "Params / trees"] = f"FT {int(np.mean(R['complexity']['ft_params'])):,} params + CatBoost ~{int(np.mean(R['complexity']['cat_trees']))} trees"
        comp_rows.append(c.reset_index())
    save_table(pd.concat(comp_rows), "T10_computational_cost")
    Q_NEMENYI = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219,
                 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544}

    def holm(p):
        p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0
        for i, idx in enumerate(o):
            run = max(run, (len(p) - i) * p[idx]); adj[idx] = min(1, run)
        return adj

    STATS = {}
    for (key, prot), R in RUNS.items():
        if prot != "B": continue
        df = R["res"][R["res"].get("Ablation", pd.Series(False, index=R["res"].index)).fillna(False) == False]
        M = df.pivot_table(index=["Repeat", "Fold"], columns="Model", values="MacroF1")
        fr = stats.friedmanchisquare(*[M[c].values for c in M.columns])
        ranks = M.rank(axis=1, ascending=False).mean().sort_values()
        k, N = M.shape[1], M.shape[0]; CD = Q_NEMENYI.get(k, 3.6) * math.sqrt(k * (k + 1) / (6 * N))
        n_te = len(R["y"]) / CFG["N_SPLITS"]; n_tr = len(R["y"]) - n_te
        rows = []
        for c in M.columns:
            if c == PROP: continue
            d = M[PROP].values - M[c].values
            try: pw = stats.wilcoxon(M[PROP].values, M[c].values, zero_method="zsplit").pvalue
            except Exception: pw = 1.0
            var = d.var(ddof=1); J = len(d)
            tcorr = d.mean() / math.sqrt((1 / J + n_te / n_tr) * var) if var > 0 else 0.0
            pnb = 2 * stats.t.sf(abs(tcorr), J - 1)
            rows.append(dict(Comparison=f"{PROP} vs {c}", MeanΔMacroF1=d.mean(), Wilcoxon_p=pw, NB_corrected_t=tcorr, NB_p=pnb,
                             RankDiff=ranks[c] - ranks[PROP], Nemenyi_significant=abs(ranks[c] - ranks[PROP]) > CD))
        st = pd.DataFrame(rows); st["Wilcoxon_p_Holm"] = holm(st.Wilcoxon_p); st["NB_p_Holm"] = holm(st.NB_p)
        st = st.sort_values("MeanΔMacroF1").round(5)
        save_table(st, f"T08_statistical_tests_{key}")
        STATS[key] = dict(friedman=fr, ranks=ranks, CD=CD, N=N, k=k)
        print(f"\n[{key}] Friedman χ²={fr.statistic:.3f}, p={fr.pvalue:.3e} | k={k}, N={N}, Nemenyi CD={CD:.3f}")
        print(st[["Comparison", "MeanΔMacroF1", "Wilcoxon_p_Holm", "NB_p_Holm", "Nemenyi_significant"]].to_string(index=False))
        pd.DataFrame({"Model": ranks.index, "AvgRank": ranks.values}).to_csv(os.path.join(CFG["TAB"], f"T08b_avg_ranks_{key}.csv"), index=False)
    # Fig 2 — class distribution raw vs dedup
    fig, axes = plt.subplots(1, len(DATA), figsize=(6 * len(DATA), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (key, D) in zip(axes, DATA.items()):
        dd = pd.DataFrame({"Class": D["classes"] * 2,
                           "Count": [D["audit"][f"Raw_{c}"] for c in D["classes"]] + [D["audit"][f"Dedup_{c}"] for c in D["classes"]],
                           "Version": ["Raw"] * D["K"] + ["Deduplicated"] * D["K"]})
        sns.barplot(data=dd, x="Class", y="Count", hue="Version", ax=ax, palette=["#95A5A6", "#2E86C1"])
        for p in ax.patches:
            if p.get_height() > 0: ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{D['name']}\n{D['audit']['Exact_duplicates']} duplicates ({D['audit']['Duplicate_pct']}%)")
    savefig(fig, "Fig02_class_distribution_raw_vs_dedup")

    # Fig 3 — maternal EDA
    Dm = DATA["maternal"]; E = Dm["Xd"].copy(); E["Risk"] = pd.Categorical([Dm["classes"][i] for i in Dm["yd"]], Dm["classes"])
    fig, ax = plt.subplots(figsize=(7, 5.8))
    sns.heatmap(Dm["Xd"].assign(RiskLevel=Dm["yd"]).corr(method="spearman"), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, square=True)
    ax.set_title("Spearman correlation (deduplicated maternal data)"); savefig(fig, "Fig03a_maternal_correlation")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, c in zip(axes.ravel(), Dm["Xd"].columns):
        sns.violinplot(data=E, x="Risk", y=c, ax=ax, palette=["#27AE60", "#F39C12", "#C0392B"], cut=0, inner="quartile"); ax.set_xlabel("")
    savefig(fig, "Fig03b_maternal_feature_by_risk")
    if "fetal" in DATA:
        Df = DATA["fetal"]; cc = Df["Xd"].assign(fetal_health=Df["yd"]).corr(method="spearman")["fetal_health"].drop("fetal_health").sort_values()
        fig, ax = plt.subplots(figsize=(7, 7)); cc.plot.barh(ax=ax, color=np.where(cc > 0, "#C0392B", "#2E86C1"))
        ax.set_title("Fetal CTG: Spearman correlation with fetal state"); savefig(fig, "Fig03c_fetal_feature_correlation")

    # Fig 4 — leakage
    lk = leak.set_index("Model")
    order = lk.sort_values("F1_B (dedup+group)", ascending=False).index
    fig, ax = plt.subplots(figsize=(13, 5))
    xx = np.arange(len(order)); w = 0.38
    ax.bar(xx - w / 2, lk.loc[order, "F1_A (naive)"], w, label=f"Protocol A: naive random split (overlap {A.TrainTestOverlap.mean():.0%})", color="#E59866")
    ax.bar(xx + w / 2, lk.loc[order, "F1_B (dedup+group)"], w, label="Protocol B: dedup + group-stratified CV (overlap 0%)", color="#2874A6")
    for i, m in enumerate(order): ax.text(i, lk.loc[m, "F1_A (naive)"] + 0.01, f"+{lk.loc[m, 'ΔF1 (inflation, pp)']:.1f}", ha="center", fontsize=8, rotation=90)
    ax.set_xticks(xx); ax.set_xticklabels(order, rotation=40, ha="right"); ax.set_ylabel("Macro-F1"); ax.set_ylim(0.4, 1.08)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False)
    ax.set_title("Performance inflation caused by duplicate leakage (Maternal Health Risk)")
    savefig(fig, "Fig04_leakage_inflation")

    # Fig 5 — main comparison
    for key in MEANS:
        R = RUNS[(key, "B")]; df = R["res"][R["res"].get("Ablation", pd.Series(False, index=R["res"].index)).fillna(False) == False]
        g = df.groupby("Model")[["MacroF1", "HighRecall"]].agg(["mean", "std"]).sort_values(("MacroF1", "mean"))
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)
        for ax, met in zip(axes, ["MacroF1", "HighRecall"]):
            cols = [PROP_COLOR if i == PROP else BASE_COLOR for i in g.index]
            ax.barh(g.index, g[(met, "mean")], xerr=g[(met, "std")], color=cols, capsize=3)
            lo = max(0, g[(met, "mean")].min() - 0.1); ax.set_xlim(lo, 1.0)
            ax.set_xlabel(f"{met} (mean ± SD over {len(df[['Repeat','Fold']].drop_duplicates())} folds)")
            ax.set_title({"MacroF1": "Macro-F1", "HighRecall": f"Recall of '{DATA[key]['classes'][-1]}' class"}[met])
        fig.suptitle(f"{DATA[key]['name']} — leakage-free comparison", fontweight="bold")
        savefig(fig, f"Fig05_model_comparison_{key}")

    # helpers for OOF (repeat 0, each sample predicted once)
    def best_baseline(key):
        m = MEANS[key].drop(index=[PROP], errors="ignore"); return m.MacroF1.idxmax()

    # Fig 6 — confusion matrices
    for key in MEANS:
        R, D = RUNS[(key, "B")], DATA[key]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, name in zip(axes, [PROP, best_baseline(key)]):
            P = R["oof"][name]; cm = confusion_matrix(R["y"], P.argmax(1)); cmn = cm / cm.sum(1, keepdims=True)
            lab = np.array([[f"{cm[i, j]}\n({cmn[i, j]:.1%})" for j in range(D["K"])] for i in range(D["K"])])
            sns.heatmap(cmn, annot=lab, fmt="", cmap="Blues", ax=ax, xticklabels=D["classes"], yticklabels=D["classes"], cbar=False)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(name)
        savefig(fig, f"Fig06_confusion_{key}")

    # Fig 7 — ROC + PR (proposed)
    for key in MEANS:
        R, D = RUNS[(key, "B")], DATA[key]; P = R["oof"][PROP]; Y = label_binarize(R["y"], classes=list(range(D["K"])))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6)); colors = ["#27AE60", "#F39C12", "#C0392B"]
        for k in range(D["K"]):
            fpr, tpr, _ = roc_curve(Y[:, k], P[:, k]); axes[0].plot(fpr, tpr, color=colors[k], lw=2, label=f"{D['classes'][k]} (AUC={auc(fpr, tpr):.3f})")
            pr, rc, _ = precision_recall_curve(Y[:, k], P[:, k]); axes[1].plot(rc, pr, color=colors[k], lw=2, label=f"{D['classes'][k]} (AP={auc(rc, pr):.3f})")
        axes[0].plot([0, 1], [0, 1], "k--", lw=1); axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate"); axes[0].set_title("ROC (one-vs-rest)")
        axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("Precision–Recall")
        for a in axes: a.legend(loc="lower right" if a is axes[0] else "lower left")
        fig.suptitle(f"{PROP} — {D['name']}", fontweight="bold"); savefig(fig, f"Fig07_roc_pr_{key}")

    # Fig 8 — reliability diagrams
    def reliability(ax, y, P, label, color, n_bins=10):
        conf, acc = P.max(1), (P.argmax(1) == y)
        bins = np.linspace(0, 1, n_bins + 1); xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() > 0: xs.append(conf[m].mean()); ys.append(acc[m].mean())
        ax.plot(xs, ys, "o-", color=color, label=f"{label} (ECE={ece_score(y, P):.3f})")
    for key in MEANS:
        R = RUNS[(key, "B")]
        fig, ax = plt.subplots(figsize=(5.8, 5.2)); ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        reliability(ax, R["y"], R["oof"]["OCH-MRNet (avg fusion)"], "Branch average (no meta)", "#F39C12")
        reliability(ax, R["y"], R["oof"][PROP], PROP, PROP_COLOR)
        reliability(ax, R["y"], R["oof"][best_baseline(key)], best_baseline(key), BASE_COLOR)
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax.legend(fontsize=8); ax.set_title(f"Reliability — {DATA[key]['name']}")
        savefig(fig, f"Fig08_reliability_{key}")

    # Fig 9 — conformal
    for key in MEANS:
        cf = RUNS[(key, "B")]["conf"].groupby(["Method", "Alpha"]).mean(numeric_only=True).reset_index()
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
        for meth, mk in zip(cf.Method.unique(), ["o", "s"]):
            s = cf[cf.Method == meth]
            axes[0].plot(1 - s.Alpha, s.Coverage, marker=mk, label=meth); axes[1].plot(1 - s.Alpha, s.HighClassCoverage, marker=mk, label=meth)
            axes[2].plot(1 - s.Alpha, s.AvgSetSize, marker=mk, label=f"{meth} — set size"); axes[2].plot(1 - s.Alpha, s.Abstain, marker=mk, ls=":", label=f"{meth} — abstain rate")
        for a in axes[:2]: a.plot([0.78, 0.97], [0.78, 0.97], "k--", lw=1, label="Target")
        axes[0].set_title("Marginal coverage"); axes[1].set_title(f"Coverage on '{DATA[key]['classes'][-1]}' patients"); axes[2].set_title("Set size & clinician-referral rate")
        for a in axes: a.set_xlabel("Target coverage (1 − α)"); a.legend(fontsize=7)
        savefig(fig, f"Fig09_conformal_{key}")

    # Fig 10 — CD-style rank plot
    for key, S in STATS.items():
        rk = S["ranks"]; fig, ax = plt.subplots(figsize=(8, 0.42 * len(rk) + 1.5))
        yy = np.arange(len(rk))[::-1]
        ax.axvspan(rk.iloc[0], rk.iloc[0] + S["CD"], color="#D5F5E3", label=f"Within CD of best (CD={S['CD']:.2f})")
        ax.scatter(rk.values, yy, c=[PROP_COLOR if m == PROP else BASE_COLOR for m in rk.index], s=60, zorder=3)
        ax.set_yticks(yy); ax.set_yticklabels(rk.index); ax.set_xlabel("Average rank (Macro-F1; lower = better)")
        ax.set_title(f"Friedman p={S['friedman'].pvalue:.2e} | Nemenyi critical difference — {DATA[key]['name']}"); ax.legend(loc="lower right")
        savefig(fig, f"Fig10_critical_difference_{key}")

    # Fig 12 — Decision curve analysis (highest-risk class vs rest)
    def net_benefit(yb, s, th):
        n = len(yb); tp = ((s >= th) & (yb == 1)).sum(); fp = ((s >= th) & (yb == 0)).sum()
        return tp / n - fp / n * th / (1 - th)
    for key in MEANS:
        R, D = RUNS[(key, "B")], DATA[key]; yb = (R["y"] == D["high"]).astype(int); th = np.linspace(0.01, 0.8, 80)
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        for name, col in [(PROP, PROP_COLOR), (best_baseline(key), BASE_COLOR)]:
            ax.plot(th, [net_benefit(yb, R["oof"][name][:, D["high"]], t) for t in th], color=col, lw=2, label=name)
        prev = yb.mean(); ax.plot(th, [prev - (1 - prev) * t / (1 - t) for t in th], "k:", label="Treat all"); ax.axhline(0, color="gray", lw=1, label="Treat none")
        ax.set_ylim(-0.05, prev + 0.05); ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
        ax.set_title(f"Decision curve — '{D['classes'][-1]}' detection ({D['name']})"); ax.legend(fontsize=8)
        savefig(fig, f"Fig12_decision_curve_{key}")

    # Fig 13 — robustness
    for key in MEANS:
        rb = RUNS[(key, "B")]["rob"].groupby(["Model", "Noise"])[["MacroF1", "HighRecall"]].mean().reset_index()
        save_table(rb.round(4), f"T11_robustness_{key}")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for m in rb.Model.unique():
            s = rb[rb.Model == m]; kw = dict(color=PROP_COLOR, lw=2.8) if m == PROP else dict(lw=1.4)
            axes[0].plot(s.Noise, s.MacroF1, "o-", label=m, **kw); axes[1].plot(s.Noise, s.HighRecall, "o-", label=m, **kw)
        axes[0].set_ylabel("Macro-F1"); axes[1].set_ylabel(f"Recall '{DATA[key]['classes'][-1]}'")
        for a in axes: a.set_xlabel("Gaussian sensor noise (× feature SD)"); a.legend(fontsize=7)
        fig.suptitle(f"Robustness to measurement noise — {DATA[key]['name']}", fontweight="bold"); savefig(fig, f"Fig13_robustness_{key}")

    # Fig 14 — ablation
    for key, t in ABL_TABLES.items():
        t2 = t.set_index("Model").drop(index="OCH-MRNet (Full)")
        fig, ax = plt.subplots(figsize=(9, 4.5)); yy = np.arange(len(t2))
        ax.barh(yy - 0.2, t2["ΔMacroF1 vs Full (pp)"], 0.4, label="Δ Macro-F1 (pp)", color="#2874A6")
        ax.barh(yy + 0.2, t2["ΔHighRecall vs Full (pp)"], 0.4, label=f"Δ Recall '{DATA[key]['classes'][-1]}' (pp)", color="#C0392B")
        ax.axvline(0, color="k", lw=1); ax.set_yticks(yy); ax.set_yticklabels(t2.index); ax.legend()
        ax.set_title(f"Ablation: change vs full OCH-MRNet — {DATA[key]['name']}"); savefig(fig, f"Fig14_ablation_{key}")

    # Fig 15 + Table 9 — subgroup fairness by maternal age
    R, D = RUNS[("maternal", "B")], DATA["maternal"]
    bands = pd.cut(R["X"]["Age"], [0, 19, 34, 200], labels=["<20 (teen)", "20–34", "≥35 (advanced)"])
    fr_rows = []
    for name in [PROP, best_baseline("maternal")]:
        P = R["oof"][name]
        for b in bands.cat.categories:
            m = (bands == b).values
            if m.sum() < 5: continue
            yy, pp = R["y"][m], P[m].argmax(1)
            fr_rows.append(dict(Model=name, AgeBand=b, n=int(m.sum()), n_high=int((yy == 2).sum()), Accuracy=accuracy_score(yy, pp),
                                MacroF1=f1_score(yy, pp, average="macro", labels=[0, 1, 2], zero_division=0),
                                HighRecall=recall_score(yy, pp, labels=[2], average="macro", zero_division=0),
                                UnderTriage=float(np.mean(pp < yy)), ECE=ece_score(yy, P[m])))
    fair = pd.DataFrame(fr_rows).round(4); save_table(fair, "T09_subgroup_fairness_age")
    print(fair.to_string(index=False))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.barplot(data=fair, x="AgeBand", y="MacroF1", hue="Model", palette=[PROP_COLOR, BASE_COLOR], ax=ax)
    ax.set_ylim(0, 1); ax.set_title("Subgroup performance by maternal age (Macro-F1)"); savefig(fig, "Fig15_subgroup_age")
    import shap
    FINAL = {}
    for key, D in DATA.items():
        set_seed(CFG["SEED"]); FINAL[key] = OCHMRNet(key, D["K"], seed=CFG["SEED"]).fit(D["Xd"], D["yd"], D["gd"])
        print(f"[final] {key}: FT epochs={FINAL[key].ft_epochs_}, CatBoost iters={FINAL[key].cat_iters_}")

    SHAP_TABLES = {}
    for key, D in DATA.items():
        mdl, Xd = FINAL[key], D["Xd"]; cols = list(Xd.columns)
        f = lambda A: mdl.predict_proba(pd.DataFrame(A, columns=cols))
        rng = np.random.default_rng(CFG["SEED"])
        n_exp = 150 if key == "maternal" else 100
        # stratified explanation sample
        idx = np.concatenate([rng.choice(np.where(D["yd"] == k)[0], min(n_exp // D["K"], (D["yd"] == k).sum()), replace=False) for k in range(D["K"])])
        Xe = Xd.iloc[idx].reset_index(drop=True)
        expl = shap.KernelExplainer(f, shap.kmeans(Xd.values, 15))
        sv = expl.shap_values(Xe.values, nsamples=300 if key == "maternal" else 400, silent=True)
        sv = np.stack(sv, -1) if isinstance(sv, list) else np.asarray(sv)          # -> (n, features, classes)
        ev = np.atleast_1d(expl.expected_value); hi = D["high"]
        # Fig 11a — beeswarm (highest-risk class)
        plt.figure(); shap.summary_plot(sv[:, :, hi], Xe, show=False, max_display=15)
        plt.title(f"SHAP — '{D['classes'][hi]}' class ({D['name']})"); savefig(plt.gcf(), f"Fig11a_shap_beeswarm_{key}")
        # Fig 11b — mean |SHAP| per class
        imp = pd.DataFrame(np.abs(sv).mean(0), index=cols, columns=D["classes"]); imp["Total"] = imp.sum(1)
        imp = imp.sort_values("Total", ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(7.5, 0.4 * len(imp) + 1.5))
        imp[D["classes"]].plot.barh(stacked=True, ax=ax, color=["#27AE60", "#F39C12", "#C0392B"])
        ax.set_xlabel("mean |SHAP value|"); ax.set_title(f"Global feature importance — {D['name']}"); savefig(fig, f"Fig11b_shap_importance_{key}")
        SHAP_TABLES[key] = imp.sort_values("Total", ascending=False).round(5).reset_index().rename(columns={"index": "Feature"})
        save_table(SHAP_TABLES[key], f"T12_shap_importance_{key}")
        # Fig 11c — dependence for top-2 features
        top2 = SHAP_TABLES[key].Feature.iloc[:2].tolist()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, ft in zip(axes, top2):
            j = cols.index(ft); sc = ax.scatter(Xe[ft], sv[:, j, hi], c=sv[:, :, hi].sum(1), cmap="coolwarm", s=22)
            ax.axhline(0, color="gray", lw=0.8); ax.set_xlabel(ft); ax.set_ylabel(f"SHAP for '{D['classes'][hi]}'")
        plt.colorbar(sc, ax=axes, label="total SHAP (high class)"); fig.suptitle(f"SHAP dependence — {D['name']}", fontweight="bold")
        fig.savefig(os.path.join(CFG["FIG"], f"Fig11c_shap_dependence_{key}.png"), dpi=300, bbox_inches="tight"); plt.close(fig)
        # Fig 11d — waterfall for one correctly-identified high-risk patient
        Pe = f(Xe.values); cand = np.where((D["yd"][idx] == hi) & (Pe.argmax(1) == hi))[0]
        if len(cand):
            i = cand[0]
            e = shap.Explanation(values=sv[i, :, hi], base_values=ev[hi] if len(ev) > 1 else ev[0], data=Xe.iloc[i].values, feature_names=cols)
            plt.figure(); shap.plots.waterfall(e, show=False, max_display=12)
            plt.title(f"Local explanation — patient predicted '{D['classes'][hi]}' (p={Pe[i, hi]:.2f})"); savefig(plt.gcf(), f"Fig11d_shap_waterfall_{key}")

    # ---- Counterfactual explanations (maternal; Age is immutable) — custom search, no extra package needed
    D, mdl = DATA["maternal"], FINAL["maternal"]; Xd = D["Xd"]
    MUTABLE = {"SystolicBP": (90, 160), "DiastolicBP": (60, 100), "BS": (6.0, 19.0), "BodyTemp": (98.0, 103.0), "HeartRate": (60, 90)}
    sd = Xd.std(); rng = np.random.default_rng(CFG["SEED"])
    Pall = mdl.predict_proba(Xd); hi_idx = np.where((D["yd"] == 2) & (Pall.argmax(1) == 2))[0]
    cf_rows = []
    for pid in rng.choice(hi_idx, min(6, len(hi_idx)), replace=False):
        x0 = Xd.iloc[pid]; cands = []
        for _ in range(4000):
            x = x0.copy(); feats = rng.choice(list(MUTABLE), rng.integers(1, 4), replace=False)
            for ft in feats:
                lo, hi_ = MUTABLE[ft]; x[ft] = rng.uniform(lo, hi_)
                x[ft] = round(x[ft], 1) if ft in ("BS", "BodyTemp") else round(x[ft])
            if x["DiastolicBP"] >= x["SystolicBP"]: continue
            cands.append(x)
        C = pd.DataFrame(cands); Pc = mdl.predict_proba(C)
        ok = Pc.argmax(1) < 2
        if not ok.any(): continue
        C, Pc = C[ok].reset_index(drop=True), Pc[ok]
        keep = ~C.duplicated(subset=list(MUTABLE)).values                 # unique counterfactuals only
        C, Pc = C[keep].reset_index(drop=True), Pc[keep]
        dist = (np.abs(C[list(MUTABLE)] - x0[list(MUTABLE)]) / sd[list(MUTABLE)]).sum(1).values
        for rank, j in enumerate(np.argsort(dist)[:3]):
            ch = {ft: f"{x0[ft]:g} → {C.loc[j, ft]:g}" for ft in MUTABLE if C.loc[j, ft] != x0[ft]}
            cf_rows.append(dict(PatientID=int(pid), Age=x0["Age"], Original_P_high=round(Pall[pid, 2], 3), CF_rank=rank + 1,
                                Changes="; ".join(f"{k}: {v}" for k, v in ch.items()), New_class=D["classes"][Pc[j].argmax()],
                                New_P_high=round(Pc[j, 2], 3), Norm_L1_distance=round(dist[j], 3)))
    cf_df = pd.DataFrame(cf_rows); save_table(cf_df, "T13_counterfactuals_maternal"); print(cf_df.to_string(index=False))
    LIT = pd.DataFrame([
        ("Monson & Sabarmathi (stacking RF+XGB+GB)", "Maternal (1014 raw)", "Random split, duplicates kept", 0.9113, np.nan, np.nan),
        ("Yesmin et al., 2026 (fuzzy-XGBoost)", "Maternal (1014 raw)", "As reported", 0.8867, np.nan, 0.9703),
        ("Solent study (Random Forest + SMOTE)", "Maternal (1014 raw)", "80/20 split, SMOTE after split", 0.882, 0.888, 0.966),
    ], columns=["Study", "Data", "Protocol", "Accuracy", "MacroF1", "AUC"])
    mB = MEANS["maternal"].loc[PROP]; mA = A[A.Model == PROP][["Accuracy", "MacroF1", "AUC"]].mean()
    LIT.loc[len(LIT)] = [f"{PROP} — Protocol A (same naive protocol)", "Maternal (1014 raw)", "Naive stratified CV", mA.Accuracy, mA.MacroF1, mA.AUC]
    LIT.loc[len(LIT)] = [f"{PROP} — Protocol B (leakage-free)", "Maternal (452 unique)", "Dedup + group-stratified 5×5 CV", mB.Accuracy, mB.MacroF1, mB.AUC]
    save_table(LIT.round(4), "T14_literature_comparison")


    print("=" * 100); print("FINAL SUMMARY"); print("=" * 100)
    print(audit_df.to_string(index=False))
    print("\n--- LEAKAGE (maternal) ---"); print(leak[["Model", "Acc_A (naive)", "Acc_B (dedup+group)", "ΔAcc (inflation, pp)", "ΔF1 (inflation, pp)"]].to_string(index=False))
    for key in MEANS:
        print(f"\n--- MAIN ({key}) ---"); print(MAIN_TABLES[key][["Model", "Accuracy", "MacroF1", "MCC", "QWK", "AUC", "HighRecall", "UnderTriage", "ECE", "Brier"]].to_string(index=False))
        print(f"\n--- ABLATION ({key}) ---"); print(ABL_TABLES[key][["Model", "MacroF1_mean", "HighRecall_mean", "ECE_mean", "ΔMacroF1 vs Full (pp)", "ΔHighRecall vs Full (pp)"]].to_string(index=False))
        cf = RUNS[(key, "B")]["conf"].groupby(["Method", "Alpha"])[["Coverage", "HighClassCoverage", "AvgSetSize", "Abstain", "SingletonAcc"]].mean().round(3)
        print(f"\n--- CONFORMAL ({key}) ---"); print(cf.to_string())
        S = STATS[key]; print(f"\n--- STATS ({key}) --- Friedman p={S['friedman'].pvalue:.2e}, CD={S['CD']:.3f}"); print(S["ranks"].round(2).to_string())
        print(f"\n--- SHAP top features ({key}) ---"); print(SHAP_TABLES[key].head(8).to_string(index=False))
    print("\n--- FAIRNESS (age) ---"); print(fair.to_string(index=False))
    print("\n--- COUNTERFACTUALS ---"); print(cf_df.head(9).to_string(index=False))
    print("\n--- LITERATURE ---"); print(LIT.round(4).to_string(index=False))
    print("\nFigures:", sorted(os.listdir(CFG["FIG"]))); print("Tables:", sorted(os.listdir(CFG["TAB"])))

    # zip everything for download
    import shutil; arc = shutil.make_archive(CFG["OUT"].rstrip("/") + "_archive", "zip", CFG["OUT"])
    print("\nAll outputs zipped to:", arc)
    return dict(main=MAIN_TABLES, means=MEANS, stats=STATS, ablation=ABL_TABLES, final_models=FINAL)
