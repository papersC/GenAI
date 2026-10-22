"""Aggregate raw per-run JSON into CSV tables (mean and 95% t-CI over seeds) and figures."""
import json, glob, os, sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "results", "raw")
AGG = os.path.join(HERE, "results", "aggregated")
FIG = os.path.join(HERE, "results", "figures")
os.makedirs(AGG, exist_ok=True); os.makedirs(FIG, exist_ok=True)
METRICS = ["isl_mbps", "lat_base_mean", "lat_base_p95", "lat_base_p99", "lat_full_mean", "lat_full_p95",
           "base_deadline_ratio", "delivery_ratio", "ontime_delivery", "enh_ontime", "quality", "quality_icu",
           "mean_tier_icu", "jitter_ms", "gpu_mean", "solve_mean_s", "solve_max_s", "n_nonoptimal", "struct_infeasible"]
ORDER = ["UC-H265", "MC-SVC-S", "MC-SVC-R", "MC-SVC-PM", "SEM-S", "SEM-R", "SEM-PG", "SEM-PM", "SEM-OM"]
LABEL = {"UC-H265": "UC-H265", "MC-SVC-S": "MC-SVC-S", "MC-SVC-R": "MC-SVC-R", "MC-SVC-PM": "MC-SVC-PM",
         "SEM-S": "SEM-S", "SEM-R": "SEM-R", "SEM-PG": "SEM-PG", "SEM-PM": "SEM-PM (proposed)", "SEM-OM": "SEM-OM (oracle)"}
STY = {"UC-H265": ("tab:gray", "x", ":"), "MC-SVC-S": ("tab:brown", "s", "--"), "MC-SVC-R": ("tab:orange", "v", "--"),
       "MC-SVC-PM": ("tab:red", "^", "--"), "SEM-S": ("tab:olive", "D", "-."), "SEM-R": ("tab:cyan", "<", "-."),
       "SEM-PG": ("tab:purple", ">", "-"), "SEM-PM": ("tab:blue", "o", "-"), "SEM-OM": ("black", "*", ":")}


def load():
    runs = {}
    for fn in glob.glob(os.path.join(RAW, "*.json")):
        o = json.load(open(fn))
        runs.setdefault(o["config"], []).append(o)
    return runs


def ci(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return np.nan, np.nan, 0
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return m, h, len(x)


def summarize(runs):
    rows = []
    for cfg, rs in sorted(runs.items()):
        for sch in ORDER:
            vals = [r["schemes"][sch] for r in rs if sch in r["schemes"]]
            if not vals:
                continue
            row = {"config": cfg, "scheme": sch, "n_seeds": len(vals)}
            for m in METRICS:
                mu, h, n = ci([v[m] for v in vals])
                row[m] = mu; row[m + "_ci95"] = h
            rows.append(row)
    import csv
    with open(os.path.join(AGG, "summary_all.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    # paired differences vs SEM-PM for default config (paired over seeds)
    return {(r["config"], r["scheme"]): r for r in rows}


def paired(runs, cfg, a, b, m):
    rs = runs[cfg]
    d = [r["schemes"][a][m] - r["schemes"][b][m] for r in rs if a in r["schemes"] and b in r["schemes"]]
    rel = [100 * (r["schemes"][a][m] - r["schemes"][b][m]) / r["schemes"][b][m] for r in rs if a in r["schemes"] and b in r["schemes"]]
    t = stats.ttest_1samp(d, 0.0) if len(d) > 1 else None
    return ci(d), ci(rel), (t.pvalue if t is not None else np.nan)


plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True, "grid.alpha": 0.3,
                     "legend.fontsize": 6.5, "lines.linewidth": 1.1, "lines.markersize": 3.5, "pdf.fonttype": 42})


def sweep_fig(S, xs, cfgs, xlabel, metrics, fname, schemes=ORDER, logx=False):
    fig, axs = plt.subplots(1, len(metrics), figsize=(7.16, 1.9))
    for ax, (m, yl) in zip(axs, metrics):
        for sch in schemes:
            pts = [(x, S.get((c, sch))) for x, c in zip(xs, cfgs)]
            pts = [(x, r) for x, r in pts if r is not None]
            if not pts:
                continue
            c, mk, ls = STY[sch]
            ax.errorbar([x for x, _ in pts], [r[m] for _, r in pts], yerr=[r[m + "_ci95"] for _, r in pts],
                        color=c, marker=mk, ls=ls, capsize=1.5, label=LABEL[sch])
        ax.set_xlabel(xlabel); ax.set_ylabel(yl)
        if logx: ax.set_xscale("log", base=2)
        if m == "isl_mbps" and fname == "fig_sweep_N": ax.set_yscale("log")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=min(9, len(l)), bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIG, fname + ".pdf"), bbox_inches="tight"); fig.savefig(os.path.join(FIG, fname + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def bars_default(S):
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.85))
    mets = [("isl_mbps", "ISL load (Mbps$\\cdot$link)"), ("ontime_delivery", "Base on-time delivery"),
            ("quality_icu", "Quality proxy (ICU)"), ("lat_base_p95", "p95 base latency (ms)")]
    x = np.arange(len(ORDER))
    for ax, (m, yl) in zip(axs, mets):
        mu = [S[("default", s)][m] for s in ORDER]; h = [S[("default", s)][m + "_ci95"] for s in ORDER]
        ax.bar(x, mu, yerr=h, color=[STY[s][0] for s in ORDER], capsize=1.5, width=0.75)
        ax.set_xticks(x); ax.set_xticklabels(ORDER, rotation=75, fontsize=5.5); ax.set_ylabel(yl)
        if m == "ontime_delivery":
            ax.set_ylim(min(mu) - 0.05, 1.0)
        if m == "lat_base_p95":
            ax.axhline(150, color="k", ls=":", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_default_bars.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(FIG, "fig_default_bars.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def timeseries(runs):
    r = [o for o in runs["default"] if o["seed"] == 1][0]
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 1.8))
    for sch in ["MC-SVC-S", "MC-SVC-R", "SEM-R", "SEM-PM"]:
        ts = r["schemes"][sch]["ts"]; t = np.arange(len(ts["isl"]))
        c, mk, ls = STY[sch]
        axs[0].plot(t, ts["isl"], color=c, ls=ls, label=LABEL[sch])
        axs[1].plot(t, ts["Lb_max"], color=c, ls=ls)
        axs[2].plot(t, ts["Q"], color=c, ls=ls)
    axs[1].axhline(150, color="k", ls=":", lw=0.8)
    axs[0].set_ylabel("ISL load (Mbps$\\cdot$link)"); axs[1].set_ylabel("max base latency (ms)"); axs[2].set_ylabel("mean quality proxy")
    for a in axs: a.set_xlabel("time (s)")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIG, "fig_timeseries.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(FIG, "fig_timeseries.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    runs = load()
    S = summarize(runs)
    print("configs:", {k: len(v) for k, v in runs.items()})
    for cfg in sorted(runs):
        print(f"\n== {cfg} ==")
        for sch in ORDER:
            r = S.get((cfg, sch))
            if r is None: continue
            print(f"{sch:10s} isl={r['isl_mbps']:7.1f}±{r['isl_mbps_ci95']:5.1f} Lb={r['lat_base_mean']:6.1f} p95={r['lat_base_p95']:6.1f} p99={r['lat_base_p99']:6.1f} "
                  f"dl={r['base_deadline_ratio']:.4f} del={r['delivery_ratio']:.4f} ont={r['ontime_delivery']:.4f}±{r['ontime_delivery_ci95']:.4f} "
                  f"Qicu={r['quality_icu']:5.1f}±{r['quality_icu_ci95']:4.1f} enh={r['enh_ontime']:.3f} tier={r['mean_tier_icu']:.2f} gpu={r['gpu_mean']:.1f} "
                  f"solve={r['solve_mean_s']:.3f}/{r['solve_max_s']:.2f} nonopt={r['n_nonoptimal']:.1f} inf={r['struct_infeasible']:.3f}")
    if "default" in runs:
        bars_default(S); timeseries(runs)
        with open(os.path.join(AGG, "paired_default.txt"), "w") as f:
            for a, b in [("SEM-PM", "MC-SVC-S"), ("SEM-PM", "UC-H265"), ("SEM-PM", "MC-SVC-PM"), ("SEM-PM", "SEM-S"), ("SEM-PM", "SEM-R"), ("SEM-PM", "SEM-PG"), ("SEM-PM", "SEM-OM"), ("MC-SVC-PM", "MC-SVC-R"), ("MC-SVC-S", "UC-H265")]:
                for m in ["isl_mbps", "ontime_delivery", "quality_icu", "lat_base_p95", "enh_ontime"]:
                    (d, hd, n), (rl, hr, _), pv = paired(runs, "default", a, b, m)
                    line = f"{a} - {b} [{m}]: diff={d:.4f}±{hd:.4f} rel={rl:.2f}%±{hr:.2f}% p={pv:.2e} n={n}"
                    print(line); f.write(line + "\n")
    def have(c): return all(x in runs for x in c)
    cN = ["N4", "N8", "default", "N32", "N48"]
    if have(cN):
        sweep_fig(S, [4, 8, 16, 32, 48], cN, "receivers $N$", [("isl_mbps", "ISL load (Mbps$\\cdot$link)"), ("ontime_delivery", "base on-time delivery"), ("quality_icu", "quality proxy (ICU)"), ("solve_mean_s", "decision time (s)")], "fig_sweep_N")
    cL = ["L0.5", "L0.6", "default", "L0.8", "L0.9"]
    if have(cL):
        sweep_fig(S, [0.5, 0.6, 0.7, 0.8, 0.9], cL, "background load scale $L$", [("ontime_delivery", "base on-time delivery"), ("lat_base_p95", "p95 base latency (ms)"), ("quality_icu", "quality proxy (ICU)"), ("enh_ontime", "enh. on-time ratio")], "fig_sweep_load", schemes=[s for s in ORDER if s != "UC-H265"])
    cM = ["M4", "default", "M36", "M64"]
    if have(cM):
        sweep_fig(S, [4, 16, 36, 64], cM, "SDC clusters $M$", [("isl_mbps", "ISL load (Mbps$\\cdot$link)"), ("quality_icu", "quality proxy (ICU)"), ("gpu_mean", "re-quantisations in use")], "fig_sweep_M", schemes=["SEM-S", "SEM-R", "SEM-PG", "SEM-PM", "SEM-OM"])


def fmt(r, m, d=1, scale=1.0):
    return f"{r[m]*scale:.{d}f}$\\pm${r[m+'_ci95']*scale:.{d}f}"


def tables(S, runs):
    # Table: default scenario, all schemes
    L = ["\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
         "Scheme & ISL load & Base on-time & p95 $L^{\\mathrm b}$ & Quality & Enh.\\ on-time & Jitter\\\\",
         " & (Mbps$\\cdot$link) & (\\%) & (ms) & (ICU) & (\\%) & (ms)\\\\", "\\midrule"]
    for s in ORDER:
        r = S[("default", s)]
        L.append(f"{s} & {fmt(r,'isl_mbps',0)} & {fmt(r,'ontime_delivery',1,100)} & {fmt(r,'lat_base_p95',1)} & {fmt(r,'quality_icu',1)} & {fmt(r,'enh_ontime',1,100)} & {r['jitter_ms']:.2f}\\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AGG, "table_default.tex"), "w").write("\n".join(L) + "\n")
    # Table: sensitivity (semantic schemes) for rho, decode scale, GPU budget
    rowsdef = [("$\\rho=2$", "rho2"), ("$\\rho=4$ (default)", "default"), ("$\\rho=6$", "rho6"), ("$\\rho=8$", "rho8"),
               ("decode $\\times0.5$", "dec0.5"), ("decode $\\times1.5$", "dec1.5"),
               ("$G_m=0$", "gpu0"), ("$G_m=1$", "gpu1"), ("$G_m=6$", "gpu6"),
               ("$M=4$", "M4"), ("$M=36$", "M36"), ("$M=64$", "M64")]
    L = ["\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
         "Setting & \\multicolumn{2}{c}{SEM-PM} & \\multicolumn{2}{c}{SEM-S}\\\\",
         " & ISL load & Quality & ISL load & Quality\\\\", "\\midrule"]
    for lab, c in rowsdef:
        if (c, "SEM-PM") not in S: continue
        a, b = S[(c, "SEM-PM")], S[(c, "SEM-S")]
        L.append(f"{lab} & {fmt(a,'isl_mbps',0)} & {fmt(a,'quality_icu',1)} & {fmt(b,'isl_mbps',0)} & {fmt(b,'quality_icu',1)}\\\\")
    r = S[("default", "MC-SVC-PM")]
    L.append("\\midrule")
    L.append(f"MC-SVC-PM (ref.) & {fmt(r,'isl_mbps',0)} & {fmt(r,'quality_icu',1)} & -- & --\\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AGG, "table_sensitivity.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    runs = load(); S = summarize(runs)
    try:
        tables(S, runs)
    except KeyError as e:
        print("tables incomplete:", e)
