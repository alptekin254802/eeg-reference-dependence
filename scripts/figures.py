"""Generate the five main figures."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results', type=Path, default=Path('results'))
parser.add_argument('--output', type=Path, default=Path('outputs/figures'))
args = parser.parse_args()
SRC, OUT = args.results, args.output
OUT.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
COL1, COL2 = 84 * MM, 174 * MM  # Springer single / double column widths
C = {"linked": "#234E70", "common": "#C66B18", "car": "#16827A", "neutral": "#5A6068",
     "ref": "#9AA0A6", "text": "#1A1A1A"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 0, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white", "mathtext.fontset": "custom",
    "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
})

vals = {v["pointer"]: v["value"] for v in json.load(open(SRC / "figure_values.json"))["values"]}


def get(prefix, est="auc", ci="auc_ci95"):
    return vals[f"{prefix}/{est}"], vals[f"{prefix}/{ci}"]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def forest(ax, rows, xlim, null, xlabel, value_fmt="{:.3f}", log=False):
    """rows: list of dicts with y, est, lo, hi, color, label (None for header rows)."""
    for r in rows:
        if r.get("est") is None:
            continue
        ax.plot([r["lo"], r["hi"]], [r["y"], r["y"]], color=r["color"], lw=1.1,
                solid_capstyle="butt", zorder=3)
        ax.plot(r["est"], r["y"], "o", ms=3.6, color=r["color"], mec="white", mew=0.5, zorder=4)
        ax.text(1.02, r["y"], value_fmt.format(r["est"]), transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=8, color=C["text"])
    if null is not None:
        ax.axvline(null, color=C["ref"], lw=0.6, ls=(0, (2.5, 2)), zorder=1)
    if log:
        ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, labelpad=2)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_yticks([r["y"] for r in rows if r.get("label")])
    ax.set_yticklabels([r["label"] for r in rows if r.get("label")])
    import matplotlib.transforms as mt
    tr = mt.blended_transform_factory(ax.figure.transFigure, ax.transData)
    for r in rows:
        if r.get("header"):
            ax.text(0.03, r["y"], r["header"], transform=tr, ha="left", va="center",
                    fontsize=8, weight="bold", color=C["text"])


# ---------------------------------------------------------------- Figure 1
def figure1():
    fig = plt.figure(figsize=(COL2, 62 * MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 174); ax.set_ylim(0, 62); ax.axis("off")

    def box(x, y, w, h, color, title, body):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                    fc="white", ec=color, lw=0.9))
        ax.add_patch(FancyBboxPatch((x, y), 1.4, h, boxstyle="square,pad=0", fc=color, ec="none"))
        ax.text(x + w / 2 + 0.7, y + h * 0.66, title, ha="center", va="center", fontsize=8,
                weight="bold", color=color)
        ax.text(x + w / 2 + 0.7, y + h * 0.30, body, ha="center", va="center", fontsize=8,
                color=C["text"])

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18",
                                    lw=0.7, color=C["neutral"], shrinkA=0, shrinkB=0))

    # input
    box(4, 22, 44, 18, C["neutral"], "As-released EEG (19 channels)",
        r"$X_j(t)=V_j(t)-R(t)$")
    ax.text(26, 15.5, r"$X_j(t)=m(t)+X_j^{\mathrm{CAR}}(t)$", ha="center", va="center",
            fontsize=8, color=C["text"])
    # representations
    rows = [(43, C["linked"], "Linked-mastoid", r"$X_j(t)$"),
            (24, C["common"], "Spatial mean (common component)",
             r"$m(t)=\overline{V}(t)-R(t)$"),
            (5, C["car"], "CAR residual", r"$X_j^{\mathrm{CAR}}(t)=X_j(t)-m(t)$")]
    for y, col, title, body in rows:
        box(64, y, 56, 14, col, title, body)
        arrow(48.5, 31, 63.5, y + 7)
    # models
    models = [(43, C["linked"], "EEGNet", "19 channels"),
              (24, C["common"], "EEGNet", r"$m(t)$ repeated on 19 inputs"),
              (5, C["car"], "EEGNet", "19 channels")]
    for y, col, title, body in models:
        box(132, y, 38, 14, col, title, body)
        arrow(120.5, y + 7, 131.5, y + 7)
    save(fig, "Figure1_decomposition")


# ---------------------------------------------------------------- Figure 2
def figure2():
    p = "/classification/fixed30~1eegnet~1{}_basic~1ensemble3~1split20261109~1final"
    m = "/classification/age_sex_matched~1eegnet~1{}_basic~1seed1109~1split20261109~1final"
    lk, lkc = get(p.format("linked")); cr, crc = get(p.format("average"))
    co, coc = get("/common_component_retraining/metrics/common_only_ensemble", "auc", "ci95")
    ml, mlc = get(m.format("linked")); mc, mcc = get(m.format("average"))
    rows = [
        {"y": 6.0, "header": "Full cohort, N = 189"},
        {"y": 5.0, "label": "Linked-mastoid", "est": lk, "lo": lkc[0], "hi": lkc[1], "color": C["linked"]},
        {"y": 4.0, "label": "CAR", "est": cr, "lo": crc[0], "hi": crc[1], "color": C["car"]},
        {"y": 3.0, "label": "Common only", "est": co, "lo": coc[0], "hi": coc[1], "color": C["common"]},
        {"y": 1.6, "header": "Age/sex-matched, N = 112"},
        {"y": 0.6, "label": "Linked-mastoid", "est": ml, "lo": mlc[0], "hi": mlc[1], "color": C["linked"]},
        {"y": -0.4, "label": "CAR", "est": mc, "lo": mcc[0], "hi": mcc[1], "color": C["car"]},
    ]
    fig, ax = plt.subplots(figsize=(COL1, 56 * MM))
    fig.subplots_adjust(left=0.38, right=0.86, top=0.97, bottom=0.17)
    forest(ax, rows, (0.4, 1.0), 0.5, "Participant-level AUC")
    ax.set_ylim(-1.0, 6.5)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.axhline(2.3, color="#D5D8DC", lw=0.5, xmin=-0.5, clip_on=False)
    save(fig, "Figure2_discrimination")


# ---------------------------------------------------------------- Figure 3
def figure3():
    base = "/common_component_power/cohorts/primary_186/features/{}_{}_log10_power"
    bands = [("delta", "Delta, 1–4 Hz"), ("theta", "Theta, 4–8 Hz"),
             ("alpha", "Alpha, 8–13 Hz"), ("beta", "Beta, 13–30 Hz"),
             ("gamma", "Gamma, 30–40 Hz"), ("total_1_40", "Total, 1–40 Hz")]
    fig, ax = plt.subplots(figsize=(COL1, 66 * MM))
    fig.subplots_adjust(left=0.34, right=0.97, top=0.82, bottom=0.16)
    ys = []
    for i, (key, label) in enumerate(bands):
        y = len(bands) - 1 - i - (0.35 if key.startswith("total") else 0)
        ys.append((y, label))
        for comp, dy in (("common", 0.17), ("car", -0.17)):
            est, ci = get(base.format(comp, key), "geometric_ratio", "geometric_ratio_ci95")
            ax.plot(ci, [y + dy] * 2, color=C[comp], lw=1.1, solid_capstyle="butt", zorder=3)
            ax.plot(est, y + dy, "o", ms=3.6, color=C[comp], mec="white", mew=0.5, zorder=4)
    ax.axvline(1, color=C["ref"], lw=0.6, ls=(0, (2.5, 2)), zorder=1)
    ax.axhline(0.32, color="#D5D8DC", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlim(0.25, 1.3)
    ax.set_xticks([0.25, 0.5, 1.0]); ax.set_xticklabels(["0.25", "0.5", "1"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks([y for y, _ in ys]); ax.set_yticklabels([l for _, l in ys])
    ax.get_yticklabels()[-1].set_weight("bold")
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_ylim(-0.8, len(bands) - 0.5)
    ax.set_xlabel("ADHD/control power ratio (log scale)", labelpad=2)
    h1 = ax.plot([], [], "o-", color=C["common"], ms=3.6, lw=1.1, mec="white", mew=0.5)[0]
    h2 = ax.plot([], [], "o-", color=C["car"], ms=3.6, lw=1.1, mec="white", mew=0.5)[0]
    fig.legend([h1, h2], ["Common component", "CAR residual"], loc="upper center", ncol=1,
               frameon=False, bbox_to_anchor=(0.65, 1.0), handlelength=1.6, columnspacing=1.4,
               fontsize=8)
    save(fig, "Figure3_power")


# ---------------------------------------------------------------- Figure 4
def figure4():
    a = "/common_bandpower_covariates/arms/{}/auc"
    def arm(k):
        return vals[a.format(k) + "/estimate"], vals[a.format(k) + "/ci95"]
    spec = [("B0", "Unadjusted"), ("BAge", "Age, sex"), ("BD", "Age, sex, quality"),
            ("BH", " + heart rate"), ("BM", " + MASS power,\ndetector-agreement F1")]
    rows = []
    for i, (k, lab) in enumerate(spec):
        e, ci = arm(k)
        rows.append({"y": len(spec) - 1 - i, "label": lab, "est": e, "lo": ci[0], "hi": ci[1],
                     "color": C["common"]})
    fig, ax = plt.subplots(figsize=(COL1, 46 * MM))
    fig.subplots_adjust(left=0.46, right=0.86, top=0.86, bottom=0.21)
    forest(ax, rows, (0.4, 1.0), 0.5, "Participant-level AUC")
    ax.set_ylim(-0.6, len(spec) - 0.4)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    save(fig, "Figure4_covariates")


# ---------------------------------------------------------------- Figure 5
def figure5():
    k = "/pairwise_top_region_agreement/comparisons/basic~1IG~1own_predicted_class/aggregate/{}/kappa"
    def kap(name):
        return vals[k.format(name) + "/estimate"], vals[k.format(name) + "/ci95_percentile"]
    spec = [("within_linked", "Within linked-mastoid", C["linked"]),
            ("within_average", "Within CAR", C["car"]),
            ("cross_same_seed", "Across references", C["neutral"])]
    rows = []
    for i, (name, lab, col) in enumerate(spec):
        e, ci = kap(name)
        rows.append({"y": len(spec) - 1 - i, "label": lab, "est": e, "lo": ci[0], "hi": ci[1],
                     "color": col})
    fig, ax = plt.subplots(figsize=(COL1, 34 * MM))
    fig.subplots_adjust(left=0.43, right=0.86, top=0.84, bottom=0.30)
    forest(ax, rows, (0.0, 0.8), None, "Mean pairwise Cohen’s kappa")
    ax.set_ylim(-0.6, len(spec) - 0.4)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    save(fig, "Figure5_agreement")


if __name__ == "__main__":
    figure1(); figure2(); figure3(); figure4(); figure5()
    import hashlib, platform
    def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    aggregate=json.loads((SRC / "aggregate.json").read_text())
    checks=[]
    for pointer,value in vals.items():
        node=aggregate
        for token in pointer.lstrip("/").split("/"):
            node=node[token.replace("~1","/").replace("~0","~")]
        assert node == value, pointer
        checks.append(dict(pointer=pointer,value=value,exact_match=True))
    report=dict(numeric_checks=checks,
        input_sha256={p.name:sha(p) for p in (SRC/"figure_values.json", SRC/"aggregate.json")},
        output_sha256={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.suffix in (".pdf",".png",".py",".ced")},
        font="8 pt sans-serif; installed font fallback follows matplotlib settings")
    (OUT/"validation.json").write_text(json.dumps(report,indent=2), encoding="utf-8")
    print("written to", OUT.resolve())
