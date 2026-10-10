"""Static figures for a run report (matplotlib, Agg).

Palette and chrome follow the dataviz reference palette: categorical hues in
fixed slot order, an ordinal one-hue ramp for temperature, recessive grid,
text in ink colours, a legend for every multi-series panel, and the replica
floor (distortion ratio 1) drawn as a labelled reference line.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ORDINAL_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
METHOD_ORDER = ["spectral_subspace_local", "spectral_subspace", "spectral_band_local", "pca_exp", "dct"]
METHOD_LABEL = {
    "spectral_subspace_local": "Spectral subspace + local",
    "spectral_subspace": "Spectral subspace",
    "spectral_band_local": "Spectral per-band + local",
    "pca_exp": "PCA + 1 timescale",
    "dct": "Truncated DCT (realisation)",
}
METRIC_LABEL = {
    "msf_rel": "MSF (RMSF²) profile",
    "log_msf_rmse": "log MSF",
    "aniso": "Anisotropy shape",
    "acf_rmse": "Autocorrelation",
    "dir_acf_rmse": "Directional autocorrelation",
    "dccm_rmse": "Cross-correlation (lag 0)",
    "lagged_dccm_rmse": "Lagged cross-correlation",
    "frob_rel": "Full lagged covariance",
    "max": "Worst metric",
}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8, "font.family": "sans-serif",
        "lines.linewidth": 2.0, "lines.markersize": 5,
    })


def _floor_line(ax, label=True):
    ax.axhline(1.0, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=1)
    if label:
        ax.annotate("replica floor", xy=(1.0, 1.0), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                    textcoords="offset points", ha="right", va="bottom", fontsize=7, color=INK2)


def _log(ax, x: bool = True, y: bool = True) -> None:
    """Log axes with plain 1-2-5 tick labels instead of 3x10^0 notation."""

    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    for flag, axis, setter in ((x, ax.xaxis, ax.set_xscale), (y, ax.yaxis, ax.set_yscale)):
        if not flag:
            continue
        setter("log")
        axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        axis.set_major_formatter(fmt)
        axis.set_minor_formatter(NullFormatter())


def _safe(name: str) -> str:
    return name.replace("/", "_")


def rate_distortion(path: Path, group: str, curves: list[dict]) -> None:
    metrics = ["msf_rel", "aniso", "acf_rmse", "dir_acf_rmse", "dccm_rmse", "lagged_dccm_rmse", "frob_rel", "max"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), layout="constrained", sharex=True)
    methods = [m for m in METHOD_ORDER if any(c["method"] == m for c in curves)]
    for ax, metric in zip(axes.flat, metrics):
        for method in methods:
            pts = [(c["params_per_res"]["median"], c[f"rel_{metric}"]["median"],
                    c[f"rel_{metric}"]["q25"], c[f"rel_{metric}"]["q75"])
                   for c in curves if c["method"] == method and c["params_per_res"]["median"]
                   and c[f"rel_{metric}"]["median"] is not None]
            if not pts:
                continue
            pts.sort()
            x, y, lo, hi = map(np.asarray, zip(*pts))
            color = SERIES[METHOD_ORDER.index(method)]
            ax.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0)
            ax.plot(x, y, color=color, marker="o", ms=4, label=METHOD_LABEL[method])
        _floor_line(ax, label=(metric == metrics[0]))
        _log(ax)
        ax.set_title(METRIC_LABEL[metric])
    for ax in axes[1]:
        ax.set_xlabel("Parameters per residue")
    for ax in axes[:, 0]:
        ax.set_ylabel("Distortion / replica floor")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(f"{group}: rate-distortion against a held-out replica (median across proteins, IQR band)",
                 color=INK, fontsize=11)
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))
    fig.savefig(path / f"rate_distortion_{_safe(group)}.png", dpi=150)
    plt.close(fig)


def rank_needed(path: Path, group: str, rows: list[dict], threshold: float = 1.1) -> None:
    by_target: dict = {}
    for r in rows:
        if r["method"] != "spectral_subspace_local" or r.get("rel_max") is None:
            continue
        if f"{r['source']}/T{r['temperature']}" != group:
            continue
        rank = r.get("rank_actual", r["param"])
        if not isinstance(rank, (int, float)):
            continue
        by_target.setdefault(r["domain"], {"n": r["n_res"], "vals": {}})["vals"].setdefault(
            int(rank), []).append(r["rel_max"])
    xs, ys = [], []
    for item in by_target.values():
        hit = [p for p, v in sorted(item["vals"].items()) if np.mean(v) <= threshold]
        if hit:
            xs.append(item["n"])
            ys.append(max(hit[0], 0.5))
    if len(xs) < 2:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.2), layout="constrained")
    lx, ly = np.log(xs), np.log(ys)
    label = None
    if np.ptp(lx) > 0 and len(xs) >= 4:
        slope, icpt = np.polyfit(lx, ly, 1)
        grid = np.linspace(lx.min(), lx.max(), 50)
        ax.plot(np.exp(grid), np.exp(icpt + slope * grid), color=INK2, lw=1.2, zorder=2)
        label = f"log-log slope {round(float(slope), 2) + 0.0:.2f}"
    ax.scatter(xs, ys, s=30, color=SERIES[0], edgecolor=SURFACE, linewidth=1.0, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log", base=2)
    top = max(2.0, 2.0 ** np.ceil(np.log2(max(ys) * 1.5)))
    ticks = [0.5] + [2.0**k for k in range(0, int(np.log2(top)) + 1)]
    ax.set_yticks(ticks, ["0"] + [f"{int(t)}" for t in ticks[1:]])
    ax.set_ylim(0.4, top * 1.2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Residues (N)")
    ax.set_ylabel(f"Collective rank to reach {threshold:g}x floor")
    ax.set_title(f"{group}: subspace rank needed, worst metric" + (f"\n{label}" if label else ""))
    fig.savefig(path / f"rank_needed_{_safe(group)}.png", dpi=150)
    plt.close(fig)


def pca_vs_n(path: Path, group: str, rows: list[dict]) -> None:
    sub = [r for r in rows if f"{r['source']}/T{r['temperature']}" == group]
    series = [("ca_pca_rep_n_pc_90", "CA, per replica"), ("bb_pca_rep_n_pc_90", "Backbone, per replica"),
              ("ca_pca_pooled_n_pc_90", "CA, pooled replicas")]
    fig, ax = plt.subplots(figsize=(5.5, 4.2), layout="constrained")
    drawn = 0
    markers = ["o", "s", "^"]
    for slot, (key, label) in enumerate(series):
        pts = [(r["n_res"], r.get(key)) for r in sub if r.get(key) not in (None, "") and np.isfinite(float(r[key]))]
        if not pts:
            continue
        x, y = zip(*pts)
        # Hollow, shape-coded markers keep coincident series visible.
        ax.scatter(x, y, s=34, marker=markers[slot], facecolor="none", edgecolor=SERIES[slot],
                   linewidth=1.6, label=label, zorder=3)
        drawn += 1
    if not drawn:
        plt.close(fig)
        return
    _log(ax)
    ax.set_xlabel("Residues (N)")
    ax.set_ylabel("PCs for 90% of coordinate variance")
    ax.set_title(f"{group}: spatial PCA compressibility")
    ax.legend(loc="upper left")
    fig.savefig(path / f"pca90_vs_N_{_safe(group)}.png", dpi=150)
    plt.close(fig)


def reliability(path: Path, stage0: dict) -> None:
    sources = sorted({g.split("/")[0] for g in stage0})
    for source in sources:
        groups = sorted([g for g in stage0 if g.startswith(source + "/")], key=lambda g: float(g.split("/T")[1]))
        panels = [("acf_corr_median", "Per-residue autocorrelation"),
                  ("lagged_dcc_corr_median", "Lagged cross-correlation"),
                  ("asym_replica_cos_median", "Antisymmetric part (irreversibility)")]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
        ramp = ORDINAL_BLUE if len(groups) > 1 else [SERIES[0]]
        step = max(1, len(ORDINAL_BLUE) // max(len(groups), 1))
        for ax, (key, title) in zip(axes, panels):
            for i, group in enumerate(groups):
                curve = stage0[group]["reliability_vs_lag"]
                x = np.asarray(curve["lag_ps"]) / 1000.0
                y = np.asarray(curve[key], dtype=np.float64)
                color = ramp[min(i * step, len(ramp) - 1)] if len(groups) > 1 else ramp[0]
                ax.plot(x, y, color=color, marker="o", ms=3, label=f"{group.split('/T')[1]} K")
            ax.axhline(0.0, color=AXIS, lw=1.0)
            if key != "asym_replica_cos_median":
                ax.axhline(0.5, color=INK2, lw=1.0, ls=(0, (4, 3)))
                ax.set_ylim(-0.2, 1.02)
            else:
                ax.set_ylim(-1.0, 1.0)
            _log(ax, y=False)
            ax.set_xlabel("Lag (ns)")
            ax.set_title(title)
        axes[0].set_ylabel("Replica-vs-replica correlation (median)")
        axes[2].set_ylabel("Replica-vs-replica cosine (median)")
        if len(groups) > 1:
            axes[0].legend(title="Temperature", loc="lower left")
        fig.suptitle(f"{source}: how far in lag are profile statistics reproducible?", color=INK, fontsize=11)
        fig.savefig(path / f"reliability_vs_lag_{source}.png", dpi=150)
        plt.close(fig)


def universality_plot(path: Path, universality: dict) -> None:
    for group, u in universality.items():
        if not u.get("n_targets"):
            continue
        fig, ax = plt.subplots(figsize=(5.5, 4.2), layout="constrained")
        k = np.arange(1, len(u["signal_cumulative"]) + 1)
        ax.plot(k, u["signal_cumulative"], color=SERIES[0], label="All proteins (in-sample)")
        if u.get("held_out_captured"):
            ax.plot(k, u["held_out_captured"], color=SERIES[1], label="Held-out proteins")
        for level in (0.9, 0.95):
            ax.axhline(level, color=AXIS, lw=1.0, ls=(0, (4, 3)))
        _log(ax, y=False)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Shared descriptor dimensions")
        ax.set_ylabel("Fraction of replica-reproducible variance")
        ax.set_title(f"{group}: per-residue dynamics vocabulary\nreliable fraction {u['reliable_fraction']:.2f}")
        ax.legend(loc="lower right")
        fig.savefig(path / f"universality_{_safe(group)}.png", dpi=150)
        plt.close(fig)


def make_figures(report_dir: Path, stage0_rows: list[dict], stage1_rows: list[dict], summary: dict) -> None:
    _style()
    out = Path(report_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for group, entry in summary.get("stage1", {}).items():
        rate_distortion(out, group, entry["curves"])
        rank_needed(out, group, stage1_rows)
    for group in summary.get("stage0", {}):
        pca_vs_n(out, group, stage0_rows)
    if summary.get("stage0"):
        reliability(out, summary["stage0"])
    universality_plot(out, summary.get("universality", {}))
