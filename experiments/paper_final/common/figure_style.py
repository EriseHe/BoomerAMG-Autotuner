"""Rendering shared by the manuscript's online and matched-policy figures."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
from matplotlib.lines import Line2D
import numpy as np

METHODS = ("default_setup_default_solve", "default_setup_lstdq", "bandit_default", "bandit_lstdq")
LABELS = ("Default", "LSTDQ", "LinUCB", "LinUCB + LSTDQ")
TEX_LABELS = ("Default", "LSTDQ", "LinUCB", "LinUCB--LSTDQ")
FIELDS = ("setup_runtime", "native_solve_runtime", "native_runtime", "infer_runtime", "bandit_overhead_runtime", "end_to_end_runtime")
# Restore the previously used Wes Anderson Zissou palette from the GitHub
# reference, with the same interpolation over relaxation weights 1–3.
ZISSOU = ("#3A9AB2", "#6FB2C1", "#91BAB6", "#A5C2A3", "#BDC881", "#DCCB4E", "#E3B710", "#E79805", "#EC7A05", "#EF5703", "#F11B00")
WEIGHT_CMAP = LinearSegmentedColormap.from_list("relaxation_weights", ZISSOU, N=256)
WEIGHT_CMAP.set_bad("white")
# Sample the displayed heatmap scale directly: LinUCB at 1, LSTDQ at the
# green anchor near 1.6, and Joint at 3. Default keeps its charcoal dash.
METHOD_WEIGHT_ANCHORS = (1.6, 1.0, 3.0)
COLORS = ("#273046",) + tuple(to_hex(WEIGHT_CMAP((weight - 1) / 2))
                              for weight in METHOD_WEIGHT_ANCHORS)
# User preference: Figure 3 always highlights LinUCB + LSTDQ in this red,
# independently of the heatmap palette (main panels, inset, and legend).
CUMULATIVE_COLORS = COLORS[:3] + ("#F11B00",)
HEATMAP_PROBLEM_RANGE = (1, 2500)


def overhead_tint(color):
    """Keep the phase hue while making recurring overhead visually lighter."""
    return to_hex(.45 * np.asarray(to_rgb(color)) + .55)


BAR_COMPONENTS = ((0, "Native setup", COLORS[2]),
                  (4, "Setup learner overhead", overhead_tint(COLORS[2])),
                  (1, "Native solve", COLORS[1]),
                  (3, "Solve controller overhead", overhead_tint(COLORS[1])))


def style():
    # Serif typography and method/phase colors from the current heatmap scale.
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 8,
        "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "text.color": "#202020", "axes.labelcolor": "#202020",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "axes.linewidth": .65,
        "axes.grid": False,
        "grid.color": "#dddddd", "grid.linewidth": .5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "paper-figures",
        "savefig.bbox": "tight", "savefig.pad_inches": .04})

def index_ticks(n, *, zero=True):
    ticks=np.unique(np.linspace(0,n,6,dtype=int))
    if not zero: ticks[0]=1
    return ticks


def heatmap_display_indices(chosen):
    """Keep the 50-cycle cap visible while compacting the standalone panel."""
    cap = chosen["w"].shape[-1]
    assert cap == 50
    return (list(range(21)) + [cap - 1], list(range(cap)))

def cycle_heatmap(chosen, title):
    first_problem, last_problem = HEATMAP_PROBLEM_RANGE
    last_problem = min(last_problem, chosen["w"].shape[1])
    assert 1 <= first_problem <= last_problem <= chosen["w"].shape[1]
    weights = chosen["w"][:, first_problem-1:last_problem, :]
    problem_ticks = np.linspace(first_problem-1, last_problem, 6, dtype=int)
    problem_ticks[0] = first_problem
    displayed = heatmap_display_indices(chosen)
    last = displayed[0][-1]
    cmap = WEIGHT_CMAP
    fig = plt.figure(figsize=(6.85, 3.1))
    grid = fig.add_gridspec(2, 1, height_ratios=[25, 50], hspace=.43,
                          left=.10, right=.87, bottom=.15, top=.84)
    upper = grid[0].subgridspec(2, 1, height_ratios=[21, 1.2], hspace=.45)
    top = fig.add_subplot(upper[0])
    endpoint = fig.add_subplot(upper[1], sharex=top)
    joint = fig.add_subplot(grid[1], sharex=top)
    for ax, q, first, final in [(top, 0, 0, 20), (endpoint, 0, last, last),
                               (joint, 1, 0, last)]:
        im = ax.imshow(weights[q, :, first:final+1].T, origin="upper", aspect="auto",
                       interpolation="nearest", extent=(first_problem-.5, last_problem+.5, final+.5, first-.5),
                       cmap=cmap, vmin=1, vmax=3, rasterized=True)
        ax.set_xlim(first_problem-.5, last_problem+.5)
        ax.set_xticks(problem_ticks)
    top.set(ylabel=r"Cycle index $k$", yticks=[0, 10, 20])
    top.set_title("(a) LSTDQ", loc="left")
    top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    top.spines["bottom"].set_visible(False)
    endpoint.set_yticks([last])
    endpoint.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    endpoint.spines["top"].set_visible(False)
    for ax, y in [(top, 0), (endpoint, 1)]:
        ax.plot([0, 1], [y, y], transform=ax.transAxes, linestyle="none",
                marker=[(-1, -.6), (1, .6)], markersize=4.2,
                markeredgewidth=.65, color="#202020", clip_on=False)
    joint.set(ylabel=r"Cycle index $k$", yticks=[0, 10, 20, 30, 40, last],
              xlabel="Problem index")
    joint.set_title("(b) LinUCB + LSTDQ", loc="left")
    color_ax = fig.add_axes([.90, .15, .017, .69])
    fig.colorbar(im, cax=color_ax, label="Relaxation weight", ticks=[1, 1.5, 2, 2.5, 3])
    fig.suptitle("RL relaxation-weight choices (" + title + ")", y=.965, fontsize=9)
    return fig

def draw_cycle_examples(data):
    fig = plt.figure(figsize=(9.15, 3.0))
    # Two main blocks; the right block contains a tightly paired comparison.
    outer = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.4], wspace=.30)
    schedule = fig.add_subplot(outer[0, 0])
    residual_grid = outer[0, 1].subgridspec(1, 2, wspace=.06)
    convergence = fig.add_subplot(residual_grid[0, 0])
    default = fig.add_subplot(residual_grid[0, 1], sharey=convergence)
    axes = (schedule, convergence, default)
    for ex in data["examples"]:
        default.semilogy(np.arange(ex["default_cycles"]+1), ex["default_relative_residuals"],
            color="#111111", linestyle=ex["linestyle"], marker=ex["marker"],
            lw=.85, ms=2.5, markerfacecolor="white", markeredgewidth=.6, zorder=1)
    schedule.axhline(1.0, color="#111111", ls="--", lw=.9, zorder=0)
    for ex in reversed(data["examples"]):
        style = dict(color=ex["color"], linestyle=ex["linestyle"], marker=ex["marker"],
                     lw=1.05, ms=3.0, markerfacecolor="white", markeredgewidth=.7, zorder=3)
        schedule.plot(np.arange(ex["cycles"]), ex["weights"],
            label=f'{ex["label"]}: input {ex["problem_index"]} ({ex["cycles"]} cycles)', **style)
        convergence.semilogy(np.arange(ex["cycles"]+1), ex["relative_residuals"], **style)
        if ex["exploration_cycles"]:
            k = np.array(ex["exploration_cycles"])
            schedule.scatter(k, np.array(ex["weights"])[k], marker="x", s=30,
                             color="#202020", linewidths=1.0, zorder=6)
    end = max(max(ex["cycles"], ex["default_cycles"]) for ex in data["examples"])
    ticks = np.arange(0, end+1, 2)
    schedule.set(xlabel=r"Cycle index $k$", ylabel=r"Relaxation weight $w_k$",
                 xlim=(-.35, end-.3), ylim=(.88, 3.13), xticks=ticks, yticks=[1, 1.5, 2, 2.5, 3])
    for ax in (convergence, default):
        ax.set(xlabel="Completed cycles", xlim=(-.35, end+.5), ylim=(1e-7, 2.0),
               xticks=np.arange(0, end+1, 4), yticks=[1, 1e-2, 1e-4, 1e-6])
        ax.axhline(1e-6, color="#666666", ls=":", lw=.7, zorder=0)
    convergence.set_ylabel("Relative residual")
    default.tick_params(axis="y", labelleft=False)
    schedule.set_title("(a) Learned relaxation schedules", loc="left", pad=8)
    convergence.set_title("(b) LinUCB + LSTDQ", loc="left", pad=8)
    default.set_title(r"(c) Default ($w=1$)", loc="left", pad=8)
    handles, labels = schedule.get_legend_handles_labels()
    handles, labels = list(reversed(handles)), list(reversed(labels))
    handles.extend([Line2D([], [], color="#111111", ls="--", lw=.9),
                    Line2D([], [], color="#202020", marker="x", ms=5, ls="None", markeredgewidth=1.0)])
    labels.extend([r"Default ($w=1$)", "Exploration"])
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, .915), ncol=5,
               frameon=False, handlelength=2.3, columnspacing=1.2, fontsize=7)
    fig.suptitle(rf"Learned relaxation schedules (Diffusion ${data['grid']}^3$, seed {data['seed']}, LinUCB + LSTDQ)",
                 fontsize=9.5, y=.995)
    fig.subplots_adjust(left=.065, right=.992, bottom=.175, top=.755)
    for ax in axes:
        ax.grid(False)
    return fig

def online_figures(a, chosen, seed, title, examples=None):
    """The manuscript's runtime stack, cumulative cost, weights, and examples."""
    n = a.shape[2]
    style()
    figures = []
    def save(fig, name):
        for ax in fig.axes:
            ax.grid(False, which="both")
            for child in ax.child_axes:
                child.grid(False, which="both")
        figures.append((name, fig))
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 3.05), sharey=True)
    for ax, start, letter in zip(axes, (0, max(0, n-1000)), ("a", "b")):
        costs = a[:, :, start:].mean(axis=2) * 1000
        means = costs.mean(axis=0)
        left = np.zeros(4)
        for field, name, color in BAR_COMPONENTS:
            ax.barh(np.arange(4), means[:, field], left=left, color=color, height=.64, label=name)
            left += means[:, field]
        for j in range(4):
            ax.text(left[j] + left.max() * .025, j, f'{costs[:, j, 5].mean():.1f} ± {(costs[:, j, 5].std(ddof=1) if len(a) > 1 else 0.):.1f}', va="center", fontsize=6.7)
        ax.set(xlabel="Measured cost (ms/input)", xlim=(0, left.max() * 1.49), yticks=range(4), yticklabels=LABELS)
        ax.set_title(f'({letter}) Problems {start+1}-{n}', loc="left")
    axes[0].invert_yaxis()
    fig.suptitle("Runtime components (" + title + f', {a.shape[0]} seeds)', y=.995, fontsize=9)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, -.01), fontsize=6.6, columnspacing=1.2, handlelength=1.5)
    fig.subplots_adjust(left=.17, right=.985, top=.84, bottom=.24, wspace=.18)
    save(fig, "runtime_breakdown")
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 3.15))
    totals = np.column_stack([np.zeros(4), np.cumsum(chosen["a"][:, :, 5], axis=1)])
    x = np.arange(n+1)
    for j in range(4):
        axes[0].plot(x, totals[j], color=CUMULATIVE_COLORS[j], ls="--" if j == 0 else "-", lw=1.3, label=LABELS[j] + f' ({totals[j,-1]:.1f} s)')
        if j:
            axes[1].plot(x[1:], 100*(1-totals[j,1:]/totals[0,1:]), color=CUMULATIVE_COLORS[j], lw=1.15)
    axes[1].axhline(0, color=CUMULATIVE_COLORS[0], ls="--", lw=.8)
    for ax in axes:
        ax.set(xlabel="Problem index", xlim=(0,n), xticks=index_ticks(n))
    axes[0].set(ylabel="Cumulative measured time (s)", ylim=(0,totals[0,-1]*1.06))
    axes[0].set_title("(a) Cumulative cost", loc="left")
    zoom = axes[0].inset_axes([.03, .54, .44, .44])
    # A square physical frame; the two data axes retain independent scales.
    zoom.set_box_aspect(1)
    zoom.set_gid("first_1000_square_inset")
    zoom_n = min(1000, n)
    for j in range(4):
        zoom.plot(x[:zoom_n+1], totals[j, :zoom_n+1], color=CUMULATIVE_COLORS[j], ls="--" if j == 0 else "-", lw=.6)
    zoom.set(xlim=(0, zoom_n), ylim=(0, totals[:, :zoom_n+1].max()*1.04),
             xticks=[0, zoom_n/2, zoom_n])
    zoom.tick_params(labelsize=5.5, length=2, pad=1)
    for spine in zoom.spines.values():
        spine.set_visible(True)
        spine.set_color("#888888")
        spine.set_linewidth(.55)
    axes[0].indicate_inset_zoom(zoom, edgecolor="#777777", alpha=.65, linewidth=.6)
    axes[1].set(ylabel="Reduction relative to Default (%)")
    axes[1].set_title("(b) Cumulative reduction", loc="left")
    fig.suptitle("Cumulative solver cost (" + title + f', seed {seed})', y=.995, fontsize=9)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
               frameon=False, fontsize=6.6, bbox_to_anchor=(.53,.94))
    fig.subplots_adjust(left=.10, right=.985, top=.72, bottom=.16, wspace=.45)
    save(fig, "cumulative_times")
    save(cycle_heatmap(chosen, title + f", seed {seed}"), "learned_cycle_weights")
    if examples is not None:
        with matplotlib.rc_context({"svg.fonttype": "path"}):
            save(draw_cycle_examples(examples), "individual_cycle_schedules")
    return figures


def matched_figures(native, weights, case_ids, selected, seeds, title):
    style()
    methods = ("reference", "fixed", "oracle", "periodic", "rl")
    labels = ("Global best fixed", "Per-problem best fixed", "Periodic (2.85, 1.10)", "Frozen RL")
    # Apply the compact manuscript style only to savings, preserving the
    # typography of the relaxation-weight panels below.
    with matplotlib.rc_context({"font.size": 8.5, "axes.titlesize": 9,
                                "axes.labelsize": 8.5, "xtick.labelsize": 8,
                                "ytick.labelsize": 8, "legend.fontsize": 8}):
        fig, ax = plt.subplots(figsize=(7.0, 1.58))
        extent = []
        for i, j in enumerate(range(1, 5)):
            gains = native[:, j]
            gains = gains[np.isfinite(gains)]
            if not len(gains):
                ax.text(.98, i, "undefined", transform=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=8)
                continue
            mean = float(gains.mean())
            sd = float(gains.std(ddof=1)) if len(gains) > 1 else 0.
            offsets = np.linspace(-.12, .12, len(gains)) if len(gains) > 1 else [0.]
            ax.scatter(gains, i + np.asarray(offsets), s=17, facecolors="white",
                       edgecolors="#222222", linewidths=.8, zorder=3)
            ax.errorbar(mean, i, xerr=sd, fmt="D", color="#222222",
                        markersize=4, elinewidth=.8, capsize=2, capthick=.7, zorder=4)
            endpoint = max(gains.max(), mean + sd)
            annotation = f"{mean:.1f} ± {sd:.1f}" if len(gains) > 1 else f"{mean:.1f}"
            ax.annotate(annotation, (endpoint, i), xytext=(7, 0),
                        textcoords="offset points", va="center", fontsize=8)
            extent.extend([gains.min(), gains.max(), mean - sd, mean + sd])
        if extent:
            # Keep every checkpoint and whisker visible on other datasets too;
            # reserve room on the right for the numerical annotations.
            span = max(max(extent) - min(extent), 1.)
            ax.set_xlim(min(extent) - .075 * span, max(extent) + .25 * span)
        ax.set(yticks=range(4), yticklabels=labels, ylim=(3.70, -.30),
               xlabel=r"Solver-only continuation-time reduction relative to $w=1$ (%)")
        ax.tick_params(axis="y", length=0, pad=7)
        ax.spines["left"].set_visible(False)
        ax.xaxis.labelpad = 6
        # The manuscript caption explains the markers and checkpoint population.
        fig.subplots_adjust(left=.235, right=.985, top=.962, bottom=.332)
        figures = [("matched_policy_savings", fig)]
    seed=seeds[selected]
    order=np.lexsort((case_ids,-np.isfinite(weights[selected,0]).sum(axis=1)))
    indices=[0,2,3,4]
    cap=max(1,int(np.max(np.isfinite(weights[selected,indices]).sum(axis=2))))
    shown={methods[j]:weights[selected,j,order,:cap] for j in indices}
    ticks=np.unique(np.r_[np.arange(0,cap,10),cap-1])
    fig,axes=plt.subplots(2,2,figsize=(6.85,3.6),sharex=True,sharey=True)
    panel_labels=(r"Fixed $w=1$","Per-instance best fixed","Periodic (2.85, 1.10)","Frozen RL")
    for ax,j,label,letter in zip(axes.flat,indices,panel_labels,"abcd"):
        im=ax.imshow(shown[methods[j]].T,origin="upper",interpolation="nearest",aspect="auto",
                     extent=(.5,len(case_ids)+.5,cap-.5,-.5),cmap=WEIGHT_CMAP,vmin=1,vmax=3,rasterized=True)
        ax.set_title(f"({letter}) {label}",loc="left")
        ax.set(xticks=np.unique(np.r_[1,np.linspace(0,len(case_ids),5,dtype=int)[1:]]),yticks=ticks)
    for ax in axes[:,0]: ax.set_ylabel(r"Cycle index $k$")
    for ax in axes[1]: ax.set_xlabel("Test problem rank")
    fig.subplots_adjust(left=.10,right=.855,top=.83,bottom=.14,wspace=.12,hspace=.4)
    color_ax=fig.add_axes([.89,.14,.017,.69])
    fig.colorbar(im,cax=color_ax,label="Relaxation weight",ticks=[1,1.5,2,2.5,3])
    fig.suptitle(f"Matched-policy relaxation weights ({title}, seed {seed})",y=.97,fontsize=9)
    figures.append(("matched_policy_weights",fig))
    return figures
