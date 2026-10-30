"""Recreate the paper's Module 1 figures from a completed four-branch run.

Usage: python -m experiments.paper_final.plot_online RUN_DIRECTORY
No solver or learner is executed. Figures, plotted arrays, captions and hashes
are saved under RUN_DIRECTORY/paper_figures (or --output).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from experiments.runtime import configure_single_thread

configure_single_thread()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

METHODS = (
    "default_setup_default_solve", "default_setup_lstdq",
    "bandit_default", "bandit_lstdq",
)
LABELS = ("Default", "Default + LSTDQ", "LinUCB", "LinUCB + LSTDQ")
COLORS = ("#333333", "#009E73", "#0072B2", "#D55E00")
FIELDS = (
    "setup_runtime", "native_solve_runtime", "native_runtime",
    "infer_runtime", "bandit_overhead_runtime", "end_to_end_runtime",
)
from experiments.paper_final.common.figure_style import online_figures

RL_INDICES = (1, 3)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_run(root):
    protocol = json.loads((root / "protocol.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    completion = json.loads((root / "completion_audit.json").read_text())
    assert summary["passed"] and completion["passed"], "Run has not passed its audits"
    assert tuple(protocol["methods"]) == METHODS
    seeds = list(protocol["seeds"])
    sources = {str(root / name): sha(root / name) for name in
               ("protocol.json", "summary.json", "completion_audit.json")}
    timings, weights, residuals, configs = [], [], [], []
    for seed in seeds:
        stage = root / f"seed_{seed}"
        config_path = stage / "experiment_config.json"
        sources[str(config_path)] = sha(config_path)
        config = json.loads(config_path.read_text())
        index_offset = config.get("evaluation", {}).get("training_cases", 0)
        result = json.loads((stage / "result.json").read_text())
        configs.append(config)
        n = protocol["stages"][seed]["cases"]
        cap = config["solve"]["max_cycles"]
        a = np.zeros((4, n, len(FIELDS)))
        w = np.full((2, n, cap), np.nan)
        residual = np.full((2, n, cap + 1), np.nan)
        residual[:, :, 0] = 1.
        paired = None
        specs = {m["id"]: m for m in config["methods"]}
        for j, method in enumerate(METHODS):
            source = stage / "trajectories" / f"{method}.jsonl"
            digest = hashlib.sha256()
            input_digest = hashlib.sha256()
            count = 0
            with source.open("rb") as handle:
                for i, line in enumerate(handle):
                    digest.update(line)
                    row = json.loads(line)
                    assert i < n and row["online_index"] == i
                    assert row["problem_index"] == index_offset + i + 1 and row["method"] == method
                    input_digest.update(json.dumps(
                        [row["mkw"], row["context"]], sort_keys=True,
                        separators=(",", ":"), allow_nan=False).encode())
                    o = row["outcome"]
                    v = np.array([o[k] for k in FIELDS])
                    assert np.isfinite(v).all() and (v >= 0).all()
                    assert abs(v[0] + v[1] - v[2]) < 1e-8
                    assert abs(v[2] + v[3] + v[4] - v[5]) < 1e-8
                    a[j, i] = v
                    if j in RL_INDICES:
                        q = RL_INDICES.index(j)
                        # The outcome-level trace may describe the final learned
                        # attempt after reselection. Use the first attempt explicitly.
                        first = o["primary_attempts"][0]
                        cycles = int(first["cycles"])
                        assert 0 <= cycles <= cap
                        active = i >= specs[method]["solve_activation_case"]
                        assert bool(row["rl_active"]) == active
                        actions = np.asarray(first["cycle_actions"], dtype=float)
                        if not active:
                            assert actions.size == 0
                            actions = np.ones(cycles)
                        assert actions.size == cycles
                        grid = config["solve"]["action_grid"]
                        assert ((actions >= grid["start"] - 1e-9) &
                                (actions <= grid["stop"] + 1e-9)).all()
                        steps = (actions - grid["start"]) / grid["step"]
                        assert np.allclose(steps, np.round(steps), atol=1e-8)
                        w[q, i, :cycles] = actions
                        if active:
                            r = np.asarray(first["cycle_residuals"], dtype=float)
                            assert r.size == cycles
                            assert (r[np.isfinite(r)] >= 0).all()
                            if first["status"] == "success":
                                assert np.isfinite(r).all()
                            r[~np.isfinite(r)] = np.nan
                            if cycles:
                                residual[q, i, 1:] = r[np.minimum(np.arange(cap), cycles - 1)]
                            else:
                                residual[q, i, 1:] = 1. if first["status"] == "success" else np.nan
                    count += 1
            assert count == n
            sources[str(source)] = digest.hexdigest()
            assert sources[str(source)] == result["audits"][method]["trajectory_sha256"]
            if paired is None:
                paired = input_digest.hexdigest()
            assert input_digest.hexdigest() == paired, "Methods must share the exact input order"
            for record in (r for r in summary["rows"] if r["seed"] == seed and
                           r["method"] == method and r["window"] in ("full", "last_1000")):
                start = 0 if record["window"] == "full" else max(0, n - 1000)
                assert record["cases"] == n - start
                np.testing.assert_allclose(a[j, start:, 2].sum(), record["native_sec"], atol=1e-7, rtol=1e-10)
                np.testing.assert_allclose(a[j, start:, 5].sum(), record["total_sec"], atol=1e-7, rtol=1e-10)
        timings.append(a)
        weights.append(w)
        residuals.append(residual)
    assert all(c["problem"] == configs[0]["problem"] and
               c["solve"] == configs[0]["solve"] and
               c["methods"] == configs[0]["methods"] for c in configs)
    return protocol, seeds, np.stack(timings), np.stack(weights), np.stack(residuals), configs[0], sources




def generate(run, *, output=None, illustrative_seed=None, cycle_examples=None):
    root = Path(run).expanduser().resolve()
    output = Path(output or root / "paper_figures").expanduser().resolve()
    protocol, seeds, a, w, residuals, config, sources = read_run(root)
    native_totals = a[:, :, :, 2].sum(axis=2)
    reductions = 100 * (1 - native_totals / native_totals[:, :1])
    total_totals = a[:, :, :, 5].sum(axis=2)
    total_reductions = 100 * (1 - total_totals / total_totals[:, :1])
    selected = seeds.index(str(illustrative_seed)) if illustrative_seed is not None else int(np.argmax(total_reductions[:, 3]))
    seed = seeds[selected]
    selection = ("Explicit --illustrative-seed override" if illustrative_seed is not None else
                 "Largest full-stream paired total-time reduction of LinUCB + LSTDQ among this run's seeds; same selection rule as the paper")
    activation = next(m["solve_activation_case"] for m in config["methods"] if m["id"] == METHODS[3])
    grid = config["problem"]["grid"]
    kind = config["problem"]["kind"]
    family = "Diffusion–advection" if "advection" in kind or "convection" in kind else "Diffusion"
    title = family + " " + (rf"${grid[0]}^3$" if len(set(grid)) == 1 else " × ".join(map(str, grid)))
    module_number = 2 if kind.startswith("rs_") else 1
    output.mkdir(parents=True, exist_ok=True)
    examples = json.loads(Path(cycle_examples).read_text()) if cycle_examples is not None else None
    if cycle_examples is not None:
        sources[str(Path(cycle_examples).resolve())] = sha(Path(cycle_examples))
    figures = online_figures(a, {"a": a[selected], "w": w[selected]}, seed, title, examples)
    if config.get("evaluation"):
        from matplotlib.ticker import FuncFormatter
        offset = config["evaluation"]["training_cases"]
        for _, fig in figures:
            for ax in fig.axes:
                if ax.get_xlabel() == "Problem index":
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value+offset:g}'))
                for text in ax.texts:
                    if "from problem 1" in text.get_text():
                        text.set_text("Frozen policy")
            for text in fig.texts:
                if "from problem 1" in text.get_text():
                    text.set_text("Frozen policies; held-out inputs 5,001–6,000")
    if config.get("frozen_stream"):
        for _, fig in figures:
            for ax in fig.axes:
                if ax.get_xlabel() == "Problem index":
                    ax.axvline(5000.5, color="#555555", ls=":", lw=.8)
    metadata = {"Title": f"Module {module_number} {family} {grid}: four branches",
                "Author": "RLTune-BoomerAMG", "Subject": "Figures from saved trajectories; no experiment rerun"}
    with PdfPages(output / f"module{module_number}_paper_figures.pdf", metadata=metadata) as pdf:
        for name, fig in figures:
            options = {"svg.fonttype": "path"} if name == "individual_cycle_schedules" else {}
            with matplotlib.rc_context(options):
                fig.savefig(output / f"{name}.pdf", dpi=600, metadata=metadata)
                fig.savefig(output / f"{name}.png", dpi=260)
                fig.savefig(output / f"{name}.svg", dpi=600)
            pdf.savefig(fig, dpi=600, bbox_inches="tight", pad_inches=.04)
            plt.close(fig)
    np.savez_compressed(output / "plot_data.npz", seeds=np.array(seeds), methods=np.array(METHODS),
                        timing_fields=np.array(FIELDS), timings_sec=a,
                        rl_methods=np.array([METHODS[j] for j in RL_INDICES]),
                        first_primary_weights=w, first_primary_residuals=residuals,
                        illustrative_seed=np.array(seed))
    n = a.shape[2]
    activation_caption = ("Both LSTDQ branches are active from problem 1." if activation == 0 else
                          f"The dotted line marks joint LSTDQ activation at problem {activation + 1}; Default + LSTDQ is active from problem 1.")
    if config.get("evaluation"):
        activation_caption = "Both learners are frozen at the final Module 1 checkpoint; these are held-out evaluation inputs 5,001–6,000."
    if config.get("frozen_stream"):
        activation_caption = "One continuous worker stream: learning on inputs 1–5,000, then both learners frozen with epsilon exploration disabled on held-out inputs 5,001–6,000. Inference overhead is measured in both phases."
    prefix_caption = ("A single measured setup prefix is included once in each setup continuation's budget; the learned setup state is copied at the activation boundary." if protocol.get("shared_prefix") else
                      "All four branches are measured independently from input one.")
    dispersion_caption = (f"mean ± sample SD across {len(seeds)} seed-level mean costs" if len(seeds) > 1 else
                          "the single seed-level mean cost; sample SD is unavailable")
    captions = (
        f"These figures use Module {module_number} {family} {grid}, {n} inputs per seed, seeds {', '.join(seeds)}, and all four branches. {prefix_caption}\n"
        f"Detailed illustration: seed {seed}. Selection: {selection}. This selected run is not a typical-run estimate.\n\n"
        f"cumulative_times: (a) Cumulative total recorded time; (b) percentage reduction against the paired Default, computed from exact prefix totals without smoothing. All attempted inputs are retained, including final failures if any. Total time includes every setup/solve attempt, recovery, and measured learner overhead. Legend values are full-stream seconds. {activation_caption}\n\n"
        f"runtime_breakdown: Mean time for the full stream" + (" and final 1000 problems" if n > 1000 else "") +
        f". Stacks show native setup, setup-learner overhead, native solve, and solve-controller overhead. Labels are full measured cost as {dispersion_caption} (ms/problem); recovery is included once.\n\n"
        "learned_cycle_weights: Executed relaxation weights for the first primary attempt of inputs 1–2500 (or the available shorter stream). Rows are zero-based action indices with cycle zero at the top. The inactive joint prefix is reconstructed as weight one from recorded first-attempt cycle counts. White cells follow first-attempt termination. Learned reselection and fallback attempts are excluded from this display.\n\n"
        "individual_cycle_schedules: Executed weights and residual histories from the supplied recorded examples; Default traces are supplied evidence, not recomputed solves.\n"
    )
    if config.get("evaluation"):
        captions = captions.replace("Random exploration is included", "Epsilon exploration is disabled; learner parameters are frozen")
        captions = captions.replace("from input one", "from held-out input 5,001")
    (output / "captions.txt").write_text(captions)
    detail = {"run": str(root), "seeds": seeds, "methods": list(METHODS),
              "illustrative_seed": seed, "selection_rule": selection,
              "full_stream_total_reduction_pct": dict(zip(seeds, total_reductions.tolist())),
              "full_stream_native_reduction_pct": dict(zip(seeds, reductions.tolist())),
              "timing_fields": list(FIELDS),
              "checks": {"passed": True, "paired_inputs": True, "native_accounting": True,
                         "learner_accounting": True, "matches_run_summary": True,
                         "action_grid_and_activation": True, "post_termination_blanks": True},
              "source_sha256": sources,
              "reproduce": f"python -m experiments.paper_final.plot_online {root}",
              "reference_style": "Local manuscript figure layouts: runtime_breakdown, cumulative_times, learned_cycle_weights, individual_cycle_schedules"}
    if Path(__file__).resolve() != (output / "plot_online.py").resolve():
        shutil.copyfile(__file__, output / "plot_online.py")
    detail["output_sha256"] = {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "manifest.json"}
    (output / "manifest.json").write_text(json.dumps(detail, indent=2, allow_nan=False) + "\n")
    info = {"output": str(output), "illustrative_seed": seed,
                      "selected_native_reductions_pct": dict(zip(LABELS, reductions[selected].tolist())),
                      "figures": [name for name, _ in figures], "checks_passed": True}
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--illustrative-seed", help="Override the paper's maximum joint-reduction selection rule")
    parser.add_argument("--cycle-examples", type=Path, help="Recorded examples for the individual-cycle paper figure")
    args = parser.parse_args()
    print(json.dumps(generate(args.run, output=args.output, illustrative_seed=args.illustrative_seed, cycle_examples=args.cycle_examples), indent=2))


if __name__ == "__main__":
    main()
