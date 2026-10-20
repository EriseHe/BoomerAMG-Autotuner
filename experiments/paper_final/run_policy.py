"""Module 2: freeze trained models, scan weights, and evaluate matched policies."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from experiments.paper_final.matched_policy.configuration import DEFAULTS, load_config
from experiments.runtime import configure_single_thread


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Write an editable configuration from explicit training run paths")
    init.add_argument("--config", type=Path, required=True)
    init.add_argument("--training-run", action="append", required=True, metavar="SEED=PATH")
    init.add_argument("--method", default="bandit_lstdq")
    init.add_argument("--random-seed", type=int, required=True)
    init.add_argument("--cases", type=int, default=100)
    check = commands.add_parser("validate", help="Check configuration and source paths without native solves")
    check.add_argument("--config", type=Path, required=True)
    prep = commands.add_parser("prepare", help="Copy checkpoints and freeze hierarchies without native solves")
    prep.add_argument("--config", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    for name in ("scan", "run", "analyze"):
        commands.add_parser(name).add_argument("--output", type=Path, required=True)
    worker = commands.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--phase", choices=("scan", "test"), required=True)
    worker.add_argument("--worker", type=int, required=True)
    args = parser.parse_args()
    if args.command == "init":
        sources = []
        for value in args.training_run:
            seed, path = value.split("=", 1)
            sources.append({"seed": int(seed), "run_dir": str(Path(path).expanduser().resolve()), "method": args.method})
        cfg = {**copy.deepcopy(DEFAULTS), "cases": args.cases, "training_runs": sources,
               "seeds": {name: args.random_seed + i for i, name in enumerate(("inputs", "candidates", "order", "repeats"))}}
        args.config.parent.mkdir(parents=True, exist_ok=True)
        with args.config.open("x") as handle:
            handle.write(json.dumps(cfg, indent=2) + "\n")
        print(f"Configuration written: {args.config.resolve()}")
        return
    if args.command == "validate":
        configure_single_thread()
        cfg = load_config(args.config)
        from experiments.paper_final.common.frozen_policy import build_bundle
        encoders = []
        for source in cfg["training_runs"]:
            bundle = build_bundle(source["training_config"], source["method"],
                                  Path(source["controller_checkpoint"]), beta=cfg["rl_beta"])
            encoders.append(dict(seed=source["seed"], state_encoding=bundle.encoder.encoding,
                                 state_features=bundle.encoder.feature_dim))
        print(json.dumps({"valid": True, "seed_labels": [s["seed"] for s in cfg["training_runs"]],
                          "cases": cfg["cases"], "weights": len(cfg["weights"]), "solver": cfg["solver"],
                          "problem": cfg["problem"], "controllers": encoders,
                          "native_solves_executed": 0}, indent=2))
        return
    configure_single_thread()
    if args.command == "prepare":
        from experiments.paper_final.matched_policy.planning import prepare
        prepare(args.config, args.output)
    else:
        from experiments.paper_final.matched_policy import workflow
        output = args.output.resolve()
        if args.command == "worker":
            workflow.worker(output, args.phase, args.worker)
        elif args.command == "analyze":
            workflow.analyze(output)
        else:
            workflow.run(output, scan_only=args.command == "scan")


if __name__ == "__main__":
    main()
