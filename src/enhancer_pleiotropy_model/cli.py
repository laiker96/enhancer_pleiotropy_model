"""epm: prepare data, train and fine-tune models, calibrate, predict and attribute."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

from .config import load_config

ARCHITECTURES = ("cnn", "attention", "dense", "enhancernet_cnn", "enhancernet_attention")


def _overrides(args) -> list[str]:
    values = list(args.set or [])
    for name in ("architecture", "init", "seed", "device"):
        if getattr(args, name, None) is not None:
            values.append(f"{name}={getattr(args, name)}")
    return values


def cmd_prepare_data(args) -> None:
    from .data.prepare import prepare
    prepare(load_config(args.config, _overrides(args)), only=args.only)


def cmd_train_regressor(args) -> None:
    from .training.regressor import train_regressor
    train_regressor(load_config(args.config, _overrides(args)), resume=args.resume)


def cmd_train_classifier(args) -> None:
    from .training.classifier import train_classifier
    train_classifier(load_config(args.config, _overrides(args)), resume=args.resume)


def cmd_calibrate(args) -> None:
    from .calibration import calibrate_run
    path = calibrate_run(args.run, args.data, args.device)
    print(f"wrote {path} (method comparison: {path.parent / 'calibration_report.md'})")


def cmd_attribute(args) -> None:
    from .attribution import run_attribution
    if not 0 <= args.shard < args.shards:
        raise SystemExit("--shard must be in [0, --shards)")
    run_attribution(load_config(args.config, _overrides(args)), args.shard, args.shards, args.limit)


def cmd_predict(args) -> None:
    from .predict import predict, sequences_from_bed, sequences_from_fasta
    if bool(args.fasta) == bool(args.bed):
        raise SystemExit("Give either --fasta or --bed (with --genome)")
    if args.bed and not args.genome:
        raise SystemExit("--bed needs --genome")
    names, codes = sequences_from_fasta(args.fasta) if args.fasta else sequences_from_bed(args.bed, args.genome)
    predict(args.checkpoint, names, codes, args.output, calibration=args.calibration, device_name=args.device)


def manuscript_commands(config: dict, only: str | None) -> list[list[str]]:
    commands = []
    regressor_args = ["--config", config["regressor_config"]]
    classifier_args = ["--config", config["classifier_config"]]
    if only in (None, "regressors"):
        for architecture in config["architectures"]:
            commands.append(["epm", "train-regressor", *regressor_args, "--architecture", architecture])
    if only in (None, "classifiers"):
        for architecture in config["architectures"]:
            for init in config["inits"]:
                for seed in config["seeds"]:
                    commands.append(["epm", "train-classifier", *classifier_args, "--architecture", architecture,
                                     "--init", init, "--seed", str(seed)])
    return commands


def cmd_manuscript(args) -> None:
    commands = manuscript_commands(load_config(args.config), args.only)
    if not args.run:
        print("\n".join(shlex.join(c) for c in commands))
        print(f"# {len(commands)} runs. Add --run to execute them here one after another, or submit each "
              "line as its own GPU job (see scripts/slurm_job.sh).", file=sys.stderr)
        return
    for command in commands:
        print("+", shlex.join(command), flush=True)
        subprocess.run([sys.executable, "-m", "enhancer_pleiotropy_model.cli", *command[1:]], check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="epm", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name, function, help_text, config=None):
        sub = commands.add_parser(name, help=help_text, description=help_text)
        if config:
            sub.add_argument("--config", type=Path, default=Path(config), help=f"YAML config (default: {config})")
            sub.add_argument("--set", action="append", metavar="KEY=VALUE",
                             help="override any config value, e.g. --set training.epochs=5 (repeatable)")
        sub.set_defaults(function=function)
        return sub

    sub = add("prepare-data", cmd_prepare_data, "build the regression and classification datasets",
              "configs/data.yaml")
    sub.add_argument("--only", choices=("regression", "classification"))

    sub = add("train-regressor", cmd_train_regressor, "pretrain an ATAC/H3K27ac profile regressor",
              "configs/regressor.yaml")
    sub.add_argument("--architecture", choices=ARCHITECTURES)
    sub.add_argument("--device", help="cuda, cpu or auto")
    sub.add_argument("--resume", action="store_true", help="continue from last_checkpoint.pt")

    sub = add("train-classifier", cmd_train_classifier, "train an eight-context enhancer classifier",
              "configs/classifier.yaml")
    sub.add_argument("--architecture", choices=ARCHITECTURES)
    sub.add_argument("--init", choices=("pretrained", "scratch"),
                     help="pretrained: start from the trained regressor; scratch: random initialisation")
    sub.add_argument("--seed", type=int)
    sub.add_argument("--device", help="cuda, cpu or auto")
    sub.add_argument("--resume", action="store_true", help="continue from last_checkpoint.pt")

    sub = add("calibrate", cmd_calibrate, "compare calibration methods and fit per-context calibration")
    sub.add_argument("run", type=Path, help="classifier run directory (contains best_model.pt)")
    sub.add_argument("--data", type=Path, help="classification data (default: the run's training data)")
    sub.add_argument("--device", default="auto")

    sub = add("attribute", cmd_attribute, "integrated-gradients maps for the eight calibrated probabilities",
              "configs/attribution.yaml")
    sub.add_argument("--shard", type=int, default=0, help="this job's share of the chunks (0-based)")
    sub.add_argument("--shards", type=int, default=1, help="number of parallel jobs")
    sub.add_argument("--limit", type=int, help="only the first N chunks of this shard (testing)")
    sub.add_argument("--device", help="cuda, cpu or auto")

    sub = add("predict", cmd_predict, "predict enhancer activity in the eight contexts")
    sub.add_argument("checkpoint", type=Path, help="classifier best_model.pt")
    sub.add_argument("--fasta", type=Path, help="FASTA of 2,048-bp sequences")
    sub.add_argument("--bed", type=Path, help="BED regions; 2,048-bp windows centred on each")
    sub.add_argument("--genome", type=Path, help="genome FASTA for --bed (e.g. dm6.fa)")
    sub.add_argument("--output", type=Path, required=True, help="output TSV")
    sub.add_argument("--calibration", type=Path, help="default: calibration.json next to the checkpoint")
    sub.add_argument("--device", default="auto")

    sub = add("manuscript", cmd_manuscript, "list (or run) every training run in the manuscript")
    sub.add_argument("--config", type=Path, default=Path("configs/manuscript.yaml"))
    sub.add_argument("--only", choices=("regressors", "classifiers"))
    sub.add_argument("--run", action="store_true", help="execute sequentially instead of printing")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
