#!/usr/bin/env python3

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path


ACCURACY_PATTERN = re.compile(
    r"Accuracy of the network on the \d+ test images: ([0-9]+(?:\.[0-9]+)?)%"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run MAE fine-tuning on each fold dataset and report per-fold and mean test accuracy. "
            "Any additional unknown arguments are passed through to main_finetune.py."
        )
    )
    parser.add_argument(
        "--folds_root",
        required=True,
        type=Path,
        help="Root directory containing fold_0 ... fold_4 datasets.",
    )
    parser.add_argument(
        "--pretrained_checkpoint",
        required=True,
        type=Path,
        help="Pretrained MAE checkpoint passed to main_finetune.py --finetune.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        type=Path,
        help="Directory where per-fold fine-tuning outputs and the summary will be stored.",
    )
    parser.add_argument(
        "--python_exe",
        default=sys.executable,
        help="Python executable used to launch main_finetune.py. Default: current Python.",
    )
    parser.add_argument(
        "--main_finetune",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "main_finetune.py",
        help="Path to main_finetune.py. Default: repo's main_finetune.py.",
    )
    parser.add_argument("--model", default="vit_base_patch16")
    parser.add_argument("--nb_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--blr", type=float, default=5e-4)
    parser.add_argument("--layer_decay", type=float, default=0.65)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--drop_path", type=float, default=0.1)
    parser.add_argument("--reprob", type=float, default=0.25)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cls_token",
        action="store_true",
        help="Use --cls_token in main_finetune.py instead of the default global pooling.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip training for a fold if best_model.pth already exists in its output directory.",
    )
    parser.add_argument(
        "--fold_names",
        nargs="*",
        default=None,
        help="Optional subset of fold directories to run, for example: fold_0 fold_1.",
    )

    args, extra_finetune_args = parser.parse_known_args()
    return args, extra_finetune_args


def discover_folds(folds_root: Path, fold_names=None):
    if fold_names:
        folds = [folds_root / fold_name for fold_name in fold_names]
    else:
        folds = sorted(
            child for child in folds_root.iterdir()
            if child.is_dir() and child.name.startswith("fold_")
        )
    if not folds:
        raise FileNotFoundError(f"No fold directories found in {folds_root}")

    for fold_dir in folds:
        for split_name in ("train", "val", "test"):
            if not (fold_dir / split_name).is_dir():
                raise FileNotFoundError(f"Missing {split_name} directory in {fold_dir}")
    return folds


def run_and_tee(command, log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n$ {shlex.join(command)}")

    collected_lines = []
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
            collected_lines.append(line)
        return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)

    return "".join(collected_lines)


def build_train_command(args, extra_args, fold_dir: Path, fold_output_dir: Path):
    command = [
        args.python_exe,
        str(args.main_finetune),
        "--model", args.model,
        "--data_path", str(fold_dir),
        "--finetune", str(args.pretrained_checkpoint),
        "--nb_classes", str(args.nb_classes),
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--blr", str(args.blr),
        "--layer_decay", str(args.layer_decay),
        "--weight_decay", str(args.weight_decay),
        "--drop_path", str(args.drop_path),
        "--reprob", str(args.reprob),
        "--num_workers", str(args.num_workers),
        "--device", args.device,
        "--seed", str(args.seed),
        "--output_dir", str(fold_output_dir),
        "--log_dir", str(fold_output_dir),
    ]
    if args.cls_token:
        command.append("--cls_token")
    command.extend(extra_args)
    return command


def build_eval_command(args, extra_args, fold_dir: Path, best_model_path: Path):
    command = [
        args.python_exe,
        str(args.main_finetune),
        "--eval",
        "--model", args.model,
        "--data_path", str(fold_dir),
        "--resume", str(best_model_path),
        "--nb_classes", str(args.nb_classes),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--device", args.device,
        "--seed", str(args.seed),
    ]
    if args.cls_token:
        command.append("--cls_token")
    command.extend(extra_args)
    return command


def parse_accuracy(output_text: str) -> float:
    matches = ACCURACY_PATTERN.findall(output_text)
    if not matches:
        raise ValueError("Could not find test accuracy in evaluation output.")
    return float(matches[-1])


def main():
    args, extra_finetune_args = parse_args()
    folds_root = args.folds_root.resolve()
    pretrained_checkpoint = args.pretrained_checkpoint.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    folds = discover_folds(folds_root, args.fold_names)
    results = []

    for fold_dir in folds:
        fold_output_dir = output_root / fold_dir.name
        best_model_path = fold_output_dir / "best_model.pth"

        if args.skip_existing and best_model_path.is_file():
            print(f"\nSkipping training for {fold_dir.name} because {best_model_path} already exists.")
        else:
            train_output = run_and_tee(
                build_train_command(args, extra_finetune_args, fold_dir, fold_output_dir),
                fold_output_dir / "train.out",
            )
            if "Traceback" in train_output and not best_model_path.is_file():
                raise RuntimeError(f"Training for {fold_dir.name} failed before producing best_model.pth")

        if not best_model_path.is_file():
            raise FileNotFoundError(
                f"Expected fine-tuned checkpoint not found: {best_model_path}"
            )

        eval_output = run_and_tee(
            build_eval_command(args, extra_finetune_args, fold_dir, best_model_path),
            fold_output_dir / "eval.out",
        )
        accuracy = parse_accuracy(eval_output)
        results.append({"fold": fold_dir.name, "accuracy": accuracy})
        print(f"\n{fold_dir.name}: test accuracy = {accuracy:.2f}%")

    accuracies = [entry["accuracy"] for entry in results]
    mean_accuracy = sum(accuracies) / len(accuracies)

    print("\nPer-fold accuracies:")
    for entry in results:
        print(f"  {entry['fold']}: {entry['accuracy']:.2f}%")
    print(f"Average accuracy: {mean_accuracy:.2f}%")

    summary_path = output_root / "crossval_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "folds_root": str(folds_root),
                "pretrained_checkpoint": str(pretrained_checkpoint),
                "output_root": str(output_root),
                "results": results,
                "mean_accuracy": mean_accuracy,
                "extra_finetune_args": extra_finetune_args,
            },
            handle,
            indent=2,
        )
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
