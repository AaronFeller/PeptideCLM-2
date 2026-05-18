from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from training.experiment.manifest import REPO_ROOT, load_manifest
from training.experiment.output_schema import build_run_layout


TASK_TO_DATASET = {
    "amp_hgt": "AmpHGT",
    "cellppd": "CellPPD",
    "thpep": "THPep",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalized experiment runner for benchmark tasks.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--model_family", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "tmp" / "runs")
    parser.add_argument("--log_root", type=Path, default=REPO_ROOT / "tmp" / "logs")
    parser.add_argument("--prepared_data_root", type=Path, default=REPO_ROOT / "tmp" / "prepared_data")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def resolve_command(args: argparse.Namespace, layout) -> list[str]:
    model_variant = args.model_name.split("/")[-1].replace(".", "_")
    if args.model_family == "peptideclm":
        if args.task in TASK_TO_DATASET:
            data_dir = REPO_ROOT / "data"
            if args.task == "thpep":
                data_dir = args.prepared_data_root / "thpep" / f"seed_{args.seed}"
            return [
                "python",
                "training/classification_finetuning.py",
                "--dataset",
                TASK_TO_DATASET[args.task],
                "--gpu",
                str(args.gpu_index),
                "--gpu_index",
                str(args.gpu_index),
                "--model_name",
                args.model_name,
                "--save_path",
                str(layout.run_dir),
                "--output_dir",
                str(layout.run_dir),
                "--data_dir",
                str(data_dir),
                "--log_dir",
                str(layout.log_dir),
                "--seed",
                str(args.seed),
            ]
        if args.task == "cycpeptmpdb_perm":
            prepared_csv = args.prepared_data_root / "cycpeptmpdb_perm" / "perm_external.csv"
            return [
                "python",
                "training/regression_finetune_ensemble.py",
                "--data_csv",
                str(prepared_csv),
                "--model",
                args.model_name,
                "--save_dir",
                str(layout.run_dir),
                "--save_name",
                "predictions",
                "--seed",
                str(args.seed),
            ]

    if args.model_family == "pepmsnd_kan":
        if args.fold is None:
            raise ValueError("--fold is required for pepmsnd_kan runs.")
        data_dir = args.prepared_data_root / "pepmsnd"
        return [
            "python",
            "training/PepMSND_analysis/train_pepmsnd_kan_paperstyle.py",
            "--model_name",
            args.model_name,
            "--fold",
            str(args.fold),
            "--data_dir",
            str(data_dir),
            "--save_path",
            str(layout.run_dir),
            "--seed",
            str(args.seed),
        ]

    if args.model_family == "pepmsnd_species_env":
        if args.fold is None:
            raise ValueError("--fold is required for pepmsnd_species_env runs.")
        data_dir = args.prepared_data_root / "pepmsnd"
        return [
            "python",
            "training/PepMSND_analysis/train_pepmsnd_species_env.py",
            "--model_name",
            args.model_name,
            "--fold",
            str(args.fold),
            "--data_dir",
            str(data_dir),
            "--save_path",
            str(layout.run_dir),
            "--seed",
            str(args.seed),
        ]

    if args.model_family == "xgboost_rdkit":
        command = ["python", "training/adapters/xgboost_baseline.py", "--feature_set", "rdkit", "--task", args.task, "--seed", str(args.seed), "--output_dir", str(layout.run_dir), "--prepared_data_root", str(args.prepared_data_root)]
        if args.dry_run or not args.execute:
            command.append("--dry_run")
        return command
    if args.model_family == "xgboost_morgan":
        command = ["python", "training/adapters/xgboost_baseline.py", "--feature_set", "morgan", "--task", args.task, "--seed", str(args.seed), "--output_dir", str(layout.run_dir), "--prepared_data_root", str(args.prepared_data_root)]
        if args.dry_run or not args.execute:
            command.append("--dry_run")
        return command
    if args.model_family == "chemberta77m":
        command = ["python", "training/adapters/chemberta_baseline.py", "--task", args.task, "--model_name", args.model_name, "--seed", str(args.seed), "--gpu_index", str(args.gpu_index), "--output_dir", str(layout.run_dir), "--prepared_data_root", str(args.prepared_data_root)]
        if args.dry_run or not args.execute:
            command.append("--dry_run")
        return command
    if args.model_family == "chemeleon":
        command = ["python", "training/adapters/chemeleon_baseline.py", "--task", args.task, "--model_name", args.model_name, "--seed", str(args.seed), "--gpu_index", str(args.gpu_index), "--output_dir", str(layout.run_dir), "--prepared_data_root", str(args.prepared_data_root)]
        if args.dry_run or not args.execute:
            command.append("--dry_run")
        return command

    raise ValueError(f"Unsupported model_family: {args.model_family}")


def main() -> int:
    args = parse_args()
    manifest = load_manifest()
    valid_tasks = {task["task_id"] for task in manifest["tasks"]}
    if args.task not in valid_tasks:
        raise ValueError(f"Unknown task: {args.task}")

    model_variant = args.model_name.split("/")[-1].replace(".", "_")
    suffix = f"fold{args.fold}" if args.fold is not None else None
    layout = build_run_layout(
        task_id=args.task,
        model_family=args.model_family,
        model_variant=model_variant,
        seed=args.seed,
        run_root=args.output_root,
        log_root=args.log_root,
        suffix=suffix,
    )
    command = resolve_command(args, layout)

    layout.run_dir.mkdir(parents=True, exist_ok=True)
    layout.log_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "task": args.task,
        "model_family": args.model_family,
        "model_name": args.model_name,
        "seed": args.seed,
        "gpu_index": args.gpu_index,
        "fold": args.fold,
        "command": command,
    }
    layout.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    shell_line = shlex.join(command)
    print(shell_line)
    if args.dry_run or not args.execute:
        return 0

    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())