from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.model_selection import train_test_split

from training.experiment.manifest import REPO_ROOT


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def prepare_thpep(seed: int, output_root: Path) -> list[Path]:
    source = REPO_ROOT / "data" / "THPep_main90_smiles_classes.csv"
    output_dir = output_root / "thpep" / f"seed_{seed}"
    ensure_dir(output_dir)

    df = pd.read_csv(source)
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["class"],
        random_state=seed * 45671,
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        stratify=train_df["class"],
        random_state=seed * 52984,
    )

    prepared = []
    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        export_df = split_df.rename(columns={"class": "label"}).copy()
        export_path = output_dir / f"THPep_{split_name}.csv"
        export_df.to_csv(export_path, index=False)
        prepared.append(export_path)
    return prepared


def prepare_perm(seed: int, output_root: Path) -> list[Path]:
    source = REPO_ROOT / "data" / "PAMPA_clusters.csv"
    output_dir = output_root / "cycpeptmpdb_perm"
    ensure_dir(output_dir)

    df = pd.read_csv(source)
    unique_clusters = sorted(df["cluster"].dropna().unique().tolist())
    fold_frames = []
    for fold, cluster_id in enumerate(unique_clusters):
        fold_frame = df[df["cluster"] == cluster_id].copy()
        fold_frame["fold"] = fold
        fold_frames.append(fold_frame)

    prepared_df = pd.concat(fold_frames, ignore_index=True)
    prepared_df = prepared_df.rename(columns={"PAMPA": "value"})
    prepared_df = prepared_df[["SMILES", "value", "fold", "cluster"]]

    export_path = output_dir / "perm_external.csv"
    prepared_df.to_csv(export_path, index=False)
    return [export_path]


def prepare_pepmsnd(seed: int, output_root: Path) -> list[Path]:
    source = REPO_ROOT / "data" / "PepMSND_clustered_data.csv"
    output_dir = output_root / "pepmsnd"
    ensure_dir(output_dir)

    df = pd.read_csv(source)
    unique_clusters = sorted(df["cluster"].dropna().unique().tolist())
    
    # Map unique cluster identifiers to 0-indexed fold integers
    cluster_to_fold = {cluster_id: fold for fold, cluster_id in enumerate(unique_clusters)}
    df["fold"] = df["cluster"].map(cluster_to_fold)

    export_path = output_dir / "pepmsnd_external.csv"
    df.to_csv(export_path, index=False)
    return [export_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare benchmark inputs for the retraining workflow.")
    parser.add_argument(
        "--task",
        choices=["all", "thpep", "cycpeptmpdb_perm", "pepmsnd"],
        default="all",
        help="Task to prepare. 'all' prepares every task that needs derived inputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed used for THPep split generation. Cluster-based tasks ignore this and use fixed cluster holdouts.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "tmp" / "prepared_data",
        help="Root directory for prepared task data.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print the output paths without writing files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = [args.task] if args.task != "all" else ["thpep", "cycpeptmpdb_perm", "pepmsnd"]
    actions = {
        "thpep": prepare_thpep,
        "cycpeptmpdb_perm": prepare_perm,
        "pepmsnd": prepare_pepmsnd,
    }

    for task in tasks:
        target_root = args.output_root
        if args.dry_run:
            if task == "thpep":
                print(f"PREPARE\t{task}\t{target_root / task / f'seed_{args.seed}'}")
            else:
                print(f"PREPARE\t{task}\t{target_root / task}")
            continue
        output_paths = actions[task](args.seed, target_root)
        for output_path in output_paths:
            print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())