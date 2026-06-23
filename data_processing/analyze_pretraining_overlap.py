from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Iterable

import pandas as pd
from datasets import get_dataset_split_names, load_dataset, load_dataset_builder
from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from tqdm import tqdm


RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FINETUNING_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "figure_generation" / "results" / "pretraining_overlap"
DEFAULT_PRETRAINING_DATASET = "aaronfeller/peptideclm-2-pretraining-data"
DEFAULT_FILE_NAMES = [
    "PepMSND_clustered_data.csv",
    "amp_train.csv",
    "amp_val.csv",
    "amp_test.csv",
    "CellPPD_train.csv",
    "CellPPD_test.csv",
    "PAMPA_clusters.csv",
    "THPep_main90_smiles_classes.csv",
]
SMILES_COLUMN_CANDIDATES = ["smiles", "SMILES", "canonical_smiles"]


@dataclass(frozen=True)
class FinetuningDataset:
    dataset_name: str
    file_path: Path
    smiles_column: str
    frame: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare finetuning SMILES overlap against the Hugging Face pretraining dataset "
            "using remote streaming, local Arrow scanning, or a full in-memory exact-match index."
        )
    )
    parser.add_argument(
        "--finetuning-root",
        type=Path,
        default=DEFAULT_FINETUNING_ROOT,
        help="Directory containing the finetuning CSV files.",
    )
    parser.add_argument(
        "--finetuning-files",
        nargs="*",
        default=DEFAULT_FILE_NAMES,
        help="Relative CSV paths under --finetuning-root to include in the overlap analysis.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where summary and per-dataset overlap CSVs will be written.",
    )
    parser.add_argument(
        "--pretraining-dataset",
        default=DEFAULT_PRETRAINING_DATASET,
        help="Hugging Face dataset name or path for the pretraining corpus.",
    )
    parser.add_argument(
        "--pretraining-split",
        default=None,
        help="Dataset split to scan. Defaults to train when present, otherwise the first available split.",
    )
    parser.add_argument(
        "--pretraining-smiles-column",
        default=None,
        help="Override the pretraining SMILES column name if auto-detection is not sufficient.",
    )
    parser.add_argument(
        "--pretraining-access-mode",
        choices=["remote_stream", "local_arrow", "in_memory_set"],
        default="in_memory_set",
        help=(
            "How to access the pretraining dataset: remote_stream keeps the old row-by-row Hugging Face stream; "
            "local_arrow downloads once into the Hugging Face cache and scans locally; "
            "in_memory_set downloads once and builds an in-memory hash set for fastest exact matching."
        ),
    )
    parser.add_argument(
        "--pretraining-batch-size",
        type=int,
        default=100000,
        help="Batch size for local Arrow and in-memory pretraining scans.",
    )
    parser.add_argument(
        "--skip-finetuning-canonicalization",
        action="store_true",
        help=(
            "Treat finetuning SMILES as already canonicalized and skip RDKit canonicalization. "
            "Values are still stripped and empty cells are treated as missing."
        ),
    )
    parser.add_argument(
        "--canonicalize-pretraining",
        action="store_true",
        help="Canonicalize pretraining SMILES with RDKit before matching. Off by default because the source data is already canonicalized.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100000,
        help="Print streaming progress every N pretraining examples.",
    )
    return parser.parse_args()


def canonicalize_smiles(smiles: object) -> str | None:
    if pd.isna(smiles):
        return None
    smiles_str = str(smiles).strip()
    if not smiles_str:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles_str)
    except Exception:
        return None
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


def normalize_existing_smiles(smiles: object) -> str | None:
    if pd.isna(smiles):
        return None
    smiles_str = str(smiles).strip()
    return smiles_str or None


def detect_smiles_column(columns: Iterable[str], candidates: Iterable[str] = SMILES_COLUMN_CANDIDATES) -> str:
    columns_list = list(columns)
    lowered = {column.lower(): column for column in columns_list}
    for candidate in candidates:
        if candidate in columns_list:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise ValueError(f"Could not detect a SMILES column from columns: {columns_list}")


def dataset_group_name(file_path: Path) -> str:
    stem = file_path.stem
    lower_stem = stem.lower()
    if lower_stem.startswith("amp_"):
        return "amp"
    if lower_stem.startswith("cellppd_"):
        return "CellPPD"
    return stem


def load_finetuning_dataset(file_path: Path) -> FinetuningDataset:
    frame = pd.read_csv(file_path).copy()
    smiles_column = detect_smiles_column(frame.columns)
    dataset_name = file_path.stem
    return FinetuningDataset(
        dataset_name=dataset_name,
        file_path=file_path,
        smiles_column=smiles_column,
        frame=frame,
    )


def prepare_finetuning_frame(
    dataset: FinetuningDataset,
    skip_finetuning_canonicalization: bool,
) -> pd.DataFrame:
    frame = dataset.frame.copy()
    print(f"Preparing finetuning dataset: {dataset.dataset_name} ({len(frame):,} rows)")
    frame["source_dataset"] = dataset.dataset_name
    frame["source_group"] = dataset_group_name(dataset.file_path)
    frame["source_file"] = dataset.file_path.name
    frame["source_smiles_column"] = dataset.smiles_column
    frame["original_smiles"] = frame[dataset.smiles_column]
    transform = normalize_existing_smiles if skip_finetuning_canonicalization else canonicalize_smiles
    transform_label = "Normalizing" if skip_finetuning_canonicalization else "Canonicalizing"
    frame["canonical_smiles"] = [
        transform(smiles)
        for smiles in tqdm(
            frame[dataset.smiles_column],
            desc=f"{transform_label} {dataset.dataset_name}",
            unit="smiles",
            leave=False,
        )
    ]
    frame["has_valid_canonical_smiles"] = frame["canonical_smiles"].notna()
    valid_count = int(frame["has_valid_canonical_smiles"].sum())
    print(
        f"Finished {dataset.dataset_name}: {valid_count:,}/{len(frame):,} rows produced canonical SMILES"
    )
    return frame


def choose_pretraining_split(dataset_name: str, requested_split: str | None) -> str:
    if requested_split:
        return requested_split
    split_names = list(get_dataset_split_names(dataset_name))
    if not split_names:
        raise ValueError(f"No dataset splits found for {dataset_name}.")
    if "train" in split_names:
        return "train"
    return split_names[0]


def detect_pretraining_smiles_column(first_example: dict[str, object], requested_column: str | None) -> str:
    if requested_column:
        if requested_column not in first_example:
            raise ValueError(
                f"Requested pretraining SMILES column '{requested_column}' was not found. "
                f"Available columns: {sorted(first_example.keys())}"
            )
        return requested_column
    return detect_smiles_column(first_example.keys())


def detect_pretraining_smiles_column_from_columns(
    column_names: Iterable[str], requested_column: str | None
) -> str:
    columns = list(column_names)
    if requested_column:
        if requested_column not in columns:
            raise ValueError(
                f"Requested pretraining SMILES column '{requested_column}' was not found. "
                f"Available columns: {sorted(columns)}"
            )
        return requested_column
    return detect_smiles_column(columns)


def normalize_smiles_value(value: object, canonicalize_pretraining: bool) -> str | None:
    if canonicalize_pretraining:
        return canonicalize_smiles(value)
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def load_local_pretraining_dataset(
    dataset_name: str,
    split_name: str,
    requested_smiles_column: str | None,
):
    print(f"Loading pretraining dataset locally into Hugging Face cache: {dataset_name} [{split_name}]")
    dataset = load_dataset(dataset_name, split=split_name)
    smiles_column = detect_pretraining_smiles_column_from_columns(
        dataset.column_names,
        requested_smiles_column,
    )
    dataset = dataset.select_columns([smiles_column])
    print(
        f"Loaded local pretraining dataset with {dataset.num_rows:,} rows using SMILES column: {smiles_column}"
    )
    return dataset, smiles_column


def get_remote_parquet_paths(dataset_name: str, split_name: str) -> list[str]:
    fs = HfFileSystem()
    parquet_paths = sorted(fs.glob(f"datasets/{dataset_name}/{split_name}/**/*.parquet"))
    if not parquet_paths:
        raise ValueError(
            f"No parquet files found for dataset '{dataset_name}' split '{split_name}'."
        )
    return parquet_paths


def inspect_remote_parquet_split(
    dataset_name: str,
    split_name: str,
    requested_smiles_column: str | None,
) -> tuple[list[str], str, int]:
    parquet_paths = get_remote_parquet_paths(dataset_name, split_name)
    fs = HfFileSystem()

    total_rows = 0
    smiles_column: str | None = None
    for path in parquet_paths:
        with fs.open(path, "rb") as handle:
            parquet_file = pq.ParquetFile(handle)
            total_rows += parquet_file.metadata.num_rows
            if smiles_column is None:
                smiles_column = detect_pretraining_smiles_column_from_columns(
                    parquet_file.schema.names,
                    requested_smiles_column,
                )

    if smiles_column is None:
        raise ValueError(
            f"Could not determine the pretraining SMILES column for dataset '{dataset_name}' split '{split_name}'."
        )

    print(
        f"Using remote parquet scan for {dataset_name} [{split_name}] with {len(parquet_paths)} shard(s) and SMILES column: {smiles_column}"
    )
    return parquet_paths, smiles_column, total_rows


def scan_local_pretraining_matches(
    dataset_name: str,
    split_name: str,
    target_smiles: set[str],
    requested_smiles_column: str | None,
    canonicalize_pretraining: bool,
    progress_every: int,
    batch_size: int,
) -> tuple[set[str], dict[str, object]]:
    dataset, smiles_column = load_local_pretraining_dataset(
        dataset_name=dataset_name,
        split_name=split_name,
        requested_smiles_column=requested_smiles_column,
    )
    iterable = dataset.to_iterable_dataset(num_shards=256)
    found_smiles: set[str] = set()
    remaining_targets = set(target_smiles)
    examples_seen = 0

    print(
        f"Scanning cached local Arrow rows against {len(target_smiles):,} unique finetuning SMILES"
    )
    progress_bar = tqdm(
        total=dataset.num_rows,
        desc="Scanning local pretraining",
        unit="rows",
        smoothing=0.05,
    )
    try:
        for batch in iterable.iter(batch_size=batch_size):
            smiles_batch = batch[smiles_column]
            batch_matches = 0
            for value in smiles_batch:
                examples_seen += 1
                normalized = normalize_smiles_value(value, canonicalize_pretraining)
                if normalized and normalized in remaining_targets:
                    found_smiles.add(normalized)
                    remaining_targets.remove(normalized)
                    batch_matches += 1
                    if not remaining_targets:
                        break

            progress_bar.update(len(smiles_batch))
            if batch_matches:
                progress_bar.set_postfix(
                    matched=len(found_smiles),
                    remaining=len(remaining_targets),
                    refresh=False,
                )
            if progress_every > 0 and examples_seen % progress_every < len(smiles_batch):
                print(
                    f"Scanned {examples_seen:,} cached local rows; "
                    f"matched {len(found_smiles):,} of {len(target_smiles):,} unique finetuning SMILES"
                )
            if not remaining_targets:
                break
    finally:
        progress_bar.close()

    print(
        f"Finished cached local scan after {examples_seen:,} rows; matched {len(found_smiles):,} unique SMILES"
    )
    metadata = {
        "pretraining_dataset": dataset_name,
        "pretraining_split": split_name,
        "pretraining_smiles_column": smiles_column,
        "pretraining_access_mode": "local_arrow",
        "canonicalize_pretraining": canonicalize_pretraining,
        "pretraining_examples_scanned": examples_seen,
        "target_unique_smiles": len(target_smiles),
        "matched_unique_smiles": len(found_smiles),
        "unmatched_unique_smiles": len(remaining_targets),
        "all_targets_found": not remaining_targets,
        "pretraining_num_rows": int(dataset.num_rows),
        "pretraining_batch_size": batch_size,
    }
    return found_smiles, metadata


def build_in_memory_pretraining_index(
    dataset_name: str,
    split_name: str,
    requested_smiles_column: str | None,
    canonicalize_pretraining: bool,
    progress_every: int,
    batch_size: int,
) -> tuple[set[str], dict[str, object]]:
    parquet_paths, smiles_column, total_rows = inspect_remote_parquet_split(
        dataset_name=dataset_name,
        split_name=split_name,
        requested_smiles_column=requested_smiles_column,
    )
    fs = HfFileSystem()
    pretraining_smiles: set[str] = set()
    examples_seen = 0

    print("Building in-memory exact-match hash set for pretraining SMILES from remote parquet shards")
    progress_bar = tqdm(
        total=total_rows,
        desc="Indexing pretraining",
        unit="rows",
        smoothing=0.05,
    )
    try:
        for path in parquet_paths:
            print(f"Reading parquet shard: {path}")
            with fs.open(path, "rb") as handle:
                parquet_file = pq.ParquetFile(handle)
                for row_group_index in range(parquet_file.num_row_groups):
                    table = parquet_file.read_row_group(
                        row_group_index,
                        columns=[smiles_column],
                    )
                    smiles_batch = table.column(0).to_pylist()
                    for value in smiles_batch:
                        examples_seen += 1
                        normalized = normalize_smiles_value(value, canonicalize_pretraining)
                        if normalized:
                            pretraining_smiles.add(normalized)

                    progress_bar.update(len(smiles_batch))
                    progress_bar.set_postfix(unique=len(pretraining_smiles), refresh=False)
                    if progress_every > 0 and examples_seen % progress_every < len(smiles_batch):
                        print(
                            f"Indexed {examples_seen:,} pretraining rows; "
                            f"current unique SMILES in memory: {len(pretraining_smiles):,}"
                        )
    finally:
        progress_bar.close()

    print(
        f"Finished building in-memory pretraining index from {examples_seen:,} rows; "
        f"loaded {len(pretraining_smiles):,} unique SMILES"
    )
    metadata = {
        "pretraining_dataset": dataset_name,
        "pretraining_split": split_name,
        "pretraining_smiles_column": smiles_column,
        "pretraining_access_mode": "in_memory_set",
        "canonicalize_pretraining": canonicalize_pretraining,
        "pretraining_examples_scanned": examples_seen,
        "pretraining_num_rows": int(total_rows),
        "pretraining_unique_smiles_loaded": len(pretraining_smiles),
        "pretraining_batch_size": batch_size,
        "pretraining_parquet_shards": parquet_paths,
    }
    return pretraining_smiles, metadata


def stream_pretraining_matches(
    dataset_name: str,
    split_name: str,
    target_smiles: set[str],
    requested_smiles_column: str | None,
    canonicalize_pretraining: bool,
    progress_every: int,
) -> tuple[set[str], dict[str, object]]:
    print(f"Opening pretraining dataset: {dataset_name} [{split_name}]")
    stream = load_dataset(dataset_name, split=split_name, streaming=True)
    iterator = iter(stream)
    try:
        first_example = next(iterator)
    except StopIteration as exc:
        raise ValueError(f"Pretraining dataset split '{split_name}' is empty.") from exc

    smiles_column = detect_pretraining_smiles_column(first_example, requested_smiles_column)
    print(f"Using pretraining SMILES column: {smiles_column}")
    found_smiles: set[str] = set()
    remaining_targets = set(target_smiles)
    examples_seen = 0

    total_examples: int | None = None
    try:
        builder = load_dataset_builder(dataset_name)
        split_info = builder.info.splits.get(split_name) if builder.info.splits else None
        if split_info is not None and split_info.num_examples is not None:
            total_examples = int(split_info.num_examples)
    except Exception:
        total_examples = None

    print(
        f"Streaming pretraining rows to match against {len(target_smiles):,} unique finetuning SMILES"
    )
    progress_bar = tqdm(
        total=total_examples,
        desc="Scanning pretraining",
        unit="rows",
        smoothing=0.05,
    )
    try:
        for example in chain([first_example], iterator):
            examples_seen += 1
            normalized = normalize_smiles_value(example.get(smiles_column), canonicalize_pretraining)
            if normalized and normalized in remaining_targets:
                found_smiles.add(normalized)
                remaining_targets.remove(normalized)
                progress_bar.set_postfix(
                    matched=len(found_smiles),
                    remaining=len(remaining_targets),
                    refresh=False,
                )
                if not remaining_targets:
                    progress_bar.update(1)
                    break

            if progress_every > 0 and examples_seen % progress_every == 0:
                print(
                    f"Scanned {examples_seen:,} pretraining rows; "
                    f"matched {len(found_smiles):,} of {len(target_smiles):,} unique finetuning SMILES"
                )
            progress_bar.update(1)
    finally:
        progress_bar.close()

    print(
        f"Finished pretraining scan after {examples_seen:,} rows; "
        f"matched {len(found_smiles):,} unique SMILES"
    )

    metadata = {
        "pretraining_dataset": dataset_name,
        "pretraining_split": split_name,
        "pretraining_smiles_column": smiles_column,
        "canonicalize_pretraining": canonicalize_pretraining,
        "pretraining_examples_scanned": examples_seen,
        "target_unique_smiles": len(target_smiles),
        "matched_unique_smiles": len(found_smiles),
        "unmatched_unique_smiles": len(remaining_targets),
        "all_targets_found": not remaining_targets,
    }
    return found_smiles, metadata


def resolve_pretraining_matches(
    dataset_name: str,
    split_name: str,
    target_smiles: set[str],
    requested_smiles_column: str | None,
    canonicalize_pretraining: bool,
    progress_every: int,
    access_mode: str,
    batch_size: int,
) -> tuple[set[str], dict[str, object]]:
    if access_mode == "remote_stream":
        found_smiles, metadata = stream_pretraining_matches(
            dataset_name=dataset_name,
            split_name=split_name,
            target_smiles=target_smiles,
            requested_smiles_column=requested_smiles_column,
            canonicalize_pretraining=canonicalize_pretraining,
            progress_every=progress_every,
        )
        metadata["pretraining_access_mode"] = access_mode
        return found_smiles, metadata

    if access_mode == "local_arrow":
        return scan_local_pretraining_matches(
            dataset_name=dataset_name,
            split_name=split_name,
            target_smiles=target_smiles,
            requested_smiles_column=requested_smiles_column,
            canonicalize_pretraining=canonicalize_pretraining,
            progress_every=progress_every,
            batch_size=batch_size,
        )

    if access_mode == "in_memory_set":
        pretraining_smiles, metadata = build_in_memory_pretraining_index(
            dataset_name=dataset_name,
            split_name=split_name,
            requested_smiles_column=requested_smiles_column,
            canonicalize_pretraining=canonicalize_pretraining,
            progress_every=progress_every,
            batch_size=batch_size,
        )
        found_smiles = target_smiles.intersection(pretraining_smiles)
        metadata.update(
            {
                "target_unique_smiles": len(target_smiles),
                "matched_unique_smiles": len(found_smiles),
                "unmatched_unique_smiles": len(target_smiles) - len(found_smiles),
                "all_targets_found": len(found_smiles) == len(target_smiles),
            }
        )
        print(
            f"In-memory set intersection matched {len(found_smiles):,} of {len(target_smiles):,} unique finetuning SMILES"
        )
        return found_smiles, metadata

    raise ValueError(f"Unsupported pretraining access mode: {access_mode}")


def build_file_summary(dataset_frame: pd.DataFrame, found_smiles: set[str]) -> dict[str, object]:
    valid_rows = dataset_frame.loc[dataset_frame["has_valid_canonical_smiles"]].copy()
    overlapping_rows = valid_rows["canonical_smiles"].isin(found_smiles)
    unique_smiles = set(valid_rows["canonical_smiles"].dropna())
    unique_overlap = unique_smiles.intersection(found_smiles)

    valid_row_count = int(len(valid_rows))
    overlapping_row_count = int(overlapping_rows.sum())
    unique_count = int(len(unique_smiles))
    unique_overlap_count = int(len(unique_overlap))

    return {
        "dataset_name": dataset_frame["source_dataset"].iloc[0],
        "dataset_group": dataset_frame["source_group"].iloc[0],
        "source_file": dataset_frame["source_file"].iloc[0],
        "source_smiles_column": dataset_frame["source_smiles_column"].iloc[0],
        "total_rows": int(len(dataset_frame)),
        "valid_canonical_rows": valid_row_count,
        "invalid_or_missing_smiles_rows": int(len(dataset_frame) - valid_row_count),
        "overlapping_rows": overlapping_row_count,
        "row_overlap_fraction": overlapping_row_count / valid_row_count if valid_row_count else None,
        "row_overlap_percent": 100.0 * overlapping_row_count / valid_row_count if valid_row_count else None,
        "unique_canonical_smiles": unique_count,
        "overlapping_unique_smiles": unique_overlap_count,
        "unique_overlap_fraction": unique_overlap_count / unique_count if unique_count else None,
        "unique_overlap_percent": 100.0 * unique_overlap_count / unique_count if unique_count else None,
    }


def build_group_summary(all_rows: pd.DataFrame, found_smiles: set[str]) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    for group_name, group_frame in all_rows.groupby("source_group", sort=True):
        valid_rows = group_frame.loc[group_frame["has_valid_canonical_smiles"]].copy()
        overlapping_rows = valid_rows["canonical_smiles"].isin(found_smiles)
        unique_smiles = set(valid_rows["canonical_smiles"].dropna())
        unique_overlap = unique_smiles.intersection(found_smiles)
        valid_row_count = int(len(valid_rows))
        overlapping_row_count = int(overlapping_rows.sum())
        unique_count = int(len(unique_smiles))
        unique_overlap_count = int(len(unique_overlap))
        summary_rows.append(
            {
                "dataset_group": group_name,
                "source_files": ";".join(sorted(group_frame["source_file"].unique())),
                "total_rows": int(len(group_frame)),
                "valid_canonical_rows": valid_row_count,
                "invalid_or_missing_smiles_rows": int(len(group_frame) - valid_row_count),
                "overlapping_rows": overlapping_row_count,
                "row_overlap_fraction": overlapping_row_count / valid_row_count if valid_row_count else None,
                "row_overlap_percent": 100.0 * overlapping_row_count / valid_row_count if valid_row_count else None,
                "unique_canonical_smiles": unique_count,
                "overlapping_unique_smiles": unique_overlap_count,
                "unique_overlap_fraction": unique_overlap_count / unique_count if unique_count else None,
                "unique_overlap_percent": 100.0 * unique_overlap_count / unique_count if unique_count else None,
            }
        )
    return pd.DataFrame(summary_rows).sort_values("dataset_group").reset_index(drop=True)


def write_dataset_details(dataset_frame: pd.DataFrame, found_smiles: set[str], output_dir: Path) -> None:
    annotated = dataset_frame.copy()
    annotated["in_pretraining"] = annotated["canonical_smiles"].isin(found_smiles)
    dataset_name = annotated["source_dataset"].iloc[0]
    output_path = output_dir / f"{dataset_name}_overlap_details.csv"
    annotated.to_csv(output_path, index=False)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    finetuning_paths = [args.finetuning_root / relative_path for relative_path in args.finetuning_files]
    missing_paths = [path for path in finetuning_paths if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing finetuning input files: {missing}")

    print(f"Loading {len(finetuning_paths)} finetuning files")

    finetuning_datasets = [load_finetuning_dataset(path) for path in finetuning_paths]
    prepared_frames = [
        prepare_finetuning_frame(
            dataset,
            skip_finetuning_canonicalization=args.skip_finetuning_canonicalization,
        )
        for dataset in finetuning_datasets
    ]
    all_rows = pd.concat(prepared_frames, ignore_index=True)
    target_smiles = set(all_rows.loc[all_rows["has_valid_canonical_smiles"], "canonical_smiles"])
    print(
        f"Prepared {len(all_rows):,} finetuning rows spanning {len(target_smiles):,} unique canonical SMILES"
    )

    split_name = choose_pretraining_split(args.pretraining_dataset, args.pretraining_split)
    found_smiles, metadata = resolve_pretraining_matches(
        dataset_name=args.pretraining_dataset,
        split_name=split_name,
        target_smiles=target_smiles,
        requested_smiles_column=args.pretraining_smiles_column,
        canonicalize_pretraining=args.canonicalize_pretraining,
        progress_every=args.progress_every,
        access_mode=args.pretraining_access_mode,
        batch_size=args.pretraining_batch_size,
    )

    file_summaries = [build_file_summary(frame, found_smiles) for frame in prepared_frames]
    file_summary_df = pd.DataFrame(file_summaries).sort_values("dataset_name").reset_index(drop=True)
    group_summary_df = build_group_summary(all_rows, found_smiles)

    print("Writing overlap summary tables")
    file_summary_df.to_csv(output_dir / "file_overlap_summary.csv", index=False)
    group_summary_df.to_csv(output_dir / "group_overlap_summary.csv", index=False)

    for frame in prepared_frames:
        write_dataset_details(frame, found_smiles, output_dir)
    print(f"Wrote {len(prepared_frames)} per-dataset overlap detail files")

    metadata["finetuning_files"] = [str(path.resolve()) for path in finetuning_paths]
    metadata["output_dir"] = str(output_dir)
    metadata["file_summary_path"] = str((output_dir / "file_overlap_summary.csv").resolve())
    metadata["group_summary_path"] = str((output_dir / "group_overlap_summary.csv").resolve())
    metadata["details_files"] = [
        str((output_dir / f"{frame['source_dataset'].iloc[0]}_overlap_details.csv").resolve())
        for frame in prepared_frames
    ]
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Prepared {len(file_summary_df)} file-level summaries.")
    print(f"Matched {metadata['matched_unique_smiles']:,} of {metadata['target_unique_smiles']:,} unique finetuning SMILES.")
    print(f"Wrote outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())