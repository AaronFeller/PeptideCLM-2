from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "training" / "experiment" / "benchmark_manifest.json"
REQUIRED_TASK_KEYS = {
    "task_id",
    "display_name",
    "task_type",
    "status",
    "primary_metrics",
    "secondary_metrics",
    "sample_id_column",
    "input_column",
    "label_column",
    "split_semantics",
    "split_files",
    "target_outputs",
}


def load_manifest(path: Path | None = None) -> dict:
    manifest_path = path or MANIFEST_PATH
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_manifest(manifest: dict, repo_root: Path | None = None) -> list[str]:
    root = repo_root or REPO_ROOT
    errors: list[str] = []
    task_ids: set[str] = set()

    if "artifact_policy" not in manifest:
        errors.append("Manifest is missing artifact_policy.")

    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        errors.append("Manifest must define a non-empty tasks list.")
        return errors

    for task in tasks:
        missing = sorted(REQUIRED_TASK_KEYS - set(task))
        if missing:
            errors.append(f"Task {task.get('task_id', '<missing>')} is missing keys: {', '.join(missing)}")
            continue

        task_id = task["task_id"]
        if task_id in task_ids:
            errors.append(f"Duplicate task_id: {task_id}")
        task_ids.add(task_id)

        if task["task_type"] not in {"classification", "regression"}:
            errors.append(f"Task {task_id} has unsupported task_type {task['task_type']}")

        if task["status"] not in {"active", "deferred"}:
            errors.append(f"Task {task_id} has unsupported status {task['status']}")

        split_files = task.get("split_files", {})
        if task["status"] == "active" and not split_files:
            errors.append(f"Active task {task_id} must define split_files.")

        for split_name, relative_path in split_files.items():
            if not relative_path:
                errors.append(f"Task {task_id} has an empty path for split {split_name}.")
                continue
            if not (root / relative_path).exists():
                errors.append(f"Task {task_id} is missing file for split {split_name}: {relative_path}")

    deferred = [task for task in tasks if task["status"] == "deferred"]
    if not any(task["task_id"] == "fibrillation" for task in deferred):
        errors.append("Manifest must keep fibrillation marked as deferred.")

    return errors