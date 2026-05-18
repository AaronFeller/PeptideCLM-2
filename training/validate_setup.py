from __future__ import annotations

import argparse
import sys

if __package__ in {None, ""}:
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from training.experiment.manifest import load_manifest, validate_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the benchmark setup manifest.")
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print the task ids from the manifest after validation.",
    )
    args = parser.parse_args()

    manifest = load_manifest()
    errors = validate_manifest(manifest)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    tasks = manifest["tasks"]
    print(f"Validated manifest with {len(tasks)} tasks.")
    if args.list_tasks:
        for task in tasks:
            print(f"{task['task_id']}\t{task['status']}\t{task['task_type']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())