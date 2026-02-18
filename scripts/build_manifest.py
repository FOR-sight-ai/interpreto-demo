import json
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
EXPLANATIONS_DIR = ROOT / "explanations"
OUTPUT_PATH = ROOT / "manifest.json"


def parse_model_id(model_id: str) -> dict:
    if model_id.startswith("clf:"):
        parts = model_id.split(":", 2)
        dataset = parts[1] if len(parts) > 1 else None
        return {"task": "classification", "dataset": dataset}
    if model_id.startswith("gen:"):
        return {"task": "generation", "dataset": None}
    return {"task": "classification", "dataset": None}


def collect_methods(path: Path) -> list[str]:
    return sorted(
        file.name
        for file in path.iterdir()
        if file.is_file() and file.suffix.lower() == ".html"
    )


def add_entry(
    entries: list[dict],
    model_id: str,
    task: str,
    type_name: str,
    scope_name: str,
    sample: Optional[str],
    methods: list[str],
) -> None:
    if not methods:
        return
    entries.append(
        {
            "model": model_id,
            "task": task,
            "type": type_name,
            "scope": scope_name,
            "sample": sample,
            "methods": methods,
        }
    )


def scan_scope_dir(
    scope_dir: Path,
    model_id: str,
    task: str,
    type_name: str,
    scope_name: str,
) -> list[dict]:
    entries: list[dict] = []

    methods = collect_methods(scope_dir)
    add_entry(entries, model_id, task, type_name, scope_name, None, methods)

    for first in sorted(path for path in scope_dir.iterdir() if path.is_dir()):
        methods_in_first = collect_methods(first)
        if methods_in_first:
            add_entry(
                entries,
                model_id,
                task,
                type_name,
                scope_name,
                first.name,
                methods_in_first,
            )

    return entries


def build_manifest() -> dict:
    manifest = {"models": {}, "explanations": []}
    if not EXPLANATIONS_DIR.exists():
        return manifest

    entries: list[dict] = []

    for model_dir in sorted(path for path in EXPLANATIONS_DIR.iterdir() if path.is_dir()):
        meta = parse_model_id(model_dir.name)
        model_entries: list[dict] = []

        for type_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            for scope_dir in sorted(
                path for path in type_dir.iterdir() if path.is_dir()
            ):
                model_entries.extend(
                    scan_scope_dir(
                        scope_dir,
                        model_dir.name,
                        meta["task"],
                        type_dir.name,
                        scope_dir.name,
                    )
                )

        if model_entries:
            manifest["models"][model_dir.name] = meta
            entries.extend(model_entries)

    manifest["explanations"] = sorted(
        entries,
        key=lambda entry: (
            entry["model"],
            entry["type"],
            entry["scope"],
            entry["sample"] or "",
        ),
    )

    return manifest


def main() -> None:
    manifest = build_manifest()
    OUTPUT_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
