import json
import sys
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
EXPLANATIONS_DIR = ROOT / "explanations"
OUTPUT_PATH = ROOT / "manifest.json"
DATA_ROOT = ROOT / "data"

sys.path.insert(0, str(Path(__file__).resolve().parent))


#: Direction and display metadata for every metric wired into the gallery.
#: Shared between ``build_manifest.py`` (writes ``metrics_meta`` at the
#: top of ``manifest.json``) and every ``*_metrics.py`` script (which
#: uses the ``direction`` value to know whether a higher score is better).
METRIC_DIRECTIONS: dict[str, dict[str, str]] = {
    "insertion": {
        "label": "Insertion",
        "direction": "higher_better",
        "applies_to": "attribution",
    },
    "deletion": {
        "label": "Deletion",
        "direction": "lower_better",
        "applies_to": "attribution",
    },
    "mse": {
        "label": "MSE",
        "direction": "lower_better",
        "applies_to": "concept",
    },
    "fid": {
        "label": "FID",
        "direction": "lower_better",
        "applies_to": "concept",
    },
    "sparsity": {
        "label": "Sparsity",
        "direction": "lower_better",
        "applies_to": "concept",
    },
    "sparsity_ratio": {
        "label": "Sparsity ratio",
        "direction": "lower_better",
        "applies_to": "concept",
    },
}


def resolve_interpreto_version() -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python <3.8
        return None
    try:
        return version("interpreto")
    except PackageNotFoundError:
        return None


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


def load_metric_scores_for(
    model_id: str,
    type_name: str,
    scope_name: str,
) -> Optional[dict]:
    """Return ``{method_name: {metric_name: value}}`` or ``None``.

    Attribution scores live under ``data/<model>/metrics/attribution/<scope>.json``;
    concept scores live under ``data/<model>/metrics/concept/general.json`` and
    are broadcast to every scope. Returns ``None`` when the sidecar is
    missing so the caller can decide whether to emit the field.
    """
    if type_name == "attribution":
        path = DATA_ROOT / model_id / "metrics" / "attribution" / f"{scope_name}.json"
    elif type_name == "concept":
        path = DATA_ROOT / model_id / "metrics" / "concept" / "general.json"
    else:
        return None
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def add_entry(
    entries: list[dict],
    model_id: str,
    task: str,
    type_name: str,
    scope_name: str,
    sample: Optional[str],
    methods: list[str],
    metric_scores: Optional[dict] = None,
) -> None:
    if not methods:
        return
    entry: dict = {
        "model": model_id,
        "task": task,
        "type": type_name,
        "scope": scope_name,
        "sample": sample,
        "methods": methods,
    }
    if metric_scores:
        # Only keep entries for methods actually present in this scope.
        filtered = {
            method: metric_scores[method]
            for method in methods
            if method in metric_scores and metric_scores[method]
        }
        if filtered:
            entry["metric_scores"] = filtered
    entries.append(entry)


def scan_scope_dir(
    scope_dir: Path,
    model_id: str,
    task: str,
    type_name: str,
    scope_name: str,
) -> list[dict]:
    entries: list[dict] = []

    metric_scores = load_metric_scores_for(model_id, type_name, scope_name)

    methods = collect_methods(scope_dir)
    add_entry(
        entries,
        model_id,
        task,
        type_name,
        scope_name,
        None,
        methods,
        metric_scores=metric_scores,
    )

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
                metric_scores=metric_scores,
            )

    return entries


def collect_used_metrics(entries: list[dict]) -> list[str]:
    """Return the sorted list of metric names actually present in the manifest."""
    used: set[str] = set()
    for entry in entries:
        for method_scores in entry.get("metric_scores", {}).values():
            used.update(method_scores.keys())
    # Preserve METRIC_DIRECTIONS ordering for the ones we know about; append
    # any unknown ones at the end (shouldn't happen but keeps forward-compat).
    ordered = [name for name in METRIC_DIRECTIONS if name in used]
    ordered.extend(sorted(name for name in used if name not in METRIC_DIRECTIONS))
    return ordered


def build_manifest() -> dict:
    manifest = {
        "interpreto_version": resolve_interpreto_version(),
        "metrics_meta": METRIC_DIRECTIONS,
        "metrics_summary": [],
        "models": {},
        "explanations": [],
    }
    if not EXPLANATIONS_DIR.exists():
        return manifest

    entries: list[dict] = []

    for model_dir in sorted(
        path for path in EXPLANATIONS_DIR.iterdir() if path.is_dir()
    ):
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
    manifest["metrics_summary"] = collect_used_metrics(manifest["explanations"])

    return manifest


def main() -> None:
    manifest = build_manifest()
    OUTPUT_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
