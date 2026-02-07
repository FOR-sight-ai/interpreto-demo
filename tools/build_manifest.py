import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTIONS_DIR = ROOT / "attributions"
OUTPUT_PATH = ROOT / "manifest.json"


def build_manifest() -> dict:
    manifest = {"models": {}}
    if not ATTRIBUTIONS_DIR.exists():
        return manifest

    for model_dir in sorted(
        path for path in ATTRIBUTIONS_DIR.iterdir() if path.is_dir()
    ):
        samples = {}
        for sample_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            methods = sorted(
                file.name
                for file in sample_dir.iterdir()
                if file.is_file() and file.suffix.lower() == ".html"
            )
            samples[sample_dir.name] = {"methods": methods}

        if samples:
            manifest["models"][model_dir.name] = {"samples": samples}

    return manifest


def main() -> None:
    manifest = build_manifest()
    OUTPUT_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
