#!/usr/bin/env python3
"""Externalize Interpreto CSS/JS from explanation HTML files."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


HEAD_RE = re.compile(r"<head>(.*?)</head>", re.DOTALL | re.IGNORECASE)
INLINE_STYLE_RE = re.compile(r"<style[^>]*>(.*?)</style>", re.DOTALL | re.IGNORECASE)
INLINE_SCRIPT_RE = re.compile(
    r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>",
    re.DOTALL | re.IGNORECASE,
)
STYLE_RE = re.compile(r"<style[^>]*>.*?</style>", re.DOTALL | re.IGNORECASE)
SCRIPT_RE = re.compile(r"<script[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)
LINK_RE = re.compile(
    r"<link[^>]*rel=[\"']stylesheet[\"'][^>]*>", re.IGNORECASE
)
DEFAULT_JS_DIRS = ("assets/js/core", "assets/js/visualizations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move shared Interpreto CSS/JS into assets and shrink HTML."
    )
    parser.add_argument(
        "--explanations-dir",
        default="explanations",
        help="Root folder containing explanation HTML files.",
    )
    parser.add_argument(
        "--mode",
        choices=("link", "extract"),
        default="link",
        help="link uses existing assets; extract writes assets from inline HTML.",
    )
    parser.add_argument(
        "--css-path",
        default="assets/css/visualization.css",
        help="CSS file to link (link mode) or write (extract mode).",
    )
    parser.add_argument(
        "--js-path",
        default="assets/js/visualization_bundle.js",
        help="JS bundle to write in extract mode.",
    )
    parser.add_argument(
        "--js-dir",
        action="append",
        default=None,
        help=(
            "Directory to include .js files from in link mode "
            "(can be repeated; order matters)."
        ),
    )
    parser.add_argument(
        "--js-file",
        action="append",
        default=None,
        help=(
            "Explicit .js files to include in link mode "
            "(use to override default directories)."
        ),
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Specific HTML file to use as the source for CSS/JS extraction.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing files.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str, dry_run: bool) -> None:
    if dry_run:
        return
    path.write_text(content, encoding="utf-8")


def iter_html_files(explanations_dir: Path) -> list[Path]:
    if not explanations_dir.exists():
        return []
    return sorted(
        path for path in explanations_dir.rglob("*.html") if path.is_file()
    )


def extract_inline_assets(html: str) -> dict | None:
    head_match = HEAD_RE.search(html)
    if not head_match:
        return None

    head = head_match.group(1)
    style_match = INLINE_STYLE_RE.search(head)
    script_match = INLINE_SCRIPT_RE.search(head)
    if not style_match or not script_match:
        return None

    return {
        "head_match": head_match,
        "head": head,
        "style_block": style_match.group(0),
        "script_block": script_match.group(0),
        "style": style_match.group(1),
        "script": script_match.group(1),
    }


def get_rel_path(target: Path, base: Path) -> str:
    return Path(os.path.relpath(target, start=base)).as_posix()


def collect_js_paths(js_files: list[str] | None, js_dirs: list[str]) -> list[Path]:
    paths: list[Path] = []
    if js_files:
        paths.extend(Path(path) for path in js_files)

    for js_dir in js_dirs:
        dir_path = Path(js_dir)
        paths.extend(sorted(dir_path.glob("*.js")))

    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = path.as_posix()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def build_head_assets(head: str, link_tag: str, script_tags: str) -> str:
    cleaned = STYLE_RE.sub("", head)
    cleaned = LINK_RE.sub("", cleaned)
    cleaned = SCRIPT_RE.sub("", cleaned)
    cleaned = cleaned.strip()
    if cleaned:
        return f"{link_tag}{script_tags}{cleaned}"
    return f"{link_tag}{script_tags}"


def collect_base_assets(
    html_files: list[Path], source: Path | None
) -> tuple[str | None, str | None, Path | None, list[Path]]:
    base_style = None
    base_script = None
    base_path = None
    mismatches: list[Path] = []

    if source:
        html = read_text(source)
        data = extract_inline_assets(html)
        if not data:
            raise ValueError(f"No inline <style>/<script> found in {source}")
        base_style = data["style"]
        base_script = data["script"]
        base_path = source

    for path in html_files:
        html = read_text(path)
        data = extract_inline_assets(html)
        if not data:
            continue
        if base_style is None:
            base_style = data["style"]
            base_script = data["script"]
            base_path = path
            continue
        if data["style"] != base_style or data["script"] != base_script:
            mismatches.append(path)

    return base_style, base_script, base_path, mismatches


def externalize_file(
    path: Path,
    css_path: Path,
    js_paths: list[Path],
    dry_run: bool,
) -> bool:
    html = read_text(path)
    head_match = HEAD_RE.search(html)
    if not head_match:
        return False

    css_rel = get_rel_path(css_path, path.parent)
    link_tag = f'<link rel="stylesheet" href="{css_rel}">'
    script_tags = "".join(
        f'<script src="{get_rel_path(js_path, path.parent)}"></script>'
        for js_path in js_paths
    )

    head = head_match.group(1)
    new_head = build_head_assets(head, link_tag, script_tags)

    new_html = (
        html[: head_match.start(1)]
        + new_head
        + html[head_match.end(1) :]
    )
    if new_html == html:
        return False

    write_text(path, new_html, dry_run=dry_run)
    return True


def main() -> int:
    args = parse_args()
    explanations_dir = Path(args.explanations_dir)
    css_path = Path(args.css_path)
    js_path = Path(args.js_path)
    source_path = Path(args.source) if args.source else None

    html_files = iter_html_files(explanations_dir)
    if not html_files:
        print(f"No HTML files found under {explanations_dir}", file=sys.stderr)
        return 1

    base_path = None
    if args.mode == "link":
        js_files = args.js_file or []
        if args.js_dir is None:
            js_dirs = [] if js_files else list(DEFAULT_JS_DIRS)
        else:
            js_dirs = args.js_dir
        js_paths = collect_js_paths(js_files, js_dirs)

        if not css_path.exists():
            print(f"Missing CSS file: {css_path}", file=sys.stderr)
            return 1
        if not js_paths:
            print("No JS files found to link.", file=sys.stderr)
            return 1
        missing_js = [path for path in js_paths if not path.exists()]
        if missing_js:
            missing_list = "\n".join(str(path) for path in missing_js[:10])
            extra = ""
            if len(missing_js) > 10:
                extra = f"\n... and {len(missing_js) - 10} more"
            print(
                "Missing JS files:\n" f"{missing_list}{extra}",
                file=sys.stderr,
            )
            return 1
    else:
        js_paths = [js_path]
        base_style, base_script, base_path, mismatches = collect_base_assets(
            html_files, source_path
        )
        if mismatches:
            mismatch_list = "\n".join(str(path) for path in mismatches[:10])
            extra = ""
            if len(mismatches) > 10:
                extra = f"\n... and {len(mismatches) - 10} more"
            print(
                "Found HTML files with different inline assets. "
                "Re-run with --source to pick a specific template.\n"
                f"{mismatch_list}{extra}",
                file=sys.stderr,
            )
            return 1

        if base_style is None or base_script is None or base_path is None:
            print("No inline assets found to extract.", file=sys.stderr)
            return 1

        if not args.dry_run:
            css_path.parent.mkdir(parents=True, exist_ok=True)
            js_path.parent.mkdir(parents=True, exist_ok=True)

        write_text(css_path, base_style, dry_run=args.dry_run)
        write_text(js_path, base_script, dry_run=args.dry_run)

    updated = 0
    for path in html_files:
        if externalize_file(path, css_path, js_paths, dry_run=args.dry_run):
            updated += 1

    action = "Would update" if args.dry_run else "Updated"
    if args.mode == "link":
        print(f"{action} {updated} HTML files. Linked {css_path}.")
    else:
        print(
            f"{action} {updated} HTML files. "
            f"Assets from {base_path} -> {css_path}, {js_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
