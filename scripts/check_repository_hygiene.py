"""Fail when tracked files contain common secrets, PII, or local artifacts."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

DISALLOWED_PARTS = {".idea", ".vscode", "__pycache__", "node_modules"}
DISALLOWED_NAMES = {".DS_Store", "Thumbs.db", "unknown_response.json"}
DISALLOWED_SUFFIXES = {
    ".jar",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".pkl",
    ".png",
    ".pptx",
    ".pt",
    ".pth",
    ".wav",
    ".webm",
    ".zip",
}
CONTENT_RULES = {
    "Chinese mobile number": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "Chinese resident ID": re.compile(
        r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])"
        r"(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"
    ),
    "email address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "private key": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    "common API token": re.compile(r"(?:sk-|hf_)[A-Za-z0-9_-]{20,}"),
}


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    )
    return [Path(item.decode()) for item in output.split(b"\0") if item]


def main() -> int:
    violations: list[tuple[Path, str]] = []
    for path in tracked_files():
        if not path.is_file():
            continue
        if path.name in DISALLOWED_NAMES or DISALLOWED_PARTS.intersection(path.parts):
            violations.append((path, "local artifact"))
            continue
        if path.suffix.lower() in DISALLOWED_SUFFIXES:
            violations.append((path, f"disallowed binary ({path.suffix.lower()})"))
            continue

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            violations.append((path, "unreviewed binary file"))
            continue
        for rule_name, pattern in CONTENT_RULES.items():
            if pattern.search(content):
                violations.append((path, rule_name))

    if violations:
        print("Repository hygiene check failed:")
        for path, reason in violations:
            print(f"- {path}: {reason}")
        return 1
    print(f"Repository hygiene check passed ({len(tracked_files())} tracked files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
