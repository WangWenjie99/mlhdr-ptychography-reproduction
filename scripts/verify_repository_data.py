#!/usr/bin/env python3
"""Verify the committed research data with Python's standard library only.

Run this after cloning or pulling the repository on another computer:
    python scripts/verify_repository_data.py

The repository and manifest are located relative to this script, so the command
also works when invoked by absolute path from a different working directory.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "docs" / "REPOSITORY_DATA.json"
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_repository_data(root: Path, manifest_path: Path) -> tuple[int, list[str]]:
    """Return the number of valid files and every encountered integrity error."""
    root = root.resolve()
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return 0, [f"Cannot read manifest {manifest_path}: {exc}"]

    if not isinstance(manifest, dict):
        return 0, ["Manifest must be a JSON object."]
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        return 0, ["Manifest schema_version must be the integer 1."]
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        return 0, ["Manifest files must be a non-empty array."]

    errors: list[str] = []
    seen: set[str] = set()
    valid_count = 0
    for index, entry in enumerate(entries):
        label = f"files[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{label}: entry must be a JSON object.")
            continue

        relative = entry.get("path")
        if not isinstance(relative, str) or not relative:
            errors.append(f"{label}: path must be a non-empty string.")
            continue
        label = relative
        # Use a portable, canonical POSIX spelling. Reject Windows drive/root
        # syntax too, including when the check runs on Linux or macOS.
        parts = relative.split("/")
        if (
            "\\" in relative
            or ":" in relative
            or "\x00" in relative
            or PurePosixPath(relative).is_absolute()
            or any(part in ("", ".", "..") for part in parts)
        ):
            errors.append(f"{label}: unsafe path; use a canonical relative path with '/'.")
            continue
        portable_key = relative.casefold()
        if portable_key in seen:
            errors.append(f"{label}: duplicate path (case-insensitive).")
            continue
        seen.add(portable_key)

        expected_size = entry.get("size_bytes")
        expected_hash = entry.get("sha256")
        role = entry.get("role")
        if type(expected_size) is not int or expected_size < 0:
            errors.append(f"{label}: size_bytes must be a non-negative integer.")
            continue
        if not isinstance(expected_hash, str) or not SHA256_PATTERN.fullmatch(expected_hash):
            errors.append(f"{label}: sha256 must contain exactly 64 hexadecimal characters.")
            continue
        if not isinstance(role, str) or not role.strip():
            errors.append(f"{label}: role must be a non-empty string.")
            continue

        try:
            path = root.joinpath(*parts).resolve()
            if not path.is_relative_to(root):
                errors.append(f"{label}: resolved path escapes the repository.")
                continue
            if not path.is_file():
                errors.append(f"{label}: file is missing or is not a regular file.")
                continue
            actual_size = path.stat().st_size
            if actual_size != expected_size:
                errors.append(
                    f"{label}: size mismatch (expected {expected_size}, found {actual_size} bytes)."
                )
                continue
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash.lower():
                errors.append(
                    f"{label}: SHA256 mismatch (expected {expected_hash.lower()}, found {actual_hash})."
                )
                continue
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(f"{label}: cannot verify file: {exc}")
            continue
        valid_count += 1

    return valid_count, errors


def main() -> int:
    valid_count, errors = verify_repository_data(REPOSITORY_ROOT, MANIFEST_PATH)
    if errors:
        print(f"DATA CHECK FAILED: {len(errors)} error(s); {valid_count} file(s) verified.", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"DATA CHECK PASSED: all {valid_count} repository data files match size and SHA256.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
