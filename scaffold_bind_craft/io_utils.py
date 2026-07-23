from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_csv(path: Path, row: dict[str, Any], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    ordered = list(fieldnames)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def ensure_csv_fields(
    path: Path,
    fieldnames: Iterable[str],
    removable_fields: Iterable[str] = (),
    renamed_fields: dict[str, str] | None = None,
) -> None:
    """Expand an existing CSV schema without dropping or shifting old data."""
    if not path.exists() or path.stat().st_size == 0:
        return
    ordered = list(fieldnames)
    removable = set(removable_fields)
    renamed = renamed_fields or {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        existing = list(reader.fieldnames or [])
        if existing == ordered:
            return
        unknown = [
            field for field in existing
            if field not in ordered and field not in removable and field not in renamed
        ]
        if unknown:
            raise ValueError(f"Cannot migrate {path}: unknown existing CSV fields: {unknown}")
        rows = list(reader)

    for row in rows:
        for old_field, new_field in renamed.items():
            if not row.get(new_field) and row.get(old_field) is not None:
                row[new_field] = row[old_field]

    temporary = path.with_suffix(path.suffix + ".schema.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def flatten_json(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten_json(f"{prefix}_{key}" if prefix else str(key), child, out)
    else:
        out[prefix] = value


def count_fasta_records(path: Path, skip_native: bool = True) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                count += 1
    if skip_native and count > 0:
        return count - 1
    return count


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    seq_parts: list[str] = []
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_parts)))
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        records.append((header, "".join(seq_parts)))
    return records


def symlink_or_copy(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.symlink(source.resolve(), dest)
    except OSError:
        shutil.copy2(source, dest)
