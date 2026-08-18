#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scaffold_bind_craft.config import load_settings, require, stage_dir
from scaffold_bind_craft.multimer_clash import check_configured_multimer_clashes


def number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def base_filter_result(row: dict[str, Any], settings: dict[str, Any]) -> tuple[bool, str]:
    filters = require(settings, "rso").get("save_filters", {})
    checks = (
        ("plddt", filters.get("min_plddt"), True),
        ("iptm", filters.get("min_iptm"), True),
        ("pae", filters.get("max_pae"), False),
        ("i_pae", filters.get("max_i_pae"), False),
    )
    failed: list[str] = []
    for field, threshold, higher in checks:
        value = number(row.get(field))
        if threshold is None or value is None:
            continue
        passed = value >= float(threshold) if higher else value <= float(threshold)
        if not passed:
            failed.append(f"{field} {value} not {'>=' if higher else '<='} {threshold}")
    return not failed, "; ".join(failed)


def load_csv(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"RSO metrics file not found: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".refilter.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"RSO manifest not found: {path}")
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".refilter.tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def refilter(settings_file: str, dry_run: bool) -> int:
    settings = load_settings(settings_file)
    out = stage_dir(settings, "rso")
    accepted_dir = out / "accepted_pdbs"
    filtered_dir = out / "filtered_pdbs"
    metrics_path = out / "rso_metrics.csv"
    manifest_path = out / "manifest.jsonl"
    fields, metrics = load_csv(metrics_path)
    manifest = load_jsonl(manifest_path)
    required = ["multimer_clash_count", "multimer_clashing_chains", "multimer_alignment_rmsd"]
    missing = [field for field in required if field not in fields]
    if missing:
        raise ValueError(f"{metrics_path} lacks required columns: {missing}")

    metric_rows = {str(row.get("design_name")): row for row in metrics if row.get("design_name")}
    manifest_rows: dict[str, list[dict[str, Any]]] = {}
    for row in manifest:
        if row.get("design_name"):
            manifest_rows.setdefault(str(row["design_name"]), []).append(row)

    pdbs = sorted(accepted_dir.glob("*.pdb")) + sorted(filtered_dir.glob("*.pdb"))
    if not pdbs:
        print(f"{settings_file}: no RSO PDBs found in {out}")
        return 0

    changed = clashes = accepted = 0
    for pdb_path in pdbs:
        name = pdb_path.stem
        row = metric_rows.get(name)
        if row is None:
            raise KeyError(f"No rso_metrics.csv row for {pdb_path.name}")
        result = check_configured_multimer_clashes(pdb_path, settings)
        if result is None:
            raise ValueError("multimer_clash_filter must be enabled")
        base_passed, base_reason = base_filter_result(row, settings)
        status = "accepted" if base_passed and not result.clashes else "filtered"
        destination = (accepted_dir if status == "accepted" else filtered_dir) / pdb_path.name
        reason = base_reason
        if result.clashes:
            clashes += 1
            clash_reason = (
                f"binder clashes with reference chain(s) {','.join(result.clashing_chains)} "
                f"at {result.clash_count} atom(s)"
            )
            reason = "; ".join(part for part in (reason, clash_reason) if part)
        if status == "accepted":
            accepted += 1

        updates: dict[str, Any] = {
            "status": status,
            "multimer_clash_count": result.clash_count,
            "multimer_clashing_chains": ",".join(result.clashing_chains),
            "multimer_alignment_rmsd": round(result.alignment_rmsd, 4),
            "output_pdb": str(destination),
            "reason": reason,
        }
        old = {key: str(row.get(key, "")) for key in updates}
        if old != {key: str(value) for key, value in updates.items()} or destination != pdb_path:
            changed += 1
        row.update(updates)
        for manifest_row in manifest_rows.get(name, []):
            manifest_row.update(updates)
        print(f"{name}: {status}, clashes={result.clash_count}, chains={','.join(result.clashing_chains) or '-'}")
        if not dry_run and destination != pdb_path:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite {destination}")
            pdb_path.replace(destination)

    if not dry_run:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(metrics_path, metrics_path.with_name(f"{metrics_path.name}.{stamp}.bak"))
        shutil.copy2(manifest_path, manifest_path.with_name(f"{manifest_path.name}.{stamp}.bak"))
        write_csv(metrics_path, fields, metrics)
        write_jsonl(manifest_path, manifest)
    mode = "DRY RUN" if dry_run else "UPDATED"
    print(f"{settings_file}: {mode}: checked={len(pdbs)}, accepted={accepted}, clashing={clashes}, changed={changed}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-run multimer clash filtering on every saved RSO PDB.")
    parser.add_argument("settings", nargs="+", help="One or more settings JSON files")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without moving or rewriting files")
    args = parser.parse_args()
    for settings_file in args.settings:
        refilter(settings_file, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
