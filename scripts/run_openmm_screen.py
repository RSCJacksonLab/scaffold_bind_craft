#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scaffold_bind_craft.config import load_settings, require, stage_dir
from scaffold_bind_craft.chai1_utils import find_chai1_predictions
from scaffold_bind_craft.io_utils import append_csv, ensure_csv_fields, flatten_json, read_csv_rows
from scaffold_bind_craft.openmm_screen_utils import (
    calculate_clash_score,
    openmm_relax_subprocess,
    pr_alternative_score_interface,
)
from scaffold_bind_craft.pdb_utils import extract_chain_sequences


BASE_FIELDS = [
    "design_name",
    "status",
    "sequence",
    "input_prediction",
    "unrelaxed_pdb",
    "relaxed_pdb",
    "unrelaxed_clashes",
    "relaxed_clashes",
    "reason",
    "failed_metric",
    "failed_value",
    "elapsed_seconds",
]

PREDICTION_FIELDS = [
    "chai1_plddt",
    "chai1_binder_plddt",
    "chai1_target_plddt",
    "chai1_chain_count",
    "chai1_aggregate_score",
    "chai1_iptm",
    "chai1_pae",
]

INTERFACE_FIELDS = [
    "interface_surface_hydrophobicity",
    "interface_interface_dSASA",
    "interface_interface_fraction",
    "interface_interface_hydrophobicity",
    "interface_interface_nres",
    "interface_interface_interface_hbonds",
    "interface_interface_hbond_percentage",
    "interface_hbonds_per_1000_dSASA",
    "interface_buried_polar_atoms_total",
    "interface_buried_satisfied_polar_atoms",
    "interface_buried_unsatisfied_polar_atoms",
    "interface_buried_polar_satisfaction_percentage",
    "interface_binder_buried_unsatisfied_polar_atoms",
    "interface_target_buried_unsatisfied_polar_atoms",
    "interface_buried_unsatisfied_donor_atoms",
    "interface_buried_unsatisfied_acceptor_atoms",
    "interface_delta_buried_unsatisfied_polar_atoms",
    "interface_salt_bridges",
    "interface_interface_delta_unsat_hbonds",
    "interface_interface_delta_unsat_hbonds_percentage",
    "interface_residues",
    "interface_amino_acids",
]

LEGACY_INTERFACE_FIELDS = {
    "interface_binder_score",
    "interface_interface_sc",
    "interface_interface_packstat",
    "interface_interface_dG",
    "interface_interface_dG_SASA_ratio",
}


def passes_filters(
    row: dict[str, Any],
    filters: dict[str, Any],
    sample_metrics: dict[str, list[float]] | None = None,
    failed_metrics: list[str] | None = None,
) -> tuple[bool, str]:
    failed: list[str] = []

    def record_failure(metric: str, value: str) -> None:
        failed.append(value)
        if failed_metrics is not None:
            failed_metrics.append(metric)

    for key, rule in filters.items():
        if not isinstance(rule, dict):
            continue
        higher = bool(rule.get("higher", True))
        threshold = rule.get("threshold")
        if threshold is not None:
            value = row.get(key)
            try:
                numeric = float(value)
            except Exception:
                record_failure(key, f"{key} missing")
            else:
                if higher and numeric < float(threshold):
                    record_failure(key, str(value))
                if not higher and numeric > float(threshold):
                    record_failure(key, str(value))

        average_threshold = rule.get("average_threshold")
        if average_threshold is not None:
            avg_key = f"{key}_avg"
            expected_values = int(row.get("prediction_sample_count", 0) or 0)
            values = (sample_metrics or {}).get(key, [])
            if expected_values and values and len(values) < expected_values:
                record_failure(key, f"{key} has {len(values)} numeric samples, expected {expected_values}")
                continue
            value = row.get(avg_key)
            try:
                numeric = float(value)
            except Exception:
                record_failure(avg_key, f"{avg_key} missing")
            else:
                if higher and numeric < float(average_threshold):
                    record_failure(avg_key, str(value))
                if not higher and numeric > float(average_threshold):
                    record_failure(avg_key, str(value))

        all_threshold = rule.get("all_threshold")
        if all_threshold is not None:
            values = (sample_metrics or {}).get(key, [])
            if not values:
                record_failure(key, f"{key} samples missing")
                continue
            expected_values = int(row.get("prediction_sample_count", len(values)) or len(values))
            if len(values) < expected_values:
                record_failure(key, f"{key} has {len(values)} numeric samples, expected {expected_values}")
                continue
            bad_values = [value for value in values if value < float(all_threshold)] if higher else [
                value for value in values if value > float(all_threshold)
            ]
            if bad_values:
                record_failure(key, ",".join(str(value) for value in bad_values))
    return (not failed, "; ".join(failed))


def chai1_filters(filters: dict[str, Any]) -> dict[str, Any]:
    return {key: rule for key, rule in filters.items() if key.startswith("chai1_")}


def post_prediction_filters(filters: dict[str, Any]) -> dict[str, Any]:
    prediction_prefixes = ("chai1_", "protenix_", "boltz_")
    return {key: rule for key, rule in filters.items() if not key.startswith(prediction_prefixes)}


def first_chain_id(value: Any, default: str = "A") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        chains = [part.strip() for part in value.replace(",", " ").split() if part.strip()]
    else:
        chains = [str(part).strip() for part in value if str(part).strip()]
    return chains[0] if chains else default


def find_prediction_records(settings: dict[str, Any], screen: dict[str, Any]) -> dict[str, tuple[Path, Path]]:
    prediction_stage = str(screen.get("prediction_stage", "chai1")).lower()
    if prediction_stage == "chai1":
        prediction_dirs = [stage_dir(settings, "chai1")]
    elif prediction_stage == "nise":
        prediction_dirs = sorted(stage_dir(settings, "nise").glob("round_*/chai1"))
    else:
        raise ValueError("screening.prediction_stage must be chai1 or nise")

    records: dict[str, tuple[Path, Path]] = {}
    for prediction_dir in prediction_dirs:
        for design_name, pdb_path, metrics_path in find_chai1_predictions(prediction_dir):
            if design_name in records:
                raise ValueError(f"Duplicate prediction name across screening inputs: {design_name}")
            records[design_name] = (pdb_path, metrics_path)
    return records


def processed_names(csv_path: Path, rerun_failed: bool) -> set[str]:
    done = set()
    for row in read_csv_rows(csv_path):
        if rerun_failed and row.get("status", "").startswith("failed"):
            continue
        if row.get("design_name"):
            done.add(row["design_name"])
    return done


def processed_sequences(csv_path: Path, binder_chain: str) -> dict[str, str]:
    """Map screened binder sequences to the first design that used them."""
    seen: dict[str, str] = {}
    for row in read_csv_rows(csv_path):
        if row.get("status", "").startswith("failed"):
            continue
        sequence = row.get("sequence", "")
        if not sequence and row.get("input_prediction"):
            prediction_path = Path(row["input_prediction"])
            if prediction_path.exists():
                sequence = extract_chain_sequences(prediction_path, [binder_chain]).get(binder_chain, "")
        if sequence:
            seen.setdefault(sequence.upper(), row.get("design_name", "unknown design"))
    return seen


def add_chai1_summary_metrics(row: dict[str, Any], confidence: dict[str, Any]) -> None:
    mapping = {
        "plddt": "chai1_plddt",
        "binder_plddt": "chai1_binder_plddt",
        "target_plddt": "chai1_target_plddt",
        "chain_count": "chai1_chain_count",
        "aggregate_score": "chai1_aggregate_score",
        "iptm": "chai1_iptm",
        "pae": "chai1_pae",
    }
    for source, dest in mapping.items():
        if source in confidence:
            row[dest] = confidence[source]


def main() -> int:
    parser = argparse.ArgumentParser(description="Relax Chai-1 predictions with OpenMM and screen interfaces.")
    parser.add_argument("--settings", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int, default=None, help="Maximum pending predictions to screen in this run.")
    args = parser.parse_args()

    settings = load_settings(args.settings, args.output_dir)
    screen = require(settings, "screening")
    out = stage_dir(settings, "openmm")
    pdb_dir = out / "pdbs"
    relaxed_dir = out / "relaxed_pdbs"
    accepted_dir = out / "accepted"
    for path in (pdb_dir, relaxed_dir, accepted_dir):
        path.mkdir(parents=True, exist_ok=True)

    prediction_records = find_prediction_records(settings, screen)
    names = list(prediction_records)
    metrics_csv = out / "screening_metrics.csv"
    fields = BASE_FIELDS + PREDICTION_FIELDS + INTERFACE_FIELDS
    ensure_csv_fields(metrics_csv, fields, removable_fields=LEGACY_INTERFACE_FIELDS)
    done = processed_names(metrics_csv, bool(screen.get("rerun_failed", False)))
    pending = [name for name in names if name not in done]
    if args.limit is not None:
        pending = pending[: max(0, int(args.limit))]
    print(f"OpenMM screening predictions={len(names)}, pending={len(pending)}")
    if not pending:
        return 0

    binder_chain = screen.get("binder_chain", settings.get("binder_chain", "B"))
    target_chain = screen.get("target_chain", first_chain_id(settings.get("target_chains", "A")))
    filters = screen.get("filters", {})
    seen_sequences = processed_sequences(metrics_csv, str(binder_chain))

    for design_name in pending:
        start = time.time()
        prediction_path: Path | None = None
        metrics_path: Path | None = None
        row: dict[str, Any] = {"design_name": design_name}
        try:
            sample_metrics: dict[str, list[float]] = {}
            prediction_path, metrics_path = prediction_records[design_name]
            if not prediction_path or not metrics_path:
                raise FileNotFoundError("Chai-1 PDB/metrics JSON not found")
            with metrics_path.open() as handle:
                confidence = json.load(handle)

            flat_conf: dict[str, Any] = {}
            flatten_json("chai1", confidence, flat_conf)
            row.update({key: value for key, value in flat_conf.items() if key in PREDICTION_FIELDS})
            add_chai1_summary_metrics(row, confidence)

            sequence = extract_chain_sequences(prediction_path, [str(binder_chain)]).get(str(binder_chain), "")
            row["sequence"] = sequence
            sequence_key = sequence.upper()
            if sequence_key and sequence_key in seen_sequences:
                row.update(
                    {
                        "status": "filtered",
                        "sequence": sequence,
                        "input_prediction": str(prediction_path),
                        "unrelaxed_pdb": "",
                        "relaxed_pdb": "",
                        "unrelaxed_clashes": "",
                        "relaxed_clashes": "",
                        "reason": f"duplicate sequence (same as {seen_sequences[sequence_key]})",
                    }
                )
                row["elapsed_seconds"] = round(time.time() - start, 2)
                append_csv(metrics_csv, row, fields)
                print(f"{design_name}: {row['status']} {row['reason']}")
                continue
            if sequence_key:
                seen_sequences[sequence_key] = design_name

            failed_metrics: list[str] = []
            passed, reason = passes_filters(row, chai1_filters(filters), sample_metrics, failed_metrics)
            if not passed:
                row.update(
                    {
                        "status": "filtered",
                        "sequence": "",
                        "input_prediction": str(prediction_path),
                        "unrelaxed_pdb": "",
                        "relaxed_pdb": "",
                        "unrelaxed_clashes": "",
                        "relaxed_clashes": "",
                        "reason": reason,
                        "failed_metric": ";".join(failed_metrics),
                        "failed_value": reason,
                    }
                )
                row["elapsed_seconds"] = round(time.time() - start, 2)
                append_csv(metrics_csv, row, fields)
                print(f"{design_name}: {row['status']} {row.get('reason', '')}")
                continue

            unrelaxed_pdb = pdb_dir / f"{design_name}_chai1.pdb"
            if not unrelaxed_pdb.exists() or unrelaxed_pdb.stat().st_size == 0:
                shutil.copy2(prediction_path, unrelaxed_pdb)
            relaxed_pdb = relaxed_dir / f"{design_name}_chai1_openmm_relaxed.pdb"
            if not relaxed_pdb.exists() or relaxed_pdb.stat().st_size == 0:
                openmm_relax_subprocess(
                    str(unrelaxed_pdb),
                    str(relaxed_pdb),
                    use_gpu_relax=bool(screen.get("use_gpu_relax", True)),
                    timeout=screen.get("openmm_timeout_seconds"),
                    max_attempts=int(screen.get("openmm_max_attempts", 3)),
                    use_faspr_repack=bool(screen.get("use_faspr_repack", True)),
                    starting_pdb=settings.get("input_pdb"),
                    target_chains=settings.get("target_chains"),
                )
            unrelaxed_clashes = calculate_clash_score(str(unrelaxed_pdb), threshold=float(screen.get("clash_threshold", 2.4)))
            relaxed_clashes = calculate_clash_score(str(relaxed_pdb), threshold=float(screen.get("clash_threshold", 2.4)))
            interface_scores, interface_aa, interface_residues = pr_alternative_score_interface(
                str(relaxed_pdb),
                binder_chain=str(binder_chain),
                target_chain=str(target_chain),
                sasa_engine=screen.get("sasa_engine", "auto"),
            )
            score_fields = {f"interface_{key}": value for key, value in interface_scores.items()}
            row.update(score_fields)
            row["interface_residues"] = interface_residues
            row["interface_amino_acids"] = json.dumps(interface_aa, sort_keys=True)
            row.update(
                {
                    "status": "accepted",
                    "sequence": sequence,
                    "input_prediction": str(prediction_path),
                    "unrelaxed_pdb": str(unrelaxed_pdb),
                    "relaxed_pdb": str(relaxed_pdb),
                    "unrelaxed_clashes": unrelaxed_clashes,
                    "relaxed_clashes": relaxed_clashes,
                    "reason": "",
                }
            )
            failed_metrics = []
            passed, reason = passes_filters(row, post_prediction_filters(filters), sample_metrics, failed_metrics)
            if not passed:
                row["status"] = "filtered"
                row["reason"] = reason
                row["failed_metric"] = ";".join(failed_metrics)
                row["failed_value"] = reason
            else:
                shutil.copy2(relaxed_pdb, accepted_dir / relaxed_pdb.name)
        except Exception as exc:
            failed_sequence = str(row.get("sequence", "")).upper()
            if failed_sequence and seen_sequences.get(failed_sequence) == design_name:
                del seen_sequences[failed_sequence]
            row.update(
                {
                    "status": "failed",
                    "sequence": row.get("sequence", ""),
                    "input_prediction": str(prediction_path or ""),
                    "unrelaxed_pdb": "",
                    "relaxed_pdb": "",
                    "unrelaxed_clashes": "",
                    "relaxed_clashes": "",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
        row["elapsed_seconds"] = round(time.time() - start, 2)
        append_csv(metrics_csv, row, fields)
        print(f"{design_name}: {row['status']} {row.get('reason', '')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
