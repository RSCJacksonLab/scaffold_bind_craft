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
from scaffold_bind_craft.fast_protenix_utils import find_protenix_predictions
from scaffold_bind_craft.io_utils import append_csv, flatten_json, read_csv_rows
from scaffold_bind_craft.openmm_screen_utils import (
    calculate_clash_score,
    openmm_relax_subprocess,
    pr_alternative_score_interface,
)
from scaffold_bind_craft.pdb_utils import cif_to_pdb, extract_chain_sequences


BASE_FIELDS = [
    "design_name",
    "status",
    "sequence",
    "input_cif",
    "unrelaxed_pdb",
    "relaxed_pdb",
    "unrelaxed_clashes",
    "relaxed_clashes",
    "reason",
    "elapsed_seconds",
]

PREDICTION_FIELDS = [
    "protenix_selected_sample_index",
    "protenix_sample_count",
    "protenix_plddt",
    "protenix_plddt_avg",
    "protenix_plddt_min",
    "protenix_plddt_max",
    "protenix_gpde",
    "protenix_gpde_avg",
    "protenix_gpde_min",
    "protenix_gpde_max",
    "protenix_ptm",
    "protenix_ptm_avg",
    "protenix_ptm_min",
    "protenix_ptm_max",
    "protenix_iptm",
    "protenix_iptm_avg",
    "protenix_iptm_min",
    "protenix_iptm_max",
    "protenix_ranking_score",
    "protenix_ranking_score_avg",
    "protenix_ranking_score_min",
    "protenix_ranking_score_max",
    "protenix_has_clash",
]

BOLTZ_COMPAT_FIELDS = [
    "boltz_plddt",
    "boltz_pae",
    "boltz_ptm",
    "boltz_iptm",
    "boltz_complex_plddt",
    "boltz_complex_iplddt",
    "boltz_complex_pde",
    "boltz_complex_ipde",
    "boltz_ptmenergy",
]

INTERFACE_FIELDS = [
    "interface_binder_score",
    "interface_surface_hydrophobicity",
    "interface_interface_sc",
    "interface_interface_packstat",
    "interface_interface_dG",
    "interface_interface_dSASA",
    "interface_interface_dG_SASA_ratio",
    "interface_interface_fraction",
    "interface_interface_hydrophobicity",
    "interface_interface_nres",
    "interface_interface_interface_hbonds",
    "interface_interface_hbond_percentage",
    "interface_interface_delta_unsat_hbonds",
    "interface_interface_delta_unsat_hbonds_percentage",
    "interface_residues",
    "interface_amino_acids",
]


def passes_filters(row: dict[str, Any], filters: dict[str, Any], sample_metrics: dict[str, list[float]] | None = None) -> tuple[bool, str]:
    failed: list[str] = []
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
                failed.append(f"{key} missing")
            else:
                if higher and numeric < float(threshold):
                    failed.append(f"{key} {numeric} < {threshold}")
                if not higher and numeric > float(threshold):
                    failed.append(f"{key} {numeric} > {threshold}")

        average_threshold = rule.get("average_threshold")
        if average_threshold is not None:
            avg_key = f"{key}_avg"
            expected_values = int(row.get("protenix_sample_count", 0) or 0)
            values = (sample_metrics or {}).get(key, [])
            if expected_values and values and len(values) < expected_values:
                failed.append(f"{key} has {len(values)} numeric samples, expected {expected_values}")
                continue
            value = row.get(avg_key)
            try:
                numeric = float(value)
            except Exception:
                failed.append(f"{avg_key} missing")
            else:
                if higher and numeric < float(average_threshold):
                    failed.append(f"{avg_key} {numeric} < {average_threshold}")
                if not higher and numeric > float(average_threshold):
                    failed.append(f"{avg_key} {numeric} > {average_threshold}")

        all_threshold = rule.get("all_threshold")
        if all_threshold is not None:
            values = (sample_metrics or {}).get(key, [])
            if not values:
                failed.append(f"{key} samples missing")
                continue
            expected_values = int(row.get("protenix_sample_count", len(values)) or len(values))
            if len(values) < expected_values:
                failed.append(f"{key} has {len(values)} numeric samples, expected {expected_values}")
                continue
            bad_values = [value for value in values if value < float(all_threshold)] if higher else [
                value for value in values if value > float(all_threshold)
            ]
            if bad_values:
                op = ">=" if higher else "<="
                failed.append(f"{key} all samples not {op} {all_threshold}: {bad_values}")
    return (not failed, "; ".join(failed))


def protenix_filters(filters: dict[str, Any]) -> dict[str, Any]:
    return {key: rule for key, rule in filters.items() if key.startswith("protenix_")}


def non_protenix_filters(filters: dict[str, Any]) -> dict[str, Any]:
    return {key: rule for key, rule in filters.items() if not key.startswith("protenix_")}


def find_prediction_names(predictions_out: Path, seed: int) -> list[str]:
    names: set[str] = set()
    patterns = [
        f"protenix_output/*/seed_{seed}/predictions/*_summary_confidence_sample_*.json",
        f"*/seed_{seed}/predictions/*_summary_confidence_sample_*.json",
    ]
    for pattern in patterns:
        for json_path in predictions_out.glob(pattern):
            design_name = json_path.name.split("_summary_confidence_sample_", 1)[0]
            if find_protenix_predictions(predictions_out, design_name, seed=seed):
                names.add(design_name)
    return sorted(names)


def processed_names(csv_path: Path, rerun_failed: bool) -> set[str]:
    done = set()
    for row in read_csv_rows(csv_path):
        if rerun_failed and row.get("status", "").startswith("failed"):
            continue
        if row.get("design_name"):
            done.add(row["design_name"])
    return done


def add_protenix_summary_metrics(row: dict[str, Any], confidence: dict[str, Any]) -> None:
    mapping = {
        "plddt": "protenix_plddt",
        "gpde": "protenix_gpde",
        "ptm": "protenix_ptm",
        "iptm": "protenix_iptm",
        "ranking_score": "protenix_ranking_score",
        "has_clash": "protenix_has_clash",
    }
    for source, dest in mapping.items():
        if source in confidence:
            row[dest] = confidence[source]

    # Compatibility aliases keep older downstream CSV consumers useful, while
    # new filters should prefer the protenix_* fields above.
    if "plddt" in confidence:
        try:
            plddt = float(confidence["plddt"])
            row["boltz_plddt"] = round(plddt / 100.0 if plddt > 1.0 else plddt, 4)
        except (TypeError, ValueError):
            pass
    if "gpde" in confidence:
        row["boltz_pae"] = confidence["gpde"]
    if "ptm" in confidence:
        row["boltz_ptm"] = confidence["ptm"]
    if "iptm" in confidence:
        row["boltz_iptm"] = confidence["iptm"]


def numeric_confidence_metrics(confidence: dict[str, Any]) -> dict[str, float]:
    mapping = {
        "plddt": "protenix_plddt",
        "gpde": "protenix_gpde",
        "ptm": "protenix_ptm",
        "iptm": "protenix_iptm",
        "ranking_score": "protenix_ranking_score",
    }
    metrics: dict[str, float] = {}
    for source, dest in mapping.items():
        if source not in confidence:
            continue
        try:
            metrics[dest] = float(confidence[source])
        except (TypeError, ValueError):
            continue
    return metrics


def add_protenix_sample_aggregates(row: dict[str, Any], sample_metrics: dict[str, list[float]]) -> None:
    for key, values in sample_metrics.items():
        if not values:
            continue
        row[f"{key}_avg"] = round(sum(values) / len(values), 4)
        row[f"{key}_min"] = round(min(values), 4)
        row[f"{key}_max"] = round(max(values), 4)


def main() -> int:
    parser = argparse.ArgumentParser(description="Relax Fast Protenix predictions with OpenMM and screen interfaces.")
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

    fast_settings = settings.get("fast_protenix", {})
    fast_out = stage_dir(settings, "fast_protenix")
    seed = int(fast_settings.get("seed", (fast_settings.get("model_seeds") or [101])[0]))
    expected_sample_count = int(fast_settings["sample"]) if "sample" in fast_settings else None
    names = find_prediction_names(fast_out, seed)
    metrics_csv = out / "screening_metrics.csv"
    done = processed_names(metrics_csv, bool(screen.get("rerun_failed", False)))
    pending = [name for name in names if name not in done]
    if args.limit is not None:
        pending = pending[: max(0, int(args.limit))]
    print(f"OpenMM screening predictions={len(names)}, pending={len(pending)}")
    if not pending:
        return 0

    fields = BASE_FIELDS + PREDICTION_FIELDS + BOLTZ_COMPAT_FIELDS + INTERFACE_FIELDS
    binder_chain = screen.get("binder_chain", settings.get("binder_chain", "B"))
    target_chain = screen.get("target_chain", "A")
    filters = screen.get("filters", {})

    for design_name in pending:
        start = time.time()
        cif_path: Path | None = None
        conf_path: Path | None = None
        row: dict[str, Any] = {"design_name": design_name}
        try:
            predictions = find_protenix_predictions(fast_out, design_name, seed=seed)
            if not predictions:
                raise FileNotFoundError("Protenix CIF/confidence JSON not found")
            if expected_sample_count is not None and len(predictions) < expected_sample_count:
                raise FileNotFoundError(
                    f"Expected {expected_sample_count} Protenix samples, found {len(predictions)}"
                )

            sample_records: list[tuple[int, Path, Path, dict[str, Any], dict[str, float]]] = []
            sample_metrics: dict[str, list[float]] = {}
            for sample_index, sample_cif, sample_conf_path in predictions:
                with sample_conf_path.open() as handle:
                    sample_confidence = json.load(handle)
                numeric_metrics = numeric_confidence_metrics(sample_confidence)
                sample_records.append((sample_index, sample_cif, sample_conf_path, sample_confidence, numeric_metrics))
                for key, value in numeric_metrics.items():
                    sample_metrics.setdefault(key, []).append(value)

            selectable = [record for record in sample_records if "protenix_iptm" in record[4]]
            if selectable:
                selected = max(selectable, key=lambda record: record[4]["protenix_iptm"])
            else:
                selected = sample_records[0]
            selected_sample_index, cif_path, conf_path, confidence, _ = selected

            row["protenix_selected_sample_index"] = selected_sample_index
            row["protenix_sample_count"] = len(sample_records)
            add_protenix_sample_aggregates(row, sample_metrics)

            flat_conf: dict[str, Any] = {}
            flatten_json("protenix", confidence, flat_conf)
            row.update({key: value for key, value in flat_conf.items() if key in PREDICTION_FIELDS})
            add_protenix_summary_metrics(row, confidence)

            passed, reason = passes_filters(row, protenix_filters(filters), sample_metrics)
            if not passed:
                row.update(
                    {
                        "status": "filtered",
                        "sequence": "",
                        "input_cif": str(cif_path),
                        "unrelaxed_pdb": "",
                        "relaxed_pdb": "",
                        "unrelaxed_clashes": "",
                        "relaxed_clashes": "",
                        "reason": reason,
                    }
                )
                row["elapsed_seconds"] = round(time.time() - start, 2)
                append_csv(metrics_csv, row, fields)
                print(f"{design_name}: {row['status']} {row.get('reason', '')}")
                continue

            sequence = extract_chain_sequences(cif_path, [str(binder_chain)]).get(str(binder_chain), "")
            unrelaxed_pdb = cif_to_pdb(cif_path, pdb_dir / f"{design_name}_sample{selected_sample_index}.pdb")
            relaxed_pdb = relaxed_dir / f"{design_name}_sample{selected_sample_index}_openmm_relaxed.pdb"
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
                    "input_cif": str(cif_path),
                    "unrelaxed_pdb": str(unrelaxed_pdb),
                    "relaxed_pdb": str(relaxed_pdb),
                    "unrelaxed_clashes": unrelaxed_clashes,
                    "relaxed_clashes": relaxed_clashes,
                    "reason": "",
                }
            )
            passed, reason = passes_filters(row, non_protenix_filters(filters), sample_metrics)
            if not passed:
                row["status"] = "filtered"
                row["reason"] = reason
            else:
                shutil.copy2(relaxed_pdb, accepted_dir / relaxed_pdb.name)
        except Exception as exc:
            row.update(
                {
                    "status": "failed",
                    "sequence": row.get("sequence", ""),
                    "input_cif": str(cif_path or ""),
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
