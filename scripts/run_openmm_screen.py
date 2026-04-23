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
from scaffold_bind_craft.fast_protenix_utils import find_protenix_prediction
from scaffold_bind_craft.io_utils import append_csv, flatten_json, read_csv_rows
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
    "protenix_plddt",
    "protenix_gpde",
    "protenix_ptm",
    "protenix_iptm",
    "protenix_ranking_score",
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


def add_freebindcraft_to_path(settings: dict[str, Any]) -> None:
    path = require(settings, "tools", "freebindcraft_path")
    sys.path.insert(0, str(Path(path).expanduser().resolve()))


def passes_filters(row: dict[str, Any], filters: dict[str, Any]) -> tuple[bool, str]:
    failed: list[str] = []
    for key, rule in filters.items():
        if not isinstance(rule, dict):
            continue
        threshold = rule.get("threshold")
        if threshold is None:
            continue
        value = row.get(key)
        try:
            numeric = float(value)
        except Exception:
            failed.append(f"{key} missing")
            continue
        higher = bool(rule.get("higher", True))
        if higher and numeric < float(threshold):
            failed.append(f"{key} {numeric} < {threshold}")
        if not higher and numeric > float(threshold):
            failed.append(f"{key} {numeric} > {threshold}")
    return (not failed, "; ".join(failed))


def find_prediction_names(predictions_out: Path, seed: int, sample_index: int) -> list[str]:
    names: set[str] = set()
    pattern = f"protenix_output/*/seed_{seed}/predictions/*_summary_confidence_sample_{sample_index}.json"
    for json_path in predictions_out.glob(pattern):
        suffix = f"_summary_confidence_sample_{sample_index}.json"
        design_name = json_path.name[: -len(suffix)]
        cif_path, conf_path = find_protenix_prediction(predictions_out, design_name, seed=seed, sample_index=sample_index)
        if cif_path and conf_path:
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
    sample_index = int(fast_settings.get("sample_index", 0))
    names = find_prediction_names(fast_out, seed, sample_index)
    metrics_csv = out / "screening_metrics.csv"
    done = processed_names(metrics_csv, bool(screen.get("rerun_failed", False)))
    pending = [name for name in names if name not in done]
    if args.limit is not None:
        pending = pending[: max(0, int(args.limit))]
    print(f"OpenMM screening predictions={len(names)}, pending={len(pending)}")
    if not pending:
        return 0

    add_freebindcraft_to_path(settings)
    from functions.biopython_utils import calculate_clash_score
    from functions.pr_alternative_utils import openmm_relax_subprocess, pr_alternative_score_interface

    fields = BASE_FIELDS + PREDICTION_FIELDS + BOLTZ_COMPAT_FIELDS + INTERFACE_FIELDS
    binder_chain = screen.get("binder_chain", settings.get("binder_chain", "B"))
    target_chain = screen.get("target_chain", "A")
    filters = screen.get("filters", {})

    for design_name in pending:
        start = time.time()
        cif_path, conf_path = find_protenix_prediction(fast_out, design_name, seed=seed, sample_index=sample_index)
        row: dict[str, Any] = {"design_name": design_name}
        try:
            if not cif_path or not conf_path:
                raise FileNotFoundError("Protenix CIF/confidence JSON not found")
            with Path(conf_path).open() as handle:
                confidence = json.load(handle)
            flat_conf: dict[str, Any] = {}
            flatten_json("protenix", confidence, flat_conf)
            row.update({key: value for key, value in flat_conf.items() if key in PREDICTION_FIELDS})
            add_protenix_summary_metrics(row, confidence)

            sequence = extract_chain_sequences(cif_path, [str(binder_chain)]).get(str(binder_chain), "")
            unrelaxed_pdb = cif_to_pdb(cif_path, pdb_dir / f"{design_name}.pdb")
            relaxed_pdb = relaxed_dir / f"{design_name}_openmm_relaxed.pdb"
            if not relaxed_pdb.exists() or relaxed_pdb.stat().st_size == 0:
                openmm_relax_subprocess(
                    str(unrelaxed_pdb),
                    str(relaxed_pdb),
                    use_gpu_relax=bool(screen.get("use_gpu_relax", True)),
                    timeout=screen.get("openmm_timeout_seconds"),
                    max_attempts=int(screen.get("openmm_max_attempts", 3)),
                    use_faspr_repack=bool(screen.get("use_faspr_repack", True)),
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
            passed, reason = passes_filters(row, filters)
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
