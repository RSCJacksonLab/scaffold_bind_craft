#!/usr/bin/env python3
"""Alternative stage 3: fold ProteinMPNN designs with local ESMFold2."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scaffold_bind_craft.chai1_utils import chain_id_list, cif_to_pdb, write_metrics_json
from scaffold_bind_craft.config import load_settings, require, stage_dir
from scaffold_bind_craft.esmfold2_utils import (
    confidence_metrics,
    find_esmfold2_prediction,
    find_esmfold2_predictions,
    validate_openmm_pdb,
)
from scaffold_bind_craft.io_utils import append_csv, ensure_csv_fields, parse_fasta, read_csv_rows
from scaffold_bind_craft.multimer_clash import check_configured_multimer_clashes
from scaffold_bind_craft.pdb_utils import extract_chain_sequences

FIELDS = [
    "design_name", "status", "pdb_path", "cif_path", "metrics_json",
    "esmfold2_mean_plddt", "esmfold2_binder_mean_plddt", "esmfold2_target_mean_plddt",
    "esmfold2_ptm", "esmfold2_iptm", "esmfold2_binder_target_iptm",
    "esmfold2_mean_pae", "esmfold2_binder_mean_pae", "esmfold2_target_mean_pae",
    "esmfold2_interface_mean_pae",
    "multimer_clash_count", "multimer_clashing_chains", "multimer_alignment_rmsd", "reason",
]


def target_sequences(settings: dict[str, Any]) -> dict[str, str]:
    config = settings["esmfold2"]
    if config.get("target_sequences"):
        return {str(k): str(v) for k, v in config["target_sequences"].items()}
    chains = chain_id_list(config.get("target_chains", settings.get("target_chains", "A")))
    return dict(extract_chain_sequences(require(settings, "input_pdb"), chains))


def load_model(config: dict[str, Any]):
    esm_path = Path(config.get("source_path", "/Users/u7282763/repos/esm")).expanduser().resolve()
    if not esm_path.is_dir():
        raise FileNotFoundError(f"ESMFold2 source_path does not exist: {esm_path}")
    sys.path.insert(0, str(esm_path))
    try:
        from esm.models.esmfold2 import EsmFold2Model
    except Exception as exc:
        raise RuntimeError(f"Could not import ESMFold2 from {esm_path}; install that checkout's dependencies") from exc
    model_name = str(config.get("model", "biohub/ESMFold2"))
    device = str(config.get("device", f"cuda:{int(config.get('gpu_device', 0))}"))
    esmc_precision = str(config.get("esmc_precision", "bf16"))
    model = EsmFold2Model.from_pretrained(
        model_name,
        device=device,
        esmc_precision=esmc_precision,
    )
    return model.eval(), device, model_name


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pending ESMFold2 binder-target predictions (alternative stage 3).")
    parser.add_argument("--settings", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    settings = load_settings(args.settings, args.output_dir)
    config = require(settings, "esmfold2")
    if config.get("offline", False):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    out = stage_dir(settings, "esmfold2")
    metrics_csv = out / "esmfold2_metrics.csv"
    ensure_csv_fields(metrics_csv, FIELDS)
    for name in ("cifs", "predictions", "filtered_pdbs", "metrics"):
        (out / name).mkdir(parents=True, exist_ok=True)

    fasta_dir = Path(config.get("input_fasta_dir", stage_dir(settings, "proteinmpnn") / "seqs")).expanduser().resolve()
    targets = target_sequences(settings)
    binder = str(config.get("binder_chain", settings.get("binder_chain", "B")))
    existing = {name for name, _, _ in find_esmfold2_predictions(out)}
    existing.update(row["design_name"] for row in read_csv_rows(metrics_csv) if row.get("status") == "filtered_multimer_clash")
    seen: set[str] = set()
    designs: list[tuple[str, dict[str, str]]] = []
    maximum = config.get("max_sequences_per_backbone")
    for fasta in sorted(fasta_dir.glob("*.fa")):
        for index, (_, sequence) in enumerate(parse_fasta(fasta)[1:], 1):
            if maximum is not None and index > int(maximum): break
            name = f"{fasta.stem}_mpnn{index}"
            if name in existing or sequence.upper() in seen: continue
            seen.add(sequence.upper())
            designs.append((name, {**targets, binder: sequence}))
    if args.limit is not None: designs = designs[:max(0, args.limit)]
    print(f"ESMFold2 fasta_files={len(list(fasta_dir.glob('*.fa')))}, pending_predictions={len(designs)}")
    if not designs: return 0

    model, device, model_name = load_model(config)
    from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput
    builder = ESMFold2InputBuilder()
    failed = False
    for name, sequences in designs:
        cif, pdb, metrics_path = out / "cifs" / f"{name}.cif", out / "predictions" / f"{name}.pdb", out / "metrics" / f"{name}.json"
        try:
            spi = StructurePredictionInput(sequences=[ProteinInput(id=chain, sequence=seq) for chain, seq in sequences.items()])
            result = builder.fold(model, spi, num_loops=int(config.get("num_loops", 20)), num_sampling_steps=int(config.get("num_sampling_steps", 100)), num_diffusion_samples=1, seed=config.get("seed", 0), lm_dropout=config.get("lm_dropout", 0.3), complex_id=name)
            cif.write_text(result.complex.to_mmcif())
            cif_to_pdb(cif, pdb)
            validate_openmm_pdb(pdb, sequences)
            metrics = confidence_metrics(result, sequences, binder)
            metrics.update({"model": model_name, "device": device, "single_sequence_mode": True, "num_loops": int(config.get("num_loops", 20)), "num_sampling_steps": int(config.get("num_sampling_steps", 100))})
            clash = check_configured_multimer_clashes(pdb, settings)
            if clash:
                metrics.update({"multimer_clash_count": clash.clash_count, "multimer_clashing_chains": list(clash.clashing_chains), "multimer_alignment_rmsd": clash.alignment_rmsd})
            write_metrics_json(metrics_path, metrics)
            status, reason, final_pdb = "completed", "", pdb
            if clash and clash.clashes:
                final_pdb = out / "filtered_pdbs" / pdb.name; pdb.replace(final_pdb)
                status, reason = "filtered_multimer_clash", f"binder clashes with reference chain(s) {','.join(clash.clashing_chains)} at {clash.clash_count} atom(s)"
            found_pdb, found_metrics = (final_pdb, metrics_path) if status.startswith("filtered") else find_esmfold2_prediction(out, name)
            row = {"design_name": name, "status": status, "pdb_path": str(found_pdb or ""), "cif_path": str(cif), "metrics_json": str(found_metrics or ""), "multimer_clash_count": clash.clash_count if clash else None, "multimer_clashing_chains": ",".join(clash.clashing_chains) if clash else "", "multimer_alignment_rmsd": clash.alignment_rmsd if clash else None, "reason": reason}
            for metric in ("mean_plddt", "binder_mean_plddt", "target_mean_plddt", "ptm", "iptm", "binder_target_iptm", "mean_pae", "binder_mean_pae", "target_mean_pae", "interface_mean_pae"):
                row[f"esmfold2_{metric}"] = metrics.get(metric)
            append_csv(metrics_csv, row, FIELDS)
            print(f"{name}: {status} {reason}")
        except Exception as exc:
            failed = True
            append_csv(metrics_csv, {"design_name": name, "status": "failed", "pdb_path": str(pdb if pdb.exists() else ""), "cif_path": str(cif if cif.exists() else ""), "metrics_json": str(metrics_path if metrics_path.exists() else ""), "reason": f"{type(exc).__name__}: {exc}"}, FIELDS)
            print(f"{name}: failed {type(exc).__name__}: {exc}")
        finally: time.sleep(float(config.get("sleep_seconds", 0)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
