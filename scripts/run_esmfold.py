#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scaffold_bind_craft.config import load_settings, require, stage_dir
from scaffold_bind_craft.esmfold_utils import (
    chain_id_list,
    find_esmfold_prediction,
    find_esmfold_predictions,
    plddt_metrics_from_pdb,
    remap_pdb_chain_ids,
    write_esmfold_fasta,
    write_metrics_json,
)
from scaffold_bind_craft.io_utils import append_csv, parse_fasta
from scaffold_bind_craft.pdb_utils import extract_chain_sequences


FIELDS = [
    "design_name",
    "status",
    "input_fasta_path",
    "pdb_path",
    "metrics_json",
    "esmfold_plddt",
    "esmfold_binder_plddt",
    "esmfold_target_plddt",
    "reason",
]


def sequence_inputs(settings: dict[str, Any]) -> dict[str, str]:
    esmfold = settings["esmfold"]
    if esmfold.get("target_sequences"):
        return {str(k): str(v) for k, v in esmfold["target_sequences"].items()}
    input_pdb = require(settings, "input_pdb")
    target_chains = chain_id_list(esmfold.get("target_chains", settings.get("target_chains", "A")))
    return dict(extract_chain_sequences(input_pdb, chains=target_chains))


def load_esmfold_model(esmfold: dict[str, Any]):
    try:
        import esm
        import torch
    except Exception as exc:
        raise RuntimeError("ESMFold requires fair-esm with the esmfold extra and torch installed") from exc

    gpu_device = esmfold.get("gpu_device")
    if esmfold.get("device"):
        device = str(esmfold["device"])
    elif gpu_device is not None and torch.cuda.is_available():
        device = f"cuda:{int(gpu_device)}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = esm.pretrained.esmfold_v1()
    model = model.eval()
    chunk_size = esmfold.get("chunk_size")
    if chunk_size is not None:
        model.set_chunk_size(int(chunk_size))
    model = model.to(device)
    return model, torch, device


def infer_pdb(model: Any, sequence: str, esmfold: dict[str, Any]) -> str:
    num_recycles = esmfold.get("num_recycles")
    if num_recycles is None:
        return model.infer_pdb(sequence)
    return model.infer_pdb(sequence, num_recycles=int(num_recycles))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pending ESMFold binder-target predictions.")
    parser.add_argument("--settings", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int, default=None, help="Maximum pending designs to predict in this run.")
    args = parser.parse_args()

    settings = load_settings(args.settings, args.output_dir)
    esmfold = require(settings, "esmfold")
    out = stage_dir(settings, "esmfold")
    input_dir = out / "inputs"
    predictions_dir = out / "predictions"
    metrics_dir = out / "metrics"
    for path in (input_dir, predictions_dir, metrics_dir):
        path.mkdir(parents=True, exist_ok=True)

    mpnn_out = stage_dir(settings, "proteinmpnn")
    fasta_dir = Path(esmfold.get("input_fasta_dir", mpnn_out / "seqs")).expanduser().resolve()
    fasta_files = sorted(fasta_dir.glob("*.fa"))
    if not fasta_files:
        print("ESMFold fasta_files=0, pending_predictions=0")
        return 0

    target_sequences = sequence_inputs(settings)
    target_chain_ids = list(target_sequences.keys())
    binder_chain = str(esmfold.get("binder_chain", settings.get("binder_chain", "B")))
    existing = {design_name for design_name, _, _ in find_esmfold_predictions(out)}

    designs: list[tuple[str, str, Path, dict[str, str]]] = []
    max_per_backbone = esmfold.get("max_sequences_per_backbone")
    for fasta in fasta_files:
        records = parse_fasta(fasta)
        for index, (header, seq) in enumerate(records[1:], start=1):
            if max_per_backbone is not None and index > int(max_per_backbone):
                break
            design_name = f"{fasta.stem}_mpnn{index}"
            if design_name in existing:
                continue
            input_path = input_dir / f"{design_name}.fa"
            sequences = dict(target_sequences)
            sequences[binder_chain] = seq
            write_esmfold_fasta(input_path, design_name, sequences)
            designs.append((design_name, header, input_path, sequences))

    if args.limit is not None:
        designs = designs[: max(0, int(args.limit))]
    print(f"ESMFold fasta_files={len(fasta_files)}, pending_predictions={len(designs)}")
    if not designs:
        return 0

    metrics_csv = out / "esmfold_metrics.csv"
    try:
        model, torch, device = load_esmfold_model(esmfold)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        for design_name, _, input_path, _ in designs:
            append_csv(
                metrics_csv,
                {
                    "design_name": design_name,
                    "status": "failed",
                    "input_fasta_path": str(input_path),
                    "pdb_path": "",
                    "metrics_json": "",
                    "reason": reason,
                },
                FIELDS,
            )
        return 1

    print(f"ESMFold device={device}")
    failed = False
    for design_name, _, input_path, sequences in designs:
        pdb_path = predictions_dir / f"{design_name}.pdb"
        metrics_path = metrics_dir / f"{design_name}.json"
        try:
            sequence = ":".join(sequences.values())
            with torch.no_grad():
                pdb_text = infer_pdb(model, sequence, esmfold)
            pdb_path.write_text(remap_pdb_chain_ids(pdb_text, sequences.keys()))
            metrics = plddt_metrics_from_pdb(
                pdb_path,
                binder_chain=binder_chain,
                target_chains=target_chain_ids,
            )
            write_metrics_json(metrics_path, metrics)
            found_pdb, found_metrics = find_esmfold_prediction(out, design_name)
            status = "completed" if found_pdb and found_metrics else "missing_outputs"
            reason = "" if status == "completed" else "PDB or metrics JSON was not found after ESMFold"
            append_csv(
                metrics_csv,
                {
                    "design_name": design_name,
                    "status": status,
                    "input_fasta_path": str(input_path),
                    "pdb_path": str(found_pdb or ""),
                    "metrics_json": str(found_metrics or ""),
                    "esmfold_plddt": metrics.get("plddt"),
                    "esmfold_binder_plddt": metrics.get("binder_plddt"),
                    "esmfold_target_plddt": metrics.get("target_plddt"),
                    "reason": reason,
                },
                FIELDS,
            )
            print(f"{design_name}: {status} {reason}")
        except Exception as exc:
            failed = True
            append_csv(
                metrics_csv,
                {
                    "design_name": design_name,
                    "status": "failed",
                    "input_fasta_path": str(input_path),
                    "pdb_path": str(pdb_path if pdb_path.exists() else ""),
                    "metrics_json": str(metrics_path if metrics_path.exists() else ""),
                    "reason": f"{type(exc).__name__}: {exc}",
                },
                FIELDS,
            )
            print(f"{design_name}: failed {type(exc).__name__}: {exc}")
        finally:
            if device != "cpu" and hasattr(torch, "cuda"):
                torch.cuda.empty_cache()
            time.sleep(float(esmfold.get("sleep_seconds", 0)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
