#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scaffold_bind_craft.chai1_utils import (
    add_chai1_scores,
    chain_id_list,
    cif_to_pdb,
    copy_best_candidate,
    find_chai1_prediction,
    find_chai1_predictions,
    plddt_metrics_from_pdb,
    write_chai1_fasta,
    write_metrics_json,
)
from scaffold_bind_craft.config import load_settings, require, stage_dir
from scaffold_bind_craft.io_utils import append_csv, parse_fasta
from scaffold_bind_craft.pdb_utils import extract_chain_sequences


FIELDS = [
    "design_name",
    "status",
    "input_fasta_path",
    "pdb_path",
    "cif_path",
    "metrics_json",
    "chai1_plddt",
    "chai1_binder_plddt",
    "chai1_target_plddt",
    "chai1_aggregate_score",
    "reason",
]


def sequence_inputs(settings: dict[str, Any]) -> dict[str, str]:
    chai1 = settings["chai1"]
    if chai1.get("target_sequences"):
        return {str(k): str(v) for k, v in chai1["target_sequences"].items()}
    input_pdb = require(settings, "input_pdb")
    target_chains = chain_id_list(chai1.get("target_chains", settings.get("target_chains", "A")))
    return dict(extract_chain_sequences(input_pdb, chains=target_chains))


def chai1_device(chai1: dict[str, Any]) -> str:
    if chai1.get("device"):
        return str(chai1["device"])
    return f"cuda:{int(chai1.get('gpu_device', 0))}"


def configure_model_paths(chai1: dict[str, Any]) -> Path:
    model_dir = Path(os.environ.get("CHAI_MODEL_DIR", chai1.get("model_dir", "/scratch/xz2/chai-1_weights/"))).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    hf_home = model_dir / "huggingface"
    os.environ["CHAI_DOWNLOADS_DIR"] = str(model_dir)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    if bool(chai1.get("offline", True)):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return model_dir


def run_inference(input_path: Path, raw_output_dir: Path, chai1: dict[str, Any]):
    try:
        from chai_lab.chai1 import run_inference as chai_run_inference
    except Exception as exc:
        raise RuntimeError(
            "Chai-1 offline requires chai_lab from git@github.com:RSCJacksonLab/chai-1-offline.git "
            "and its PyTorch/CUDA dependencies"
        ) from exc

    if raw_output_dir.exists():
        shutil.rmtree(raw_output_dir)
    raw_output_dir.mkdir(parents=True)

    return chai_run_inference(
        fasta_file=input_path,
        output_dir=raw_output_dir,
        use_esm_embeddings=bool(chai1.get("use_esm_embeddings", True)),
        use_msa_server=False,
        msa_directory=None,
        use_templates_server=False,
        template_hits_path=None,
        num_trunk_recycles=int(chai1.get("num_trunk_recycles", 3)),
        num_diffn_timesteps=int(chai1.get("num_diffn_timesteps", 200)),
        num_diffn_samples=int(chai1.get("num_diffn_samples", 5)),
        num_trunk_samples=int(chai1.get("num_trunk_samples", 1)),
        seed=chai1.get("seed"),
        device=chai1_device(chai1),
        low_memory=bool(chai1.get("low_memory", True)),
        fasta_names_as_cif_chains=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pending Chai-1 offline binder-target predictions.")
    parser.add_argument("--settings", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int, default=None, help="Maximum pending designs to predict in this run.")
    args = parser.parse_args()

    settings = load_settings(args.settings, args.output_dir)
    chai1 = require(settings, "chai1")
    model_dir = configure_model_paths(chai1)
    out = stage_dir(settings, "chai1")
    input_dir = out / "inputs"
    raw_dir = out / "raw_outputs"
    cif_dir = out / "cifs"
    predictions_dir = out / "predictions"
    metrics_dir = out / "metrics"
    for path in (input_dir, raw_dir, cif_dir, predictions_dir, metrics_dir):
        path.mkdir(parents=True, exist_ok=True)

    mpnn_out = stage_dir(settings, "proteinmpnn")
    fasta_dir = Path(chai1.get("input_fasta_dir", mpnn_out / "seqs")).expanduser().resolve()
    fasta_files = sorted(fasta_dir.glob("*.fa"))
    if not fasta_files:
        print("Chai-1 fasta_files=0, pending_predictions=0")
        return 0

    target_sequences = sequence_inputs(settings)
    target_chain_ids = list(target_sequences.keys())
    binder_chain = str(chai1.get("binder_chain", settings.get("binder_chain", "B")))
    existing = {design_name for design_name, _, _ in find_chai1_predictions(out)}

    designs: list[tuple[str, str, Path, dict[str, str]]] = []
    max_per_backbone = chai1.get("max_sequences_per_backbone")
    for fasta in fasta_files:
        records = parse_fasta(fasta)
        for index, (header, seq) in enumerate(records[1:], start=1):
            if max_per_backbone is not None and index > int(max_per_backbone):
                break
            design_name = f"{fasta.stem}_mpnn{index}"
            if design_name in existing:
                continue
            input_path = input_dir / f"{design_name}.fasta"
            sequences = dict(target_sequences)
            sequences[binder_chain] = seq
            write_chai1_fasta(input_path, sequences)
            designs.append((design_name, header, input_path, sequences))

    if args.limit is not None:
        designs = designs[: max(0, int(args.limit))]
    print(f"Chai-1 fasta_files={len(fasta_files)}, pending_predictions={len(designs)}, model_dir={model_dir}")
    if not designs:
        return 0

    metrics_csv = out / "chai1_metrics.csv"
    failed = False
    for design_name, _, input_path, _ in designs:
        raw_output_dir = raw_dir / design_name
        cif_path = cif_dir / f"{design_name}.cif"
        pdb_path = predictions_dir / f"{design_name}.pdb"
        metrics_path = metrics_dir / f"{design_name}.json"
        try:
            candidates = run_inference(input_path, raw_output_dir, chai1)
            best_index, source_cif, aggregate_score = copy_best_candidate(
                list(candidates.cif_paths),
                list(candidates.ranking_data),
                cif_path,
            )
            cif_to_pdb(cif_path, pdb_path)
            metrics = plddt_metrics_from_pdb(
                pdb_path,
                binder_chain=binder_chain,
                target_chains=target_chain_ids,
            )
            metrics.update(
                {
                    "best_model_index": best_index,
                    "source_cif": str(source_cif),
                    "aggregate_score": aggregate_score,
                    "single_sequence_mode": True,
                    "use_esm_embeddings": bool(chai1.get("use_esm_embeddings", True)),
                    "use_msa_server": False,
                    "use_templates_server": False,
                }
            )
            add_chai1_scores(metrics, raw_output_dir, best_index)
            write_metrics_json(metrics_path, metrics)
            found_pdb, found_metrics = find_chai1_prediction(out, design_name)
            status = "completed" if found_pdb and found_metrics else "missing_outputs"
            reason = "" if status == "completed" else "PDB or metrics JSON was not found after Chai-1"
            append_csv(
                metrics_csv,
                {
                    "design_name": design_name,
                    "status": status,
                    "input_fasta_path": str(input_path),
                    "pdb_path": str(found_pdb or ""),
                    "cif_path": str(cif_path if cif_path.exists() else ""),
                    "metrics_json": str(found_metrics or ""),
                    "chai1_plddt": metrics.get("plddt"),
                    "chai1_binder_plddt": metrics.get("binder_plddt"),
                    "chai1_target_plddt": metrics.get("target_plddt"),
                    "chai1_aggregate_score": metrics.get("aggregate_score"),
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
                    "cif_path": str(cif_path if cif_path.exists() else ""),
                    "metrics_json": str(metrics_path if metrics_path.exists() else ""),
                    "reason": f"{type(exc).__name__}: {exc}",
                },
                FIELDS,
            )
            print(f"{design_name}: failed {type(exc).__name__}: {exc}")
        finally:
            time.sleep(float(chai1.get("sleep_seconds", 0)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
