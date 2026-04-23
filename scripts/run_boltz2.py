#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scaffold_bind_craft.boltz_utils import find_boltz_prediction, write_boltz_yaml
from scaffold_bind_craft.config import load_settings, require, stage_dir
from scaffold_bind_craft.io_utils import append_csv, parse_fasta
from scaffold_bind_craft.pdb_utils import extract_chain_sequences


FIELDS = ["design_name", "status", "yaml_path", "cif_path", "confidence_json", "reason"]


def sequence_inputs(settings: dict) -> dict[str, str]:
    boltz = settings["boltz2"]
    if boltz.get("target_sequences"):
        return {str(k): str(v) for k, v in boltz["target_sequences"].items()}
    input_pdb = require(settings, "input_pdb")
    target_chains = boltz.get("target_chains", settings.get("target_chains", "A"))
    if isinstance(target_chains, str):
        chains = [part.strip() for part in target_chains.replace(",", " ").split() if part.strip()]
    else:
        chains = list(target_chains)
    return dict(extract_chain_sequences(input_pdb, chains=chains))


def find_existing_names(boltz_dir: Path, model_index: int) -> set[str]:
    names = set()
    for json_path in boltz_dir.glob(f"boltz_results_*/predictions/*/confidence_*_model_{model_index}.json"):
        design_name = json_path.parent.name
        cif_path, conf_path = find_boltz_prediction(boltz_dir, design_name, model_index=model_index)
        if cif_path and conf_path:
            names.add(design_name)
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Boltz-2 YAMLs and run pending predictions.")
    parser.add_argument("--settings", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int, default=None, help="Maximum pending designs to predict in this run.")
    args = parser.parse_args()

    settings = load_settings(args.settings, args.output_dir)
    boltz = require(settings, "boltz2")
    out = stage_dir(settings, "boltz2")
    yaml_dir = out / "yaml"
    predictions_dir = out / "predictions"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    mpnn_out = stage_dir(settings, "proteinmpnn")
    fasta_dir = Path(boltz.get("input_fasta_dir", mpnn_out / "seqs")).expanduser().resolve()
    fasta_files = sorted(fasta_dir.glob("*.fa"))
    if not fasta_files:
        print(f"Boltz-2 fasta_files=0, pending_predictions=0")
        return 0

    target_sequences = sequence_inputs(settings)
    binder_chain = boltz.get("binder_chain", settings.get("binder_chain", "B"))
    model_index = int(boltz.get("model_index", 0))
    existing = find_existing_names(predictions_dir, model_index)

    designs: list[tuple[str, str, Path]] = []
    max_per_backbone = boltz.get("max_sequences_per_backbone")
    for fasta in fasta_files:
        records = parse_fasta(fasta)
        for index, (header, seq) in enumerate(records[1:], start=1):
            if max_per_backbone is not None and index > int(max_per_backbone):
                break
            design_name = f"{fasta.stem}_mpnn{index}"
            if design_name in existing:
                continue
            yaml_path = yaml_dir / f"{design_name}.yaml"
            sequences = dict(target_sequences)
            sequences[str(binder_chain)] = seq
            write_boltz_yaml(
                yaml_path,
                sequences,
                msa_paths=boltz.get("msa_paths", {}),
                binder_chain=str(binder_chain),
                constraints=boltz.get("constraints"),
                templates=boltz.get("templates"),
                properties=boltz.get("properties"),
            )
            designs.append((design_name, header, yaml_path))

    if args.limit is not None:
        designs = designs[: max(0, int(args.limit))]
    print(f"Boltz-2 fasta_files={len(fasta_files)}, pending_predictions={len(designs)}")
    if not designs:
        return 0

    metrics_csv = out / "boltz2_metrics.csv"
    boltz_exe = boltz.get("executable", "boltz")
    extra_args = [str(x) for x in boltz.get("extra_args", [])]
    cache = boltz.get("cache")
    for design_name, _, yaml_path in designs:
        cmd = [boltz_exe, "predict", str(yaml_path), "--out_dir", str(predictions_dir)]
        if cache:
            cmd.extend(["--cache", str(Path(cache).expanduser())])
        cmd.extend(extra_args)
        print(" ".join(cmd))
        try:
            subprocess.run(cmd, check=True, cwd=str(predictions_dir))
            cif_path, json_path = find_boltz_prediction(predictions_dir, design_name, model_index=model_index)
            status = "completed" if cif_path and json_path else "missing_outputs"
            reason = "" if status == "completed" else "confidence JSON or CIF was not found after boltz predict"
        except Exception as exc:
            cif_path, json_path = None, None
            status = "failed"
            reason = f"{type(exc).__name__}: {exc}"
        append_csv(
            metrics_csv,
            {
                "design_name": design_name,
                "status": status,
                "yaml_path": str(yaml_path),
                "cif_path": str(cif_path or ""),
                "confidence_json": str(json_path or ""),
                "reason": reason,
            },
            FIELDS,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
