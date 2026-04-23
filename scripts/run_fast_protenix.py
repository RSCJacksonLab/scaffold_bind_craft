#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scaffold_bind_craft.config import load_settings, require, stage_dir
from scaffold_bind_craft.fast_protenix_utils import find_protenix_prediction, write_af3_json
from scaffold_bind_craft.io_utils import append_csv, parse_fasta
from scaffold_bind_craft.pdb_utils import extract_chain_sequences


FIELDS = ["design_name", "status", "input_json_path", "cif_path", "confidence_json", "reason"]


def sequence_inputs(settings: dict[str, Any]) -> dict[str, str]:
    fast = settings["fast_protenix"]
    if fast.get("target_sequences"):
        return {str(k): str(v) for k, v in fast["target_sequences"].items()}
    input_pdb = require(settings, "input_pdb")
    target_chains = fast.get("target_chains", settings.get("target_chains", "A"))
    if isinstance(target_chains, str):
        chains = [part.strip() for part in target_chains.replace(",", " ").split() if part.strip()]
    else:
        chains = list(target_chains)
    return dict(extract_chain_sequences(input_pdb, chains=chains))


def find_existing_names(stage_out: Path, seed: int, sample_index: int) -> set[str]:
    names = set()
    pattern = f"protenix_output/*/seed_{seed}/predictions/*_summary_confidence_sample_{sample_index}.json"
    for json_path in stage_out.glob(pattern):
        filename = json_path.name
        suffix = f"_summary_confidence_sample_{sample_index}.json"
        design_name = filename[: -len(suffix)]
        cif_path, conf_path = find_protenix_prediction(stage_out, design_name, seed=seed, sample_index=sample_index)
        if cif_path and conf_path:
            names.add(design_name)
    return names


def fast_protenix_env(fast: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    repo_path = Path(fast.get("path", "../fast_protenix")).expanduser().resolve()
    paths = [str(repo_path)]
    existing = env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def add_repeated_arg(cmd: list[str], option: str, values: list[Any]) -> None:
    for value in values:
        cmd.append(f"{option}={value}")


def has_cli_option(values: list[str], *options: str) -> bool:
    prefixes = tuple(f"{option}=" for option in options)
    return any(value in options or value.startswith(prefixes) for value in values)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate AF3 JSONs and run pending Fast Protenix predictions.")
    parser.add_argument("--settings", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int, default=None, help="Maximum pending designs to predict in this run.")
    args = parser.parse_args()

    settings = load_settings(args.settings, args.output_dir)
    fast = require(settings, "fast_protenix")
    out = stage_dir(settings, "fast_protenix")
    input_dir = out / "af3_inputs"
    predictions_dir = out
    input_dir.mkdir(parents=True, exist_ok=True)

    mpnn_out = stage_dir(settings, "proteinmpnn")
    fasta_dir = Path(fast.get("input_fasta_dir", mpnn_out / "seqs")).expanduser().resolve()
    fasta_files = sorted(fasta_dir.glob("*.fa"))
    if not fasta_files:
        print("Fast Protenix fasta_files=0, pending_predictions=0")
        return 0

    target_sequences = sequence_inputs(settings)
    binder_chain = fast.get("binder_chain", settings.get("binder_chain", "B"))
    seed = int(fast.get("seed", (fast.get("model_seeds") or [101])[0]))
    sample_index = int(fast.get("sample_index", 0))
    model_seeds = [int(value) for value in (fast.get("model_seeds") or [seed])]
    existing = find_existing_names(predictions_dir, seed, sample_index)

    designs: list[tuple[str, str, Path, dict[str, str]]] = []
    max_per_backbone = fast.get("max_sequences_per_backbone")
    for fasta in fasta_files:
        records = parse_fasta(fasta)
        for index, (header, seq) in enumerate(records[1:], start=1):
            if max_per_backbone is not None and index > int(max_per_backbone):
                break
            design_name = f"{fasta.stem}_mpnn{index}"
            if design_name in existing:
                continue
            input_path = input_dir / f"{design_name}.json"
            sequences = dict(target_sequences)
            sequences[str(binder_chain)] = seq
            write_af3_json(input_path, design_name, sequences, model_seeds=model_seeds)
            designs.append((design_name, header, input_path, sequences))

    if args.limit is not None:
        designs = designs[: max(0, int(args.limit))]
    print(f"Fast Protenix fasta_files={len(fasta_files)}, pending_predictions={len(designs)}")
    if not designs:
        return 0

    batch_dir = out / "pending_inputs" / f"batch_{int(time.time())}_{os.getpid()}"
    for design_name, _, _, sequences in designs:
        write_af3_json(batch_dir / f"{design_name}.json", design_name, sequences, model_seeds=model_seeds)

    python_exe = fast.get("python", sys.executable)
    cmd = [
        str(python_exe),
        "-m",
        "fast_protenix.cli",
        "run",
        "--input_dir",
        str(batch_dir),
        "--output_dir",
        str(out),
        "--db_dir",
        str(Path(require(fast, "db_dir")).expanduser().resolve()),
        "--mmseqs_db_dir",
        str(Path(require(fast, "mmseqs_db_dir")).expanduser().resolve()),
        "--alphafast_dir",
        str(Path(fast.get("alphafast_dir", "../alphafast")).expanduser().resolve()),
        "--batch_size",
        str(int(fast.get("batch_size", 512))),
    ]
    if fast.get("gpu_device") is not None:
        cmd.extend(["--gpu_device", str(int(fast["gpu_device"]))])
    if fast.get("temp_dir"):
        cmd.extend(["--temp_dir", str(Path(fast["temp_dir"]).expanduser().resolve())])
    if fast.get("run_protenix", True):
        cmd.append("--run_protenix")
    cmd.extend(["--protenix_bin", str(fast.get("protenix_bin", "protenix"))])
    if fast.get("protenix_model"):
        cmd.extend(["--protenix_model", str(fast["protenix_model"])])
    add_repeated_arg(cmd, "--alphafast_arg", [str(x) for x in fast.get("alphafast_args", [])])
    protenix_args = [str(x) for x in fast.get("protenix_args", [])]
    if not has_cli_option(protenix_args, "--seeds", "-s"):
        protenix_args.extend(["--seeds", str(seed)])
    if "sample" in fast and not has_cli_option(protenix_args, "--sample", "-e"):
        protenix_args.extend(["--sample", str(int(fast["sample"]))])
    add_repeated_arg(cmd, "--protenix_arg", protenix_args)

    metrics_csv = out / "fast_protenix_metrics.csv"
    print(" ".join(cmd))
    run_reason = ""
    try:
        subprocess.run(cmd, check=True, cwd=str(out), env=fast_protenix_env(fast))
    except Exception as exc:
        run_reason = f"{type(exc).__name__}: {exc}"

    for design_name, _, input_path, _ in designs:
        cif_path, json_path = find_protenix_prediction(predictions_dir, design_name, seed=seed, sample_index=sample_index)
        if cif_path and json_path:
            status = "completed"
            reason = ""
        elif run_reason:
            status = "failed"
            reason = run_reason
        else:
            status = "missing_outputs"
            reason = "Protenix summary confidence JSON or CIF was not found after fast-protenix run"
        append_csv(
            metrics_csv,
            {
                "design_name": design_name,
                "status": status,
                "input_json_path": str(input_path),
                "cif_path": str(cif_path or ""),
                "confidence_json": str(json_path or ""),
                "reason": reason,
            },
            FIELDS,
        )
    return 1 if run_reason else 0


if __name__ == "__main__":
    raise SystemExit(main())
