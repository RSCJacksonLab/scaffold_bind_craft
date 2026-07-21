#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scaffold_bind_craft.config import load_settings, require, stage_dir
from scaffold_bind_craft.io_utils import append_csv, count_fasta_records, symlink_or_copy
from scaffold_bind_craft.pdb_utils import position_list_for_mpnn


FIELDS = ["pdb_name", "input_pdb", "status", "generated_sequences", "fasta_path", "reason"]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def completed_for_target(fasta_path: Path, required: int) -> bool:
    return count_fasta_records(fasta_path, skip_native=True) >= required


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ProteinMPNN on accepted RSO structures.")
    parser.add_argument("--settings", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int, default=None, help="Maximum pending PDBs to process in this run.")
    args = parser.parse_args()

    settings = load_settings(args.settings, args.output_dir)
    mpnn = require(settings, "proteinmpnn")
    out = stage_dir(settings, "proteinmpnn")
    seqs_dir = out / "seqs"
    staging_dir = out / "pending_pdbs"
    seqs_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    input_pdb_setting = mpnn.get("input_pdb")
    if input_pdb_setting:
        input_pdb = Path(str(input_pdb_setting)).expanduser().resolve()
        if not input_pdb.exists():
            raise FileNotFoundError(f"ProteinMPNN input PDB not found: {input_pdb}")
        pdbs = [input_pdb]
    else:
        rso_dir = stage_dir(settings, "rso") / "accepted_pdbs"
        input_dir = Path(mpnn.get("input_pdb_dir", rso_dir)).expanduser().resolve()
        pdbs = sorted(input_dir.glob("*.pdb"))
    required = int(mpnn.get("num_seq_per_target", 4))

    pending: list[Path] = []
    for pdb in pdbs:
        fasta_path = seqs_dir / f"{pdb.stem}.fa"
        if completed_for_target(fasta_path, required):
            continue
        pending.append(pdb)
    if args.limit is not None:
        pending = pending[: max(0, int(args.limit))]

    print(f"ProteinMPNN input_pdbs={len(pdbs)}, pending={len(pending)}, required_sequences={required}")
    if not pending:
        return 0

    shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    for pdb in pending:
        symlink_or_copy(pdb, staging_dir / pdb.name)

    proteinmpnn_path = Path(require(settings, "tools", "proteinmpnn_path")).expanduser().resolve()
    python_bin = mpnn.get("python", sys.executable)
    parsed = out / "parsed_pdbs.jsonl"
    assigned = out / "assigned_pdbs.jsonl"
    fixed = out / "fixed_pdbs.jsonl"
    chains = mpnn.get("chains_to_design", settings.get("binder_chain", "B"))
    design_positions = position_list_for_mpnn(mpnn.get("design_positions", settings.get("design_positions")))
    if not design_positions:
        raise ValueError("proteinmpnn.design_positions or top-level design_positions must specify relaxed positions")

    run([python_bin, str(proteinmpnn_path / "helper_scripts" / "parse_multiple_chains.py"), "--input_path", str(staging_dir), "--output_path", str(parsed)])
    run([python_bin, str(proteinmpnn_path / "helper_scripts" / "assign_fixed_chains.py"), "--input_path", str(parsed), "--output_path", str(assigned), "--chain_list", str(chains)])
    run(
        [
            python_bin,
            str(proteinmpnn_path / "helper_scripts" / "make_fixed_positions_dict.py"),
            "--input_path",
            str(parsed),
            "--output_path",
            str(fixed),
            "--chain_list",
            str(chains),
            "--position_list",
            design_positions,
            "--specify_non_fixed",
        ]
    )

    cmd = [
        python_bin,
        str(proteinmpnn_path / "protein_mpnn_run.py"),
        "--jsonl_path",
        str(parsed),
        "--chain_id_jsonl",
        str(assigned),
        "--fixed_positions_jsonl",
        str(fixed),
        "--out_folder",
        str(out),
        "--num_seq_per_target",
        str(required),
        "--sampling_temp",
        str(mpnn.get("sampling_temp", "0.1")),
        "--seed",
        str(mpnn.get("seed", 37)),
        "--batch_size",
        str(mpnn.get("batch_size", 1)),
        "--save_score",
        str(int(bool(mpnn.get("save_score", True)))),
        "--save_probs",
        str(int(bool(mpnn.get("save_probs", False)))),
    ]
    if mpnn.get("use_soluble_model", False):
        cmd.append("--use_soluble_model")
    if mpnn.get("model_name"):
        cmd.extend(["--model_name", str(mpnn["model_name"])])
    if mpnn.get("path_to_model_weights"):
        cmd.extend(["--path_to_model_weights", str(Path(mpnn["path_to_model_weights"]).expanduser().resolve())])
    if mpnn.get("omit_AAs"):
        cmd.extend(["--omit_AAs", str(mpnn["omit_AAs"])])
    run(cmd, cwd=proteinmpnn_path)

    metrics_csv = out / "proteinmpnn_metrics.csv"
    for pdb in pending:
        fasta_path = seqs_dir / f"{pdb.stem}.fa"
        generated = count_fasta_records(fasta_path, skip_native=True)
        status = "completed" if generated >= required else "partial"
        append_csv(
            metrics_csv,
            {
                "pdb_name": pdb.stem,
                "input_pdb": str(pdb),
                "status": status,
                "generated_sequences": generated,
                "fasta_path": str(fasta_path),
                "reason": "" if status == "completed" else f"expected {required}",
            },
            FIELDS,
        )

    manifest = out / "settings_used.json"
    manifest.write_text(json.dumps(mpnn, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
