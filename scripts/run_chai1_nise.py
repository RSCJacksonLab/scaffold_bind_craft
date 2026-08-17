#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scaffold_bind_craft.chai1_utils import (
    add_chai1_error_metrics,
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
from scaffold_bind_craft.io_utils import (
    append_csv,
    count_fasta_records,
    ensure_csv_fields,
    parse_fasta,
    read_csv_rows,
    symlink_or_copy,
)
from scaffold_bind_craft.pdb_utils import extract_chain_sequences, parse_position_string, position_list_for_mpnn
from scaffold_bind_craft.multimer_clash import check_configured_multimer_clashes
from scripts.run_chai1_offline import configure_model_paths, run_inference, sequence_inputs


FIELDS = [
    "round",
    "design_name",
    "rso_name",
    "rso_pdb",
    "parent_name",
    "parent_pdb",
    "parent_score",
    "status",
    "input_fasta_path",
    "pdb_path",
    "cif_path",
    "metrics_json",
    "chai1_plddt",
    "chai1_binder_plddt",
    "chai1_target_plddt",
    "chai1_aggregate_score",
    "chai1_iptm",
    "chai1_pae",
    "multimer_clash_count",
    "multimer_clashing_chains",
    "multimer_alignment_rmsd",
    "reason",
]

BACKBONE_FIELDS = ["round", "rso_name", "rso_pdb", "parent_name", "parent_pdb", "score_field", "score", "source"]


def rso_name_for_design(design_name: str) -> str:
    rso_name, separator, mpnn_number = design_name.rpartition("_mpnn")
    return rso_name if separator and mpnn_number.isdigit() else design_name


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_value(row: dict[str, Any], metrics: dict[str, Any], field: str) -> float | None:
    aliases = [field]
    if field.startswith("chai1_"):
        aliases.append(field.removeprefix("chai1_"))
    else:
        aliases.append(f"chai1_{field}")
    for key in aliases:
        value = to_float(row.get(key))
        if value is not None:
            return value
        value = to_float(metrics.get(key))
        if value is not None:
            return value
    return None


def read_metrics(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    metrics_path = Path(path)
    if not metrics_path.exists():
        return {}
    with metrics_path.open() as handle:
        return json.load(handle)


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def chain_ids(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.replace(",", " ").split() if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def is_interface_design_positions(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "interface"


def residue_ordinals_by_number(pdb_path: Path, chain_id: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")) or len(line) < 27:
                continue
            if line[21].strip() != chain_id:
                continue
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            ordinal = len(seen)
            mapping[resseq] = ordinal
            if icode:
                mapping[f"{resseq}{icode}"] = ordinal
    return mapping


def chain_sasa_by_residue(pdb_path: Path, chain_id: str, python_bin: str | None = None) -> dict[str, float]:
    try:
        import freesasa  # type: ignore
    except Exception as exc:
        if python_bin and Path(python_bin).resolve() != Path(sys.executable).resolve():
            code = """
import json
import sys

import freesasa

structure = freesasa.Structure(sys.argv[1])
result = freesasa.calc(structure)
chain_areas = result.residueAreas().get(sys.argv[2], {})
print(json.dumps({str(resnum).strip(): float(area.total) for resnum, area in chain_areas.items()}))
"""
            result = subprocess.run(
                [python_bin, "-c", code, str(pdb_path), chain_id],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return json.loads(result.stdout)
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                "NISE interface design positions require freesasa in the active "
                f"environment or configured ProteinMPNN Python ({python_bin}). {detail}"
            ) from exc
        raise RuntimeError(
            "NISE interface design positions require freesasa in the active environment"
        ) from exc

    structure = freesasa.Structure(str(pdb_path))
    result = freesasa.calc(structure)
    residue_areas = result.residueAreas()
    chain_areas = residue_areas.get(chain_id, {})
    return {str(resnum).strip(): float(area.total) for resnum, area in chain_areas.items()}


def pdb_with_only_chain(pdb_path: Path, chain_id: str) -> Path:
    temp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pdb")
    with temp:
        with pdb_path.open() as handle:
            for line in handle:
                if line.startswith(("ATOM", "HETATM")) and len(line) > 21 and line[21].strip() != chain_id:
                    continue
                temp.write(line)
    return Path(temp.name)


def interface_positions_for_chain(
    pdb_path: Path,
    chain_id: str,
    threshold: float,
    python_bin: str | None = None,
) -> list[int]:
    complex_sasa = chain_sasa_by_residue(pdb_path, chain_id, python_bin)
    monomer_path = pdb_with_only_chain(pdb_path, chain_id)
    try:
        monomer_sasa = chain_sasa_by_residue(monomer_path, chain_id, python_bin)
    finally:
        try:
            monomer_path.unlink()
        except OSError:
            pass

    ordinal_by_resnum = residue_ordinals_by_number(pdb_path, chain_id)
    positions: list[int] = []
    for resnum, mono_sasa in monomer_sasa.items():
        if resnum not in complex_sasa:
            continue
        delta = mono_sasa - complex_sasa[resnum]
        if delta < threshold:
            continue
        ordinal = ordinal_by_resnum.get(resnum)
        if ordinal is not None:
            positions.append(ordinal)
    return sorted(set(positions))


def interface_positions_by_name(
    staging_dir: Path,
    chains_to_design: Any,
    threshold: float,
    python_bin: str | None = None,
) -> dict[str, list[int]]:
    positions: dict[str, list[int]] = {}
    for pdb_path in sorted(staging_dir.glob("*.pdb")):
        chain_positions: list[int] = []
        for chain_id in chain_ids(chains_to_design):
            chain_positions.extend(interface_positions_for_chain(pdb_path, chain_id, threshold, python_bin))
        if not chain_positions:
            raise ValueError(
                f"No interface design positions found for {pdb_path.name} at SASA threshold {threshold}"
            )
        positions[pdb_path.stem] = sorted(set(chain_positions))
    return positions


def choose_design_positions(
    all_positions: list[int],
    nise: dict[str, Any],
    round_name: str,
    backbone_name: str,
) -> list[int]:
    count = nise.get("num_random_design_positions")
    if count is None:
        return all_positions
    count = int(count)
    if count < 1:
        raise ValueError("nise.num_random_design_positions must be at least 1")
    if count > len(all_positions):
        raise ValueError(
            "nise.num_random_design_positions cannot exceed available design positions: "
            f"requested={count}, available={len(all_positions)}"
        )
    seed = nise.get("random_seed", nise.get("seed", 37))
    rng = random.Random(stable_seed(seed, round_name, backbone_name))
    return sorted(rng.sample(all_positions, count))


def write_nise_fixed_positions(
    parsed_path: Path,
    output_path: Path,
    chains_to_design: Any,
    design_positions: list[int] | dict[str, list[int]],
    nise: dict[str, Any],
    round_name: str,
) -> None:
    design_chains = set(chain_ids(chains_to_design))
    fixed_by_name: dict[str, dict[str, list[int]]] = {}
    chosen_by_name: dict[str, list[int]] = {}
    with parsed_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            name = record["name"]
            position_pool = design_positions[name] if isinstance(design_positions, dict) else design_positions
            chosen = choose_design_positions(position_pool, nise, round_name, name)
            chosen_by_name[name] = chosen
            fixed_by_name[name] = {}
            for key, sequence in record.items():
                if not key.startswith("seq_chain_"):
                    continue
                chain_id = key.removeprefix("seq_chain_")
                residues = list(range(1, len(sequence) + 1))
                if chain_id in design_chains:
                    design_set = set(chosen)
                    fixed_by_name[name][chain_id] = [pos for pos in residues if pos not in design_set]
                else:
                    fixed_by_name[name][chain_id] = residues
    output_path.write_text(json.dumps(fixed_by_name) + "\n")
    manifest_path = output_path.with_name("chosen_design_positions.json")
    manifest_path.write_text(json.dumps(chosen_by_name, indent=2, sort_keys=True) + "\n")
    if nise.get("num_random_design_positions") is not None:
        print(f"NISE random design positions written to {manifest_path}")


def select_initial_backbones(settings: dict[str, Any], nise: dict[str, Any]) -> list[dict[str, Any]]:
    initial_pdb_setting = nise.get("initial_pdb")
    if initial_pdb_setting:
        initial_pdb = Path(str(initial_pdb_setting)).expanduser().resolve()
        if not initial_pdb.exists():
            raise FileNotFoundError(f"NISE initial PDB not found: {initial_pdb}")
        initial_name = str(nise.get("initial_name", initial_pdb.stem))
        return [
            {
                "rso_name": initial_name,
                "rso_pdb": str(initial_pdb),
                "parent_name": initial_name,
                "parent_pdb": str(initial_pdb),
                "score": float(nise.get("initial_score", 1.0)),
                "source": "nise.initial_pdb",
            }
        ]

    chai1_out = stage_dir(settings, "chai1")
    metrics_csv = chai1_out / "chai1_metrics.csv"
    score_field = str(nise.get("score_field", "chai1_aggregate_score"))
    threshold = float(nise.get("threshold", 0.7))
    rows = read_csv_rows(metrics_csv)
    mpnn_input_setting = settings.get("proteinmpnn", {}).get("input_pdb")
    mpnn_input_pdb = Path(str(mpnn_input_setting)).expanduser().resolve() if mpnn_input_setting else None
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "completed":
            continue
        pdb_path = Path(row.get("pdb_path", ""))
        if not pdb_path.exists():
            continue
        metrics = read_metrics(row.get("metrics_json"))
        score = metric_value(row, metrics, score_field)
        if score is None or score < threshold:
            continue
        selected.append(
            {
                "rso_name": mpnn_input_pdb.stem if mpnn_input_pdb else rso_name_for_design(row["design_name"]),
                "rso_pdb": str(mpnn_input_pdb or (stage_dir(settings, "rso") / "accepted_pdbs" / f"{rso_name_for_design(row['design_name'])}.pdb")),
                "parent_name": row["design_name"],
                "parent_pdb": str(pdb_path),
                "score": score,
                "source": "03_chai1",
            }
        )
    selected.sort(key=lambda item: float(item["score"]), reverse=True)
    max_backbones = nise.get("num_top_backbones_per_round")
    if max_backbones is not None:
        selected = selected[: max(0, int(max_backbones))]
    return selected


def select_round_backbones(out: Path, round_index: int, nise: dict[str, Any]) -> list[dict[str, Any]]:
    score_field = str(nise.get("score_field", "chai1_aggregate_score"))
    threshold = float(nise.get("threshold", 0.7))
    rows = read_csv_rows(out / "nise_metrics.csv")
    selected: list[dict[str, Any]] = []
    for row in rows:
        if int(row.get("round") or 0) != round_index:
            continue
        if row.get("status") != "completed":
            continue
        pdb_path = Path(row.get("pdb_path", ""))
        if not pdb_path.exists():
            continue
        metrics = read_metrics(row.get("metrics_json"))
        score = metric_value(row, metrics, score_field)
        if score is None or score < threshold:
            continue
        selected.append(
            {
                "rso_name": row["rso_name"],
                "rso_pdb": row["rso_pdb"],
                "parent_name": row["design_name"],
                "parent_pdb": str(pdb_path),
                "score": score,
                "source": f"03a_NISE_round_{round_index:03d}",
            }
        )
    selected.sort(key=lambda item: float(item["score"]), reverse=True)
    max_backbones = nise.get("num_top_backbones_per_round")
    if max_backbones is not None:
        selected = selected[: max(0, int(max_backbones))]
    return selected


def run_proteinmpnn_round(
    settings: dict[str, Any],
    nise: dict[str, Any],
    round_dir: Path,
    backbones: list[dict[str, Any]],
) -> Path:
    mpnn = dict(require(settings, "proteinmpnn"))
    mpnn.update(nise.get("proteinmpnn", {}))
    mpnn_out = round_dir / "proteinmpnn"
    seqs_dir = mpnn_out / "seqs"
    staging_dir = mpnn_out / "pending_pdbs"
    seqs_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    required = int(nise.get("sequences_sampled_per_backbone", mpnn.get("num_seq_per_target", 4)))
    pending: list[dict[str, Any]] = []
    for backbone in backbones:
        fasta_path = seqs_dir / f"{backbone['parent_name']}.fa"
        if not count_fasta_records(fasta_path, skip_native=True) >= required:
            pending.append(backbone)

    print(f"NISE ProteinMPNN round={round_dir.name} backbones={len(backbones)}, pending={len(pending)}, required_sequences={required}")
    if not pending:
        return seqs_dir

    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    for backbone in pending:
        symlink_or_copy(Path(backbone["parent_pdb"]), staging_dir / f"{backbone['parent_name']}.pdb")

    proteinmpnn_path = Path(require(settings, "tools", "proteinmpnn_path")).expanduser().resolve()
    python_bin = mpnn.get("python", os.environ.get("PROTEINMPNN_PYTHON", sys.executable))
    parsed = mpnn_out / "parsed_pdbs.jsonl"
    assigned = mpnn_out / "assigned_pdbs.jsonl"
    fixed = mpnn_out / "fixed_pdbs.jsonl"
    chains = mpnn.get("chains_to_design", settings.get("binder_chain", "B"))
    design_position_setting = nise.get("design_positions")
    if design_position_setting is None:
        design_position_setting = mpnn.get("design_positions")
    if design_position_setting is None:
        design_position_setting = settings.get("design_positions")
    use_interface_positions = is_interface_design_positions(design_position_setting)
    design_position_values: list[int] | dict[str, list[int]]
    if use_interface_positions:
        threshold = float(nise.get("interface_sasa_threshold", 1.0))
        design_position_values = interface_positions_by_name(staging_dir, chains, threshold, python_bin)
        design_positions = "interface"
    else:
        design_position_values = parse_position_string(design_position_setting)
        design_positions = position_list_for_mpnn(design_position_values)
        if not design_positions:
            raise ValueError("nise.design_positions, proteinmpnn.design_positions, or top-level design_positions must specify relaxed positions")

    run([python_bin, str(proteinmpnn_path / "helper_scripts" / "parse_multiple_chains.py"), "--input_path", str(staging_dir), "--output_path", str(parsed)])
    run([python_bin, str(proteinmpnn_path / "helper_scripts" / "assign_fixed_chains.py"), "--input_path", str(parsed), "--output_path", str(assigned), "--chain_list", str(chains)])
    if nise.get("num_random_design_positions") is None and not use_interface_positions:
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
    else:
        write_nise_fixed_positions(parsed, fixed, chains, design_position_values, nise, round_dir.name)

    batch_size = max(1, min(int(mpnn.get("batch_size", 1)), required))
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
        str(mpnn_out),
        "--num_seq_per_target",
        str(required),
        "--sampling_temp",
        str(mpnn.get("sampling_temp", "0.1")),
        "--seed",
        str(mpnn.get("seed", 37)),
        "--batch_size",
        str(batch_size),
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
    return seqs_dir


def build_chai_designs(
    settings: dict[str, Any],
    nise: dict[str, Any],
    out: Path,
    round_dir: Path,
    round_index: int,
    backbones: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any], Path]]:
    input_dir = round_dir / "chai1" / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    existing = {design_name for design_name, _, _ in find_chai1_predictions(round_dir / "chai1")}
    existing.update(
        row["design_name"]
        for row in read_csv_rows(out / "nise_metrics.csv")
        if row.get("design_name")
        and row.get("status") == "filtered_multimer_clash"
        and row.get("round") == str(round_index)
    )
    target_sequences = sequence_inputs(settings)
    binder_chain = str(nise.get("binder_chain", settings.get("binder_chain", "B")))
    seen_sequences: dict[str, str] = {}
    for chai_dir in sorted(out.glob("round_*/chai1")):
        for prior_name, pdb_path, _ in find_chai1_predictions(chai_dir):
            sequence = extract_chain_sequences(pdb_path, [binder_chain]).get(binder_chain, "")
            if sequence:
                seen_sequences.setdefault(sequence.upper(), prior_name)
    max_per_backbone = nise.get("max_sequences_per_backbone")
    designs: list[tuple[str, dict[str, Any], Path]] = []
    mpnn_number = 0
    for backbone in backbones:
        fasta_path = round_dir / "proteinmpnn" / "seqs" / f"{backbone['parent_name']}.fa"
        if not fasta_path.exists():
            continue
        records = parse_fasta(fasta_path)
        for index, (_, seq) in enumerate(records[1:], start=1):
            if max_per_backbone is not None and index > int(max_per_backbone):
                break
            mpnn_number += 1
            design_name = f"NISE-{round_index}-{mpnn_number}"
            if design_name in existing:
                continue
            sequence_key = seq.upper()
            if sequence_key in seen_sequences:
                print(f"{design_name}: skipped duplicate sequence (same as {seen_sequences[sequence_key]})")
                continue
            seen_sequences[sequence_key] = design_name
            input_path = input_dir / f"{design_name}.fasta"
            sequences = dict(target_sequences)
            sequences[binder_chain] = seq
            write_chai1_fasta(input_path, sequences)
            designs.append((design_name, backbone, input_path))
    return designs


def run_chai_round(
    settings: dict[str, Any],
    nise: dict[str, Any],
    out: Path,
    round_dir: Path,
    round_index: int,
    designs: list[tuple[str, dict[str, Any], Path]],
    limit: int | None,
) -> bool:
    chai1 = dict(require(settings, "chai1"))
    chai1.update(nise.get("chai1", {}))
    if limit is not None:
        designs = designs[: max(0, int(limit))]

    chai_dir = round_dir / "chai1"
    raw_dir = chai_dir / "raw_outputs"
    cif_dir = chai_dir / "cifs"
    predictions_dir = chai_dir / "predictions"
    filtered_dir = chai_dir / "filtered_pdbs"
    metrics_dir = chai_dir / "metrics"
    for path in (raw_dir, cif_dir, predictions_dir, filtered_dir, metrics_dir):
        path.mkdir(parents=True, exist_ok=True)

    target_sequences = sequence_inputs(settings)
    target_chain_ids = list(target_sequences.keys())
    binder_chain = str(nise.get("binder_chain", chai1.get("binder_chain", settings.get("binder_chain", "B"))))
    failed = False
    print(f"NISE Chai-1 round={round_index} pending_predictions={len(designs)}")
    for design_name, backbone, input_path in designs:
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
            clash_result = check_configured_multimer_clashes(pdb_path, settings)
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
                    "nise_round": round_index,
                    "rso_name": backbone["rso_name"],
                    "rso_pdb": backbone["rso_pdb"],
                    "parent_name": backbone["parent_name"],
                    "parent_pdb": backbone["parent_pdb"],
                    "parent_score": backbone["score"],
                }
            )
            add_chai1_error_metrics(metrics, candidates, best_index)
            add_chai1_scores(metrics, raw_output_dir, best_index)
            if clash_result is not None:
                metrics.update(
                    {
                        "multimer_clash_count": clash_result.clash_count,
                        "multimer_clashing_chains": list(clash_result.clashing_chains),
                        "multimer_alignment_rmsd": clash_result.alignment_rmsd,
                    }
                )
            write_metrics_json(metrics_path, metrics)
            if clash_result is not None and clash_result.clashes:
                filtered_path = filtered_dir / pdb_path.name
                pdb_path.replace(filtered_path)
                found_pdb, found_metrics = filtered_path, metrics_path
                status = "filtered_multimer_clash"
                reason = (
                    f"binder clashes with reference chain(s) {','.join(clash_result.clashing_chains)} "
                    f"at {clash_result.clash_count} atom(s)"
                )
            else:
                found_pdb, found_metrics = find_chai1_prediction(chai_dir, design_name)
                status = "completed" if found_pdb and found_metrics else "missing_outputs"
                reason = "" if status == "completed" else "PDB or metrics JSON was not found after Chai-1"
            append_csv(
                out / "nise_metrics.csv",
                {
                    "round": round_index,
                    "design_name": design_name,
                    "rso_name": backbone["rso_name"],
                    "rso_pdb": backbone["rso_pdb"],
                    "parent_name": backbone["parent_name"],
                    "parent_pdb": backbone["parent_pdb"],
                    "parent_score": backbone["score"],
                    "status": status,
                    "input_fasta_path": str(input_path),
                    "pdb_path": str(found_pdb or ""),
                    "cif_path": str(cif_path if cif_path.exists() else ""),
                    "metrics_json": str(found_metrics or ""),
                    "chai1_plddt": metrics.get("plddt"),
                    "chai1_binder_plddt": metrics.get("binder_plddt"),
                    "chai1_target_plddt": metrics.get("target_plddt"),
                    "chai1_aggregate_score": metrics.get("aggregate_score"),
                    "chai1_iptm": metrics.get("iptm"),
                    "chai1_pae": metrics.get("pae"),
                    "multimer_clash_count": clash_result.clash_count if clash_result else None,
                    "multimer_clashing_chains": ",".join(clash_result.clashing_chains) if clash_result else "",
                    "multimer_alignment_rmsd": round(clash_result.alignment_rmsd, 4) if clash_result else None,
                    "reason": reason,
                },
                FIELDS,
            )
            print(f"{design_name}: {status} {reason}")
        except Exception as exc:
            failed = True
            append_csv(
                out / "nise_metrics.csv",
                {
                    "round": round_index,
                    "design_name": design_name,
                    "rso_name": backbone["rso_name"],
                    "rso_pdb": backbone["rso_pdb"],
                    "parent_name": backbone["parent_name"],
                    "parent_pdb": backbone["parent_pdb"],
                    "parent_score": backbone["score"],
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
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description="Run optional NISE-style ProteinMPNN/Chai-1 expansion from high-confidence Chai-1 structures.")
    parser.add_argument("--settings", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int, default=None, help="Maximum pending Chai-1 predictions per round.")
    args = parser.parse_args()

    settings = load_settings(args.settings, args.output_dir)
    nise = dict(settings.get("nise", {}))
    if not bool(nise.get("enabled", False)):
        print("NISE stage disabled; set nise.enabled=true to run 03a_NISE.")
        return 0

    chai1 = dict(require(settings, "chai1"))
    chai1.update(nise.get("chai1", {}))
    configure_model_paths(chai1)
    out = stage_dir(settings, "nise")
    out.mkdir(parents=True, exist_ok=True)
    ensure_csv_fields(out / "nise_metrics.csv", FIELDS)
    (out / "settings_used.json").write_text(json.dumps(nise, indent=2, sort_keys=True) + "\n")

    num_iterations = int(nise.get("num_iterations", 1))
    backbones = select_initial_backbones(settings, nise)
    print(
        "NISE initial_backbones="
        f"{len(backbones)}, score_field={nise.get('score_field', 'chai1_aggregate_score')}, "
        f"threshold={nise.get('threshold', 0.7)}"
    )
    failed = False
    for round_index in range(1, num_iterations + 1):
        if not backbones:
            print(f"NISE round={round_index} has no selected backbones; stopping.")
            break
        round_dir = out / f"round_{round_index:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        for backbone in backbones:
            append_csv(
                out / "selected_backbones.csv",
                {
                    "round": round_index,
                    "rso_name": backbone["rso_name"],
                    "rso_pdb": backbone["rso_pdb"],
                    "parent_name": backbone["parent_name"],
                    "parent_pdb": backbone["parent_pdb"],
                    "score_field": nise.get("score_field", "chai1_aggregate_score"),
                    "score": backbone["score"],
                    "source": backbone["source"],
                },
                BACKBONE_FIELDS,
            )
        run_proteinmpnn_round(settings, nise, round_dir, backbones)
        designs = build_chai_designs(settings, nise, out, round_dir, round_index, backbones)
        failed = run_chai_round(settings, nise, out, round_dir, round_index, designs, args.limit) or failed
        backbones = select_round_backbones(out, round_index, nise)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
