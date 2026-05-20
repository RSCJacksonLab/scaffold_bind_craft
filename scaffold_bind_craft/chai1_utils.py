from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from Bio.PDB import MMCIFParser, PDBIO


def chain_id_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.replace(",", " ").split() if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def write_chai1_fasta(path: Path, sequences: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for chain_id, sequence in sequences.items():
        lines.append(f">protein|name={chain_id}")
        lines.append(str(sequence))
    path.write_text("\n".join(lines) + "\n")


def find_chai1_prediction(chai1_dir: Path, design_name: str) -> tuple[Path | None, Path | None]:
    pdb_path = chai1_dir / "predictions" / f"{design_name}.pdb"
    metrics_path = chai1_dir / "metrics" / f"{design_name}.json"
    if pdb_path.exists() and metrics_path.exists():
        return pdb_path, metrics_path
    return None, None


def find_chai1_predictions(chai1_dir: Path) -> list[tuple[str, Path, Path]]:
    records: list[tuple[str, Path, Path]] = []
    for pdb_path in sorted((chai1_dir / "predictions").glob("*.pdb")):
        design_name = pdb_path.stem
        metrics_path = chai1_dir / "metrics" / f"{design_name}.json"
        if metrics_path.exists():
            records.append((design_name, pdb_path, metrics_path))
    return records


def cif_to_pdb(cif_path: Path, pdb_path: Path) -> None:
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(cif_path.stem, str(cif_path))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_path))


def copy_best_candidate(cif_paths: list[Path], ranking_data: list[Any], destination: Path) -> tuple[int, Path, float | None]:
    if not cif_paths:
        raise FileNotFoundError("Chai-1 did not return any CIF candidates")
    scores: list[float | None] = []
    for ranking in ranking_data:
        score = getattr(ranking, "aggregate_score", None)
        try:
            scores.append(float(score.item()))
        except Exception:
            scores.append(None)
    best_index = 0
    numeric_scores = [(index, score) for index, score in enumerate(scores) if score is not None]
    if numeric_scores:
        best_index = max(numeric_scores, key=lambda item: item[1])[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cif_paths[best_index], destination)
    best_score = scores[best_index] if best_index < len(scores) else None
    return best_index, cif_paths[best_index], best_score


def plddt_metrics_from_pdb(
    pdb_path: Path,
    binder_chain: str,
    target_chains: Iterable[str],
) -> dict[str, Any]:
    target_set = set(target_chains)
    all_values: list[float] = []
    binder_values: list[float] = []
    target_values: list[float] = []
    per_chain: dict[str, list[float]] = {}

    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 66:
                continue
            if line[12:16].strip() != "CA":
                continue
            chain_id = line[21]
            try:
                value = float(line[60:66])
            except ValueError:
                continue
            all_values.append(value)
            per_chain.setdefault(chain_id, []).append(value)
            if chain_id == binder_chain:
                binder_values.append(value)
            if chain_id in target_set:
                target_values.append(value)

    def mean(values: list[float]) -> float | None:
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    metrics: dict[str, Any] = {
        "plddt": mean(all_values),
        "binder_plddt": mean(binder_values),
        "target_plddt": mean(target_values),
        "chain_count": len(per_chain),
    }
    metrics["chain_plddt"] = {chain_id: mean(values) for chain_id, values in sorted(per_chain.items())}
    return metrics


def add_chai1_scores(metrics: dict[str, Any], output_dir: Path, model_index: int) -> None:
    scores_path = output_dir / f"scores.model_idx_{model_index}.npz"
    if not scores_path.exists():
        return
    scores = np.load(scores_path)
    for key in ("aggregate_score", "ptm", "iptm", "chain_ptm", "chain_pair_iptm", "pae", "pde"):
        if key not in scores:
            continue
        value = scores[key]
        if np.isscalar(value) or value.shape == ():
            metrics[key] = float(value)


def write_metrics_json(path: Path, metrics: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
