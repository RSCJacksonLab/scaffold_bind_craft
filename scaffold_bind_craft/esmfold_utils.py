from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def chain_id_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.replace(",", " ").split() if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def write_esmfold_fasta(path: Path, name: str, sequences: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sequence = ":".join(str(seq) for seq in sequences.values())
    path.write_text(f">{name}\n{sequence}\n")


def find_esmfold_prediction(esmfold_dir: Path, design_name: str) -> tuple[Path | None, Path | None]:
    pdb_path = esmfold_dir / "predictions" / f"{design_name}.pdb"
    metrics_path = esmfold_dir / "metrics" / f"{design_name}.json"
    if pdb_path.exists() and metrics_path.exists():
        return pdb_path, metrics_path
    return None, None


def find_esmfold_predictions(esmfold_dir: Path) -> list[tuple[str, Path, Path]]:
    records: list[tuple[str, Path, Path]] = []
    for pdb_path in sorted((esmfold_dir / "predictions").glob("*.pdb")):
        design_name = pdb_path.stem
        metrics_path = esmfold_dir / "metrics" / f"{design_name}.json"
        if metrics_path.exists():
            records.append((design_name, pdb_path, metrics_path))
    return records


def pdb_chain_order(pdb_text: str) -> list[str]:
    chains: list[str] = []
    seen: set[str] = set()
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 22:
            continue
        chain_id = line[21]
        if chain_id not in seen:
            seen.add(chain_id)
            chains.append(chain_id)
    return chains


def remap_pdb_chain_ids(pdb_text: str, desired_chain_ids: Iterable[str]) -> str:
    desired = [str(chain_id) for chain_id in desired_chain_ids]
    if any(len(chain_id) != 1 for chain_id in desired):
        raise ValueError(f"PDB chain ids must be one character: {desired}")
    observed = pdb_chain_order(pdb_text)
    if len(observed) != len(desired):
        raise ValueError(f"ESMFold returned {len(observed)} chains, expected {len(desired)}")
    mapping = dict(zip(observed, desired))
    lines: list[str] = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM  ", "HETATM", "TER")) and len(line) >= 22 and line[21] in mapping:
            line = f"{line[:21]}{mapping[line[21]]}{line[22:]}"
        lines.append(line)
    return "\n".join(lines) + "\n"


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


def write_metrics_json(path: Path, metrics: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
