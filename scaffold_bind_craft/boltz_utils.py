from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def yaml_quote(value: str) -> str:
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def write_boltz_yaml(
    path: Path,
    sequences: Mapping[str, str],
    msa_paths: Mapping[str, str] | None = None,
    binder_chain: str = "B",
    cyclic: bool = False,
    constraints: list[Any] | None = None,
    templates: list[Any] | None = None,
    properties: list[Any] | None = None,
) -> None:
    msa_paths = msa_paths or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["sequences:"]
    for chain_id, sequence in sequences.items():
        lines.extend(
            [
                "  - protein:",
                f"      id: {chain_id}",
                f"      sequence: {sequence}",
                f"      cyclic: {'true' if cyclic else 'false'}",
            ]
        )
        if chain_id in msa_paths and msa_paths[chain_id]:
            lines.append(f"      msa: {yaml_quote(msa_paths[chain_id])}")

    lines.append("")
    lines.append("constraints: " + json.dumps(constraints or []))
    lines.append("")
    lines.append("templates: " + json.dumps(templates or []))
    lines.append("")
    lines.append("properties: " + json.dumps(properties or []))
    path.write_text("\n".join(lines) + "\n")


def find_boltz_prediction(boltz_dir: Path, design_name: str, model_index: int = 0) -> tuple[Path | None, Path | None]:
    candidates = [
        boltz_dir / f"boltz_results_{design_name}" / "predictions" / design_name,
        boltz_dir / "predictions" / design_name,
    ]
    for pred_dir in candidates:
        json_path = pred_dir / f"confidence_{design_name}_model_{model_index}.json"
        cif_path = pred_dir / f"{design_name}_model_{model_index}.cif"
        alt_cif_path = pred_dir / f"confidence_{design_name}_model_{model_index}.cif"
        if json_path.exists() and cif_path.exists():
            return cif_path, json_path
        if json_path.exists() and alt_cif_path.exists():
            return alt_cif_path, json_path
    return None, None


def flatten_json(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten_json(f"{prefix}_{key}" if prefix else str(key), child, out)
    else:
        out[prefix] = value
