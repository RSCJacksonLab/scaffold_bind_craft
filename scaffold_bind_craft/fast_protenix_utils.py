from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def write_af3_json(
    path: Path,
    name: str,
    sequences: Mapping[str, str],
    model_seeds: list[int] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    job = {
        "name": name,
        "sequences": [
            {
                "protein": {
                    "id": [str(chain_id)],
                    "sequence": sequence,
                }
            }
            for chain_id, sequence in sequences.items()
        ],
        "modelSeeds": model_seeds or [101],
        "dialect": "alphafold3",
        "version": 3,
    }
    path.write_text(json.dumps(job, indent=2) + "\n")


def find_protenix_prediction(
    protenix_dir: Path,
    design_name: str,
    seed: int = 101,
    sample_index: int = 0,
) -> tuple[Path | None, Path | None]:
    candidates = [
        protenix_dir / "protenix_output" / design_name / f"seed_{seed}" / "predictions",
        protenix_dir / design_name / f"seed_{seed}" / "predictions",
    ]
    for pred_dir in candidates:
        cif_path = pred_dir / f"{design_name}_sample_{sample_index}.cif"
        json_path = pred_dir / f"{design_name}_summary_confidence_sample_{sample_index}.json"
        if cif_path.exists() and json_path.exists():
            return cif_path, json_path
    return None, None
