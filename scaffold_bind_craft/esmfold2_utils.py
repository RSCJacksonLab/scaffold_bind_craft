from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from Bio.PDB import PDBParser, Polypeptide


ESMFOLD2_CONFIDENCE_FIELDS = (
    "esmfold2_mean_plddt",
    "esmfold2_binder_mean_plddt",
    "esmfold2_target_mean_plddt",
    "esmfold2_ptm",
    "esmfold2_iptm",
    "esmfold2_binder_target_iptm",
    "esmfold2_mean_pae",
    "esmfold2_binder_mean_pae",
    "esmfold2_target_mean_pae",
    "esmfold2_interface_mean_pae",
)


def validate_confidence_filters(filters: Mapping[str, Any]) -> None:
    unknown = sorted(
        key for key in filters
        if key.startswith("esmfold2_") and key not in ESMFOLD2_CONFIDENCE_FIELDS
    )
    if unknown:
        raise ValueError(
            "Unknown ESMFold2 confidence filter(s): " + ", ".join(unknown)
            + ". Allowed fields: " + ", ".join(ESMFOLD2_CONFIDENCE_FIELDS)
        )


def validate_openmm_pdb(pdb_path: Path, sequences: Mapping[str, str]) -> None:
    """Validate the protein PDB contract consumed by PDBFixer/OpenMM in Step 4."""
    invalid_ids = [chain for chain in sequences if len(chain) != 1 or not chain.strip()]
    if invalid_ids:
        raise ValueError(f"PDB/OpenMM requires one-character nonblank chain IDs: {invalid_ids}")
    structure = PDBParser(QUIET=True).get_structure(pdb_path.stem, str(pdb_path))
    models = list(structure)
    if len(models) != 1:
        raise ValueError(f"Expected one PDB model for OpenMM, found {len(models)}")
    model = models[0]
    observed_chains = {chain.id for chain in model}
    expected_chains = set(sequences)
    if observed_chains != expected_chains:
        raise ValueError(
            f"PDB chain IDs do not match ESMFold2 input: expected {sorted(expected_chains)}, "
            f"found {sorted(observed_chains)}"
        )
    for chain_id, sequence in sequences.items():
        residues = [residue for residue in model[chain_id] if residue.id[0] == " "]
        if len(residues) != len(sequence):
            raise ValueError(
                f"Chain {chain_id} has {len(residues)} PDB residues for {len(sequence)} input residues"
            )
        for index, residue in enumerate(residues, start=1):
            if not Polypeptide.is_aa(residue, standard=True):
                raise ValueError(
                    f"Chain {chain_id} residue {index} is not a standard OpenMM protein residue: "
                    f"{residue.get_resname()}"
                )
            missing = sorted({"N", "CA", "C", "O"} - {atom.name for atom in residue})
            if missing:
                raise ValueError(
                    f"Chain {chain_id} residue {index} is missing backbone atom(s): {', '.join(missing)}"
                )
            coordinates = np.asarray([atom.coord for atom in residue], dtype=float)
            if not np.isfinite(coordinates).all():
                raise ValueError(f"Chain {chain_id} residue {index} has non-finite coordinates")


def find_esmfold2_prediction(stage: Path, design_name: str) -> tuple[Path | None, Path | None]:
    pdb = stage / "predictions" / f"{design_name}.pdb"
    metrics = stage / "metrics" / f"{design_name}.json"
    return (pdb, metrics) if pdb.exists() and metrics.exists() else (None, None)


def find_esmfold2_predictions(stage: Path) -> list[tuple[str, Path, Path]]:
    records: list[tuple[str, Path, Path]] = []
    for pdb in sorted((stage / "predictions").glob("*.pdb")):
        metrics = stage / "metrics" / f"{pdb.stem}.json"
        if metrics.exists():
            records.append((pdb.stem, pdb, metrics))
    return records


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=float)


def _mean(value: Any) -> float | None:
    array = _array(value)
    return round(float(array.mean()), 4) if array.size else None


def confidence_metrics(result: Any, sequences: Mapping[str, str], binder_chain: str) -> dict[str, Any]:
    """Summarize confidence fields present on ESMFold2's public fold result.

    ESMFold2 pLDDT is returned on [0, 1] and is converted to the conventional
    [0, 100] scale. PAE remains in Angstroms. Per-chain-pair ipTM is
    directional, so both directions are retained in JSON and their maximum is
    reported as the scalar binder-target ipTM summary.
    """
    if result.plddt is None:
        raise ValueError("ESMFold2 result did not contain pLDDT")
    plddt = _array(result.plddt) * 100.0
    chain_slices: dict[str, slice] = {}
    cursor = 0
    for chain, sequence in sequences.items():
        chain_slices[chain] = slice(cursor, cursor + len(sequence))
        cursor += len(sequence)
    if plddt.size != cursor:
        raise ValueError(f"ESMFold2 returned {plddt.size} pLDDT values for {cursor} input residues")
    if binder_chain not in chain_slices:
        raise ValueError(f"Binder chain {binder_chain!r} is not present in the ESMFold2 input")

    target_chains = [chain for chain in sequences if chain != binder_chain]
    binder_slice = chain_slices[binder_chain]
    target_indices = np.concatenate(
        [np.arange(chain_slices[c].start, chain_slices[c].stop) for c in target_chains]
    ) if target_chains else np.array([], dtype=int)
    metrics: dict[str, Any] = {
        "mean_plddt": _mean(plddt),
        "binder_mean_plddt": _mean(plddt[binder_slice]),
        "target_mean_plddt": _mean(plddt[target_indices]),
        "chain_mean_plddt": {chain: _mean(plddt[index]) for chain, index in chain_slices.items()},
        "ptm": None if result.ptm is None else float(result.ptm),
        "iptm": None if result.iptm is None else float(result.iptm),
    }

    if result.pae is not None:
        pae = _array(result.pae)
        if pae.shape != (cursor, cursor):
            raise ValueError(f"ESMFold2 returned PAE shape {pae.shape}, expected {(cursor, cursor)}")
        binder_indices = np.arange(binder_slice.start, binder_slice.stop)
        interface = np.concatenate(
            [pae[np.ix_(binder_indices, target_indices)].ravel(), pae[np.ix_(target_indices, binder_indices)].ravel()]
        ) if target_indices.size else np.array([])
        metrics.update({
            "mean_pae": _mean(pae),
            "binder_mean_pae": _mean(pae[np.ix_(binder_indices, binder_indices)]),
            "target_mean_pae": _mean(pae[np.ix_(target_indices, target_indices)]),
            "interface_mean_pae": _mean(interface),
            "chain_pair_mean_pae": {
                row_chain: {
                    column_chain: _mean(pae[row_slice, column_slice])
                    for column_chain, column_slice in chain_slices.items()
                }
                for row_chain, row_slice in chain_slices.items()
            },
        })

    if result.pair_chains_iptm is not None:
        pair = _array(result.pair_chains_iptm)
        chain_names = list(sequences)
        if pair.shape != (len(chain_names), len(chain_names)):
            raise ValueError(
                f"ESMFold2 returned pair_chains_iptm shape {pair.shape}, "
                f"expected {(len(chain_names), len(chain_names))}"
            )
        pair_scores = {
            row: {column: round(float(pair[i, j]), 6) for j, column in enumerate(chain_names)}
            for i, row in enumerate(chain_names)
        }
        binder_index = chain_names.index(binder_chain)
        target_indexes = [chain_names.index(chain) for chain in target_chains]
        binder_target_values = [
            score
            for target_index in target_indexes
            for score in (pair[binder_index, target_index], pair[target_index, binder_index])
        ]
        metrics["pair_chain_iptm"] = pair_scores
        metrics["binder_target_iptm"] = (
            round(float(max(binder_target_values)), 6) if binder_target_values else None
        )
    return metrics
