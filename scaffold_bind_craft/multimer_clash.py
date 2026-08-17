from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from Bio.PDB import NeighborSearch, PDBParser, Superimposer

from scaffold_bind_craft.config import setting_path


@dataclass(frozen=True)
class MultimerClashResult:
    clashes: bool
    alignment_rmsd: float
    clash_count: int
    clashing_chains: tuple[str, ...]


def _model(structure: Any) -> Any:
    try:
        return next(structure.get_models())
    except StopIteration as exc:
        raise ValueError("Structure contains no models") from exc


def _chain(model: Any, chain_id: str, structure_name: str) -> Any:
    if chain_id not in model:
        raise ValueError(f"Chain {chain_id!r} not found in {structure_name}")
    return model[chain_id]


def _ca_by_residue(chain: Any) -> dict[tuple[int, str], Any]:
    return {
        (residue.id[1], residue.id[2]): residue["CA"]
        for residue in chain
        if residue.id[0] == " " and "CA" in residue
    }


def _ca_in_order(chain: Any) -> list[Any]:
    return [residue["CA"] for residue in chain if residue.id[0] == " " and "CA" in residue]


def _heavy_atoms(entity: Any) -> list[Any]:
    return [
        atom
        for atom in entity.get_atoms()
        if (atom.element or "").upper() != "H" and not atom.get_name().upper().startswith("H")
    ]


def align_target_chain(mobile: Any, reference: Any, target_chain: str) -> float:
    mobile_chain = _chain(_model(mobile), target_chain, "design")
    reference_chain = _chain(_model(reference), target_chain, "reference")
    mobile_ca = _ca_by_residue(mobile_chain)
    reference_ca = _ca_by_residue(reference_chain)
    common = sorted(set(mobile_ca) & set(reference_ca))

    if len(common) >= 3:
        fixed = [reference_ca[key] for key in common]
        moving = [mobile_ca[key] for key in common]
    else:
        fixed = _ca_in_order(reference_chain)
        moving = _ca_in_order(mobile_chain)
        if len(fixed) != len(moving):
            raise ValueError(
                "Fewer than 3 matching target-chain CA atoms and ordered CA counts differ: "
                f"reference={len(fixed)}, design={len(moving)}"
            )
        if len(fixed) < 3:
            raise ValueError("Fewer than 3 target-chain CA atoms available for alignment")

    superimposer = Superimposer()
    superimposer.set_atoms(fixed, moving)
    superimposer.apply(mobile.get_atoms())
    return float(superimposer.rms)


def check_multimer_clashes(
    design_pdb: str | Path,
    reference_pdb: str | Path,
    target_chain: str = "A",
    binder_chain: str = "B",
    clash_cutoff: float = 2.0,
) -> MultimerClashResult:
    parser = PDBParser(QUIET=True)
    design = parser.get_structure("design", str(design_pdb))
    reference = parser.get_structure("reference", str(reference_pdb))
    rmsd = align_target_chain(design, reference, target_chain)

    design_model = _model(design)
    reference_model = _model(reference)
    binder = _chain(design_model, binder_chain, "design")
    reference_atoms = [
        atom
        for chain in reference_model
        if chain.id != target_chain
        for atom in _heavy_atoms(chain)
    ]
    if not reference_atoms:
        raise ValueError("Reference PDB has no non-target-chain heavy atoms to check")

    neighbours = NeighborSearch(reference_atoms)
    clashing_atoms: set[int] = set()
    clashing_chains: set[str] = set()
    for binder_atom in _heavy_atoms(binder):
        hits = neighbours.search(binder_atom.coord, float(clash_cutoff), level="A")
        if hits:
            clashing_atoms.add(id(binder_atom))
            clashing_chains.update(atom.get_parent().get_parent().id for atom in hits)

    return MultimerClashResult(
        clashes=bool(clashing_atoms),
        alignment_rmsd=rmsd,
        clash_count=len(clashing_atoms),
        clashing_chains=tuple(sorted(clashing_chains)),
    )


def configured_multimer_clash_filter(settings: Mapping[str, Any]) -> dict[str, Any] | None:
    config = settings.get("multimer_clash_filter", {})
    if not isinstance(config, Mapping) or not bool(config.get("enabled", False)):
        return None
    reference_value = config.get("reference_pdb")
    if not reference_value:
        raise KeyError("multimer_clash_filter.reference_pdb is required when the filter is enabled")
    reference_pdb = setting_path(settings, str(reference_value))
    if not reference_pdb.is_file():
        raise FileNotFoundError(f"Multimer clash reference PDB not found: {reference_pdb}")
    target_chain = str(config.get("target_chain", settings.get("target_chains", "A")))
    binder_chain = str(config.get("binder_chain", settings.get("binder_chain", "B")))
    if "," in target_chain or len(target_chain) != 1:
        raise ValueError("multimer_clash_filter.target_chain must be one chain ID")
    if "," in binder_chain or len(binder_chain) != 1:
        raise ValueError("multimer_clash_filter.binder_chain must be one chain ID")
    return {
        "reference_pdb": reference_pdb,
        "target_chain": target_chain,
        "binder_chain": binder_chain,
        "clash_cutoff": float(config.get("clash_cutoff", 2.0)),
    }


def check_configured_multimer_clashes(
    pdb_path: str | Path, settings: Mapping[str, Any]
) -> MultimerClashResult | None:
    config = configured_multimer_clash_filter(settings)
    if config is None:
        return None
    return check_multimer_clashes(pdb_path, **config)
