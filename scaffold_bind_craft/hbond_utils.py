from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, Selection
from Bio.PDB.SASA import ShrakeRupley
from scipy.spatial import cKDTree


POLAR_ELEMENTS = {"N", "O", "S"}
ACCEPTOR_ELEMENTS = {"N", "O", "S"}
DONOR_ELEMENTS = {"N", "O", "S"}
POSITIVE_SIDECHAIN_ATOMS = {("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"), ("LYS", "NZ")}
NEGATIVE_SIDECHAIN_ATOMS = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}


@dataclass(frozen=True)
class HBondStats:
    binder_target_hbonds: int = 0
    buried_polar_atoms_total: int = 0
    buried_satisfied_polar_atoms: int = 0
    buried_unsat_hbonds: int = 0
    binder_buried_unsatisfied_polar_atoms: int = 0
    target_buried_unsatisfied_polar_atoms: int = 0
    buried_unsatisfied_donor_atoms: int = 0
    buried_unsatisfied_acceptor_atoms: int = 0
    delta_buried_unsatisfied_polar_atoms: int = 0
    binder_target_salt_bridges: int = 0


def count_hbonds_and_buried_unsats(
    pdb_path: str | Path,
    binder_chain: str = "B",
    target_chain: str = "A",
    hbond_distance: float = 3.5,
    hydrogen_acceptor_distance: float = 2.6,
    min_dha_angle: float = 120.0,
    buried_sasa_cutoff: float = 0.2,
    interface_distance: float = 4.0,
) -> HBondStats:
    """Count binder-target H-bonds and buried unsatisfied polar atoms.

    The H-bond count uses explicit hydrogens when present and falls back to a
    donor/acceptor heavy-atom distance rule when they are absent. Buried unsats
    are approximated as buried polar heavy atoms in either interface chain that
    do not participate in any H-bond in the complex.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", str(pdb_path))
    model = structure[0]
    if binder_chain not in model or target_chain not in model:
        return HBondStats()

    atoms = [atom for atom in model.get_atoms() if _is_heavy_atom(atom)]
    hydrogens = [atom for atom in model.get_atoms() if _element(atom) == "H"]
    hydrogen_parent_by_donor = _map_hydrogens_to_donors(atoms, hydrogens)

    hbonds = _find_hbonds(
        atoms,
        hydrogen_parent_by_donor,
        hbond_distance=hbond_distance,
        hydrogen_acceptor_distance=hydrogen_acceptor_distance,
        min_dha_angle=min_dha_angle,
    )

    binder_target_hbonds = sum(
        1
        for donor, acceptor in hbonds
        if {donor.get_parent().get_parent().id, acceptor.get_parent().get_parent().id}
        == {str(binder_chain), str(target_chain)}
    )

    satisfied_atom_ids = {id(donor) for donor, _ in hbonds} | {id(acceptor) for _, acceptor in hbonds}
    buried_interface_polar_atoms = _buried_interface_polar_atoms(
        model,
        str(binder_chain),
        str(target_chain),
        buried_sasa_cutoff=buried_sasa_cutoff,
        interface_distance=interface_distance,
    )
    buried_satisfied = sum(id(atom) in satisfied_atom_ids for atom in buried_interface_polar_atoms)
    buried_unsatisfied_atoms = [atom for atom in buried_interface_polar_atoms if id(atom) not in satisfied_atom_ids]
    binder_buried_unsats = sum(_chain_id(atom) == str(binder_chain) for atom in buried_unsatisfied_atoms)
    target_buried_unsats = sum(_chain_id(atom) == str(target_chain) for atom in buried_unsatisfied_atoms)

    intrachain_satisfied_atom_ids = {
        id(atom)
        for donor, acceptor in hbonds
        if _chain_id(donor) == _chain_id(acceptor)
        for atom in (donor, acceptor)
    }
    unbound_buried_unsats = _count_unbound_buried_unsatisfied_interface_atoms(
        model,
        str(binder_chain),
        str(target_chain),
        intrachain_satisfied_atom_ids,
        buried_sasa_cutoff,
        interface_distance,
    )
    buried_unsats = len(buried_unsatisfied_atoms)

    return HBondStats(
        binder_target_hbonds=binder_target_hbonds,
        buried_polar_atoms_total=len(buried_interface_polar_atoms),
        buried_satisfied_polar_atoms=buried_satisfied,
        buried_unsat_hbonds=buried_unsats,
        binder_buried_unsatisfied_polar_atoms=binder_buried_unsats,
        target_buried_unsatisfied_polar_atoms=target_buried_unsats,
        buried_unsatisfied_donor_atoms=sum(_is_donor(atom) for atom in buried_unsatisfied_atoms),
        buried_unsatisfied_acceptor_atoms=sum(_is_acceptor(atom) for atom in buried_unsatisfied_atoms),
        delta_buried_unsatisfied_polar_atoms=buried_unsats - unbound_buried_unsats,
        binder_target_salt_bridges=_count_salt_bridges(model, str(binder_chain), str(target_chain)),
    )


def _find_hbonds(
    atoms: list,
    hydrogen_parent_by_donor: dict[int, list],
    hbond_distance: float,
    hydrogen_acceptor_distance: float,
    min_dha_angle: float,
) -> set[tuple[object, object]]:
    donors = [atom for atom in atoms if _is_donor(atom)]
    acceptors = [atom for atom in atoms if _is_acceptor(atom)]
    if not donors or not acceptors:
        return set()

    acceptor_coords = np.array([atom.coord for atom in acceptors])
    tree = cKDTree(acceptor_coords)
    hbonds: set[tuple[object, object]] = set()
    has_hydrogens = bool(hydrogen_parent_by_donor)

    for donor in donors:
        donor_chain = donor.get_parent().get_parent().id
        donor_residue = donor.get_parent()
        for acceptor_index in tree.query_ball_point(donor.coord, hbond_distance):
            acceptor = acceptors[acceptor_index]
            if acceptor is donor:
                continue
            acceptor_residue = acceptor.get_parent()
            if donor_chain == acceptor_residue.get_parent().id and donor_residue.id == acceptor_residue.id:
                continue
            hydrogens = hydrogen_parent_by_donor.get(id(donor), [])
            if has_hydrogens and hydrogens:
                if any(
                    np.linalg.norm(hydrogen.coord - acceptor.coord) <= hydrogen_acceptor_distance
                    and _angle_degrees(donor.coord, hydrogen.coord, acceptor.coord) >= min_dha_angle
                    for hydrogen in hydrogens
                ):
                    hbonds.add((donor, acceptor))
            elif not has_hydrogens or not hydrogens:
                hbonds.add((donor, acceptor))
    return hbonds


def _buried_interface_polar_atoms(
    model,
    binder_chain: str,
    target_chain: str,
    buried_sasa_cutoff: float,
    interface_distance: float,
) -> list:
    sr = ShrakeRupley(probe_radius=1.4, n_points=960)
    sr.compute(model, level="A")
    return [
        atom
        for atom in _interface_polar_atoms(model, binder_chain, target_chain, interface_distance)
        if getattr(atom, "sasa", 0.0) <= buried_sasa_cutoff
    ]


def _interface_polar_atoms(model, binder_chain: str, target_chain: str, interface_distance: float) -> list:
    if binder_chain not in model or target_chain not in model:
        return []

    chain_atoms = {
        binder_chain: Selection.unfold_entities(model[binder_chain], "A"),
        target_chain: Selection.unfold_entities(model[target_chain], "A"),
    }
    interface_atom_ids = set()
    for chain_id, other_chain_id in ((binder_chain, target_chain), (target_chain, binder_chain)):
        atoms = chain_atoms[chain_id]
        other_atoms = chain_atoms[other_chain_id]
        if not atoms or not other_atoms:
            continue
        tree = cKDTree(np.array([atom.coord for atom in other_atoms]))
        for atom in atoms:
            if tree.query_ball_point(atom.coord, interface_distance):
                interface_atom_ids.add(id(atom))

    interface_polar_atoms = []
    for chain_id in (binder_chain, target_chain):
        for atom in Selection.unfold_entities(model[chain_id], "A"):
            if id(atom) in interface_atom_ids and _is_polar_heavy_atom(atom):
                interface_polar_atoms.append(atom)
    return interface_polar_atoms


def _count_unbound_buried_unsatisfied_interface_atoms(
    model,
    binder_chain: str,
    target_chain: str,
    intrachain_satisfied_atom_ids: set[int],
    buried_sasa_cutoff: float,
    interface_distance: float,
) -> int:
    interface_polar_atoms = _interface_polar_atoms(model, binder_chain, target_chain, interface_distance)
    count = 0
    sr = ShrakeRupley(probe_radius=1.4, n_points=960)
    for chain_id in (binder_chain, target_chain):
        sr.compute(model[chain_id], level="A")
        count += sum(
            _chain_id(atom) == chain_id
            and getattr(atom, "sasa", 0.0) <= buried_sasa_cutoff
            and id(atom) not in intrachain_satisfied_atom_ids
            for atom in interface_polar_atoms
        )
    return count


def _count_salt_bridges(model, binder_chain: str, target_chain: str, distance_cutoff: float = 4.0) -> int:
    charged_atoms: dict[str, dict[str, list]] = {}
    for chain_id in (binder_chain, target_chain):
        positive = []
        negative = []
        for atom in Selection.unfold_entities(model[chain_id], "A"):
            key = (atom.get_parent().get_resname().strip().upper(), atom.get_name().strip().upper())
            if key in POSITIVE_SIDECHAIN_ATOMS:
                positive.append(atom)
            elif key in NEGATIVE_SIDECHAIN_ATOMS:
                negative.append(atom)
        charged_atoms[chain_id] = {"positive": positive, "negative": negative}

    residue_pairs = set()
    for positive_chain, negative_chain in ((binder_chain, target_chain), (target_chain, binder_chain)):
        negative_atoms = charged_atoms[negative_chain]["negative"]
        if not negative_atoms:
            continue
        tree = cKDTree(np.array([atom.coord for atom in negative_atoms]))
        for positive_atom in charged_atoms[positive_chain]["positive"]:
            for index in tree.query_ball_point(positive_atom.coord, distance_cutoff):
                negative_atom = negative_atoms[index]
                residue_pairs.add(
                    (
                        positive_chain,
                        positive_atom.get_parent().id,
                        negative_chain,
                        negative_atom.get_parent().id,
                    )
                )
    return len(residue_pairs)


def _map_hydrogens_to_donors(heavy_atoms: list, hydrogens: list) -> dict[int, list]:
    donors = [atom for atom in heavy_atoms if _is_donor(atom)]
    if not donors or not hydrogens:
        return {}
    donor_coords = np.array([atom.coord for atom in donors])
    tree = cKDTree(donor_coords)
    mapping: dict[int, list] = {}
    for hydrogen in hydrogens:
        nearby = tree.query_ball_point(hydrogen.coord, 1.25)
        for donor_index in nearby:
            donor = donors[donor_index]
            if donor.get_parent() is hydrogen.get_parent():
                mapping.setdefault(id(donor), []).append(hydrogen)
    return mapping


def _is_heavy_atom(atom) -> bool:
    return _element(atom) != "H"


def _is_polar_heavy_atom(atom) -> bool:
    return _is_heavy_atom(atom) and _element(atom) in POLAR_ELEMENTS


def _is_donor(atom) -> bool:
    if _element(atom) not in DONOR_ELEMENTS:
        return False
    name = atom.get_name().strip().upper()
    residue = atom.get_parent()
    resname = residue.get_resname().strip().upper()
    if name.startswith("O") and resname not in {"SER", "THR", "TYR"}:
        return False
    return True


def _is_acceptor(atom) -> bool:
    if _element(atom) not in ACCEPTOR_ELEMENTS:
        return False
    name = atom.get_name().strip().upper()
    resname = atom.get_parent().get_resname().strip().upper()
    if name == "N" or name.startswith("NZ") or name.startswith("NH"):
        return False
    if resname == "ARG" and name.startswith(("NE", "NH")):
        return False
    if resname == "LYS" and name == "NZ":
        return False
    return True


def _element(atom) -> str:
    element = (getattr(atom, "element", "") or "").strip().upper()
    if element:
        return element
    name = atom.get_name().strip().upper()
    return name[0] if name else ""


def _chain_id(atom) -> str:
    return str(atom.get_parent().get_parent().id)


def _angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cosine = float(np.dot(ba, bc) / denom)
    cosine = max(-1.0, min(1.0, cosine))
    return float(np.degrees(np.arccos(cosine)))
