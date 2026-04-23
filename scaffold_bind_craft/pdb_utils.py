from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterable


def extract_chain_sequences(structure_path: str | Path, chains: Iterable[str] | None = None) -> OrderedDict[str, str]:
    from Bio.PDB import MMCIFParser, PDBParser
    from Bio.SeqUtils import seq1

    path = Path(structure_path)
    if path.suffix.lower() in {".cif", ".mmcif"}:
        structure = MMCIFParser(QUIET=True).get_structure(path.stem, str(path))
    else:
        structure = PDBParser(QUIET=True).get_structure(path.stem, str(path))

    wanted = set(chains) if chains is not None else None
    model = structure[0]
    seqs: OrderedDict[str, str] = OrderedDict()
    for chain in model:
        if wanted is not None and chain.id not in wanted:
            continue
        residues: list[str] = []
        for residue in chain:
            if residue.id[0] != " ":
                continue
            try:
                residues.append(seq1(residue.resname))
            except Exception:
                residues.append("X")
        if residues:
            seqs[chain.id] = "".join(residues)
    return seqs


def cif_to_pdb(cif_path: str | Path, pdb_path: str | Path) -> Path:
    from Bio.PDB import MMCIFParser, PDBIO

    cif_path = Path(cif_path)
    pdb_path = Path(pdb_path)
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    structure = MMCIFParser(QUIET=True).get_structure(cif_path.stem, str(cif_path))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_path))
    return pdb_path


def parse_position_string(value: str | Iterable[int] | None) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        tokens = value.replace(",", " ").split()
        out: list[int] = []
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                start, end = token.split("-", 1)
                out.extend(range(int(start), int(end) + 1))
            else:
                out.append(int(token))
        return sorted(set(out))
    return sorted(set(int(v) for v in value))


def position_list_for_mpnn(value: str | Iterable[int] | None) -> str:
    return " ".join(str(pos) for pos in parse_position_string(value))
