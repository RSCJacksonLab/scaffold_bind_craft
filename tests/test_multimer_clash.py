from pathlib import Path
import tempfile
import unittest

from scaffold_bind_craft.multimer_clash import check_multimer_clashes


def atom(serial: int, name: str, residue: int, chain: str, x: float, y: float, z: float) -> str:
    element = name[0]
    return (
        f"ATOM  {serial:5d} {name:^4s} ALA {chain}{residue:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2s}\n"
    )


def write_pdb(path: Path, chains: dict[str, list[tuple[float, float, float]]]) -> None:
    lines: list[str] = []
    serial = 1
    for chain, coordinates in chains.items():
        for residue, (x, y, z) in enumerate(coordinates, start=1):
            lines.append(atom(serial, "CA", residue, chain, x, y, z))
            serial += 1
        lines.append("TER\n")
    lines.append("END\n")
    path.write_text("".join(lines))


class MultimerClashTests(unittest.TestCase):
    def test_aligns_target_and_detects_binder_clash_with_any_reference_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.pdb"
            design = Path(directory) / "design.pdb"
            write_pdb(
                reference,
                {
                    "A": [(0, 0, 0), (4, 0, 0), (0, 4, 0)],
                    "C": [(10, 10, 10)],
                    "D": [(30, 30, 30)],
                },
            )
            # The whole design is translated +5 on x; alignment puts binder B
            # at the reference-chain C atom.
            write_pdb(
                design,
                {
                    "A": [(5, 0, 0), (9, 0, 0), (5, 4, 0)],
                    "B": [(15, 10, 10)],
                },
            )

            result = check_multimer_clashes(design, reference, clash_cutoff=2.0)

            self.assertTrue(result.clashes)
            self.assertEqual(result.clash_count, 1)
            self.assertEqual(result.clashing_chains, ("C",))
            self.assertLess(result.alignment_rmsd, 1e-6)

            thresholded = check_multimer_clashes(
                design,
                reference,
                clash_cutoff=2.0,
                clash_count_threshold=2,
            )
            self.assertFalse(thresholded.clashes)
            self.assertEqual(thresholded.clash_count, 1)

    def test_accepts_when_binder_misses_all_non_target_reference_chains(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.pdb"
            design = Path(directory) / "design.pdb"
            write_pdb(
                reference,
                {"A": [(0, 0, 0), (4, 0, 0), (0, 4, 0)], "C": [(10, 10, 10)]},
            )
            write_pdb(
                design,
                {"A": [(0, 0, 0), (4, 0, 0), (0, 4, 0)], "B": [(20, 20, 20)]},
            )

            result = check_multimer_clashes(design, reference, clash_cutoff=2.0)

            self.assertFalse(result.clashes)
            self.assertEqual(result.clash_count, 0)
            self.assertEqual(result.clashing_chains, ())


if __name__ == "__main__":
    unittest.main()
