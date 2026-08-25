from types import SimpleNamespace
import unittest

import numpy as np

from pathlib import Path
import tempfile

from scaffold_bind_craft.esmfold2_utils import (
    confidence_metrics,
    validate_confidence_filters,
    validate_openmm_pdb,
)


class ConfidenceMetricsTests(unittest.TestCase):
    def test_summarizes_only_public_esmfold2_confidence_outputs(self) -> None:
        result = SimpleNamespace(
            plddt=np.array([0.8, 0.9, 0.6]),
            ptm=0.7,
            iptm=0.65,
            pae=np.array([[1.0, 2.0, 8.0], [3.0, 4.0, 10.0], [6.0, 12.0, 5.0]]),
            pair_chains_iptm=np.array([[0.9, 0.55], [0.7, 0.8]]),
        )

        metrics = confidence_metrics(result, {"A": "AA", "B": "G"}, "B")

        self.assertEqual(metrics["mean_plddt"], 76.6667)
        self.assertEqual(metrics["binder_mean_plddt"], 60.0)
        self.assertEqual(metrics["target_mean_plddt"], 85.0)
        self.assertEqual(metrics["interface_mean_pae"], 9.0)
        self.assertEqual(metrics["binder_target_iptm"], 0.7)
        self.assertEqual(metrics["pair_chain_iptm"]["A"]["B"], 0.55)
        self.assertNotIn("aggregate_score", metrics)
        self.assertNotIn("pde", metrics)

    def test_rejects_confidence_shapes_that_do_not_match_input(self) -> None:
        result = SimpleNamespace(
            plddt=np.array([0.8]), ptm=0.7, iptm=0.6, pae=None,
            pair_chains_iptm=None,
        )
        with self.assertRaisesRegex(ValueError, "pLDDT values"):
            confidence_metrics(result, {"A": "AA", "B": "G"}, "B")

    def test_rejects_nonexistent_esmfold2_filter_names(self) -> None:
        validate_confidence_filters({"esmfold2_iptm": {"threshold": 0.7}})
        with self.assertRaisesRegex(ValueError, "esmfold2_aggregate_score"):
            validate_confidence_filters({"esmfold2_aggregate_score": {"threshold": 0.7}})

    def test_openmm_handoff_accepts_standard_multichain_protein_pdb(self) -> None:
        lines = []
        serial = 1
        for chain, resname in (("A", "ALA"), ("B", "GLY")):
            for atom_index, atom_name in enumerate(("N", "CA", "C", "O")):
                lines.append(
                    f"ATOM  {serial:5d} {atom_name:^4s} {resname} {chain}{1:4d}    "
                    f"{float(atom_index):8.3f}{float(ord(chain) - 65):8.3f}{0.0:8.3f}"
                    f"  1.00 80.00           {atom_name[0]:>2s}\n"
                )
                serial += 1
            lines.append("TER\n")
        lines.append("END\n")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prediction.pdb"
            path.write_text("".join(lines))
            validate_openmm_pdb(path, {"A": "A", "B": "G"})

    def test_openmm_handoff_rejects_missing_backbone_atoms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prediction.pdb"
            path.write_text(
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 80.00           C\nEND\n"
            )
            with self.assertRaisesRegex(ValueError, "missing backbone"):
                validate_openmm_pdb(path, {"A": "A"})


if __name__ == "__main__":
    unittest.main()
