from __future__ import annotations

import contextlib
import copy
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from itertools import zip_longest
from pathlib import Path

import numpy as np
from Bio.PDB import Model, PDBIO, PDBParser, Polypeptide, Selection, Structure, Superimposer
from Bio.PDB.SASA import ShrakeRupley
from Bio.SeqUtils import seq1
from openmm import OpenMMException, Platform, app, unit
import openmm
from pdbfixer import PDBFixer
from scipy.spatial import cKDTree

from scaffold_bind_craft.hbond_utils import count_hbonds_and_buried_unsats


R_CHOTHIA = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}
HYDROPHOBIC_AA_SET = set("ACFGILMPVWY")
THREE_TO_ONE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

try:
    import freesasa  # type: ignore

    HAS_FREESASA = True
except Exception:
    freesasa = None  # type: ignore
    HAS_FREESASA = False

_OPENMM_FORCEFIELD = None


def calculate_clash_score(pdb_file: str | Path, threshold: float = 2.4, only_ca: bool = False) -> int:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_file))
    coords = []
    atom_info = []
    for atom in structure.get_atoms():
        if _element(atom) == "H":
            continue
        if only_ca and atom.get_name() != "CA":
            continue
        residue = atom.get_parent()
        chain = residue.get_parent()
        coords.append(atom.coord)
        atom_info.append((chain.id, residue.id[1], atom.get_name()))
    if not coords:
        return 0

    tree = cKDTree(np.array(coords))
    valid_pairs = 0
    for i, j in tree.query_pairs(float(threshold)):
        chain_i, res_i, _ = atom_info[i]
        chain_j, res_j, _ = atom_info[j]
        if chain_i == chain_j and res_i == res_j:
            continue
        if chain_i == chain_j and abs(res_i - res_j) == 1:
            continue
        if not only_ca and chain_i == chain_j:
            continue
        valid_pairs += 1
    return valid_pairs


def pr_alternative_score_interface(pdb_file: str | Path, binder_chain: str = "B", target_chain: str = "A", sasa_engine: str = "auto"):
    interface_residues = hotspot_residues(pdb_file, binder_chain=binder_chain, target_chain=target_chain)
    interface_residue_ids = [f"{binder_chain}{resnum}" for resnum in interface_residues.keys()]
    interface_residue_ids_str = ",".join(interface_residue_ids)

    interface_aa = {aa: 0 for aa in "ACDEFGHIKLMNPQRSTVWY"}
    for aa in interface_residues.values():
        if aa in interface_aa:
            interface_aa[aa] += 1

    if str(sasa_engine).lower() == "freesasa":
        sasa_values = _compute_sasa_metrics_with_freesasa(pdb_file, binder_chain, target_chain)
    elif str(sasa_engine).lower() == "biopython":
        sasa_values = _compute_sasa_metrics(pdb_file, binder_chain, target_chain)
    elif HAS_FREESASA:
        sasa_values = _compute_sasa_metrics_with_freesasa(pdb_file, binder_chain, target_chain)
    else:
        sasa_values = _compute_sasa_metrics(pdb_file, binder_chain, target_chain)

    (
        surface_hydrophobicity_fraction,
        binder_sasa_in_complex,
        binder_sasa_monomer,
        target_sasa_in_complex,
        target_sasa_monomer,
    ) = sasa_values

    interface_binder_dSASA = max(binder_sasa_monomer - binder_sasa_in_complex, 0.0)
    interface_target_dSASA = max(target_sasa_monomer - target_sasa_in_complex, 0.0)
    interface_total_dSASA = interface_binder_dSASA + interface_target_dSASA
    interface_binder_fraction = (
        interface_total_dSASA / binder_sasa_in_complex * 100.0 if binder_sasa_in_complex > 0.0 else 0.0
    )
    interface_nres = len(interface_residue_ids)
    hbond_stats = count_hbonds_and_buried_unsats(pdb_file, binder_chain=binder_chain, target_chain=target_chain)

    interface_hbond_percentage = (
        hbond_stats.binder_target_hbonds / interface_nres * 100.0 if interface_nres else None
    )
    interface_bunsch_percentage = (
        hbond_stats.buried_unsat_hbonds / interface_nres * 100.0 if interface_nres else None
    )

    hydrophobic_count = sum(interface_aa[aa] for aa in HYDROPHOBIC_AA_SET)
    interface_scores = {
        "binder_score": -1.0,
        "surface_hydrophobicity": surface_hydrophobicity_fraction,
        "interface_sc": _calculate_shape_complementarity(pdb_file, binder_chain, target_chain),
        "interface_packstat": 0.65,
        "interface_dG": -10.0,
        "interface_dSASA": interface_total_dSASA,
        "interface_dG_SASA_ratio": 0.0,
        "interface_fraction": interface_binder_fraction,
        "interface_hydrophobicity": hydrophobic_count / interface_nres * 100.0 if interface_nres else 0.0,
        "interface_nres": interface_nres,
        "interface_interface_hbonds": hbond_stats.binder_target_hbonds,
        "interface_hbond_percentage": interface_hbond_percentage,
        "interface_delta_unsat_hbonds": hbond_stats.buried_unsat_hbonds,
        "interface_delta_unsat_hbonds_percentage": interface_bunsch_percentage,
    }
    interface_scores = {key: round(value, 2) if isinstance(value, float) else value for key, value in interface_scores.items()}
    return interface_scores, interface_aa, interface_residue_ids_str


def hotspot_residues(
    pdb_file: str | Path,
    binder_chain: str = "B",
    target_chain: str = "A",
    atom_distance_cutoff: float = 4.0,
) -> dict[int, str]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", str(pdb_file))
    model = structure[0]
    if binder_chain not in model or target_chain not in model:
        return {}
    binder_atoms = Selection.unfold_entities(model[binder_chain], "A")
    target_atoms = Selection.unfold_entities(model[target_chain], "A")
    if not binder_atoms or not target_atoms:
        return {}
    binder_coords = np.array([atom.coord for atom in binder_atoms])
    target_coords = np.array([atom.coord for atom in target_atoms])
    pairs = cKDTree(binder_coords).query_ball_tree(cKDTree(target_coords), atom_distance_cutoff)

    interacting = {}
    for binder_index, close_indices in enumerate(pairs):
        if not close_indices:
            continue
        residue = binder_atoms[binder_index].get_parent()
        aa = THREE_TO_ONE.get(residue.get_resname())
        if aa:
            interacting[residue.id[1]] = aa
    return interacting


def openmm_relax_subprocess(
    pdb_file_path: str | Path,
    output_pdb_path: str | Path,
    use_gpu_relax: bool = True,
    timeout: int | None = None,
    max_attempts: int = 3,
    use_faspr_repack: bool = True,
    starting_pdb: str | Path | None = None,
    target_chains: str | None = None,
) -> None:
    pdb_file_path = str(Path(pdb_file_path).resolve())
    output_pdb_path = str(Path(output_pdb_path).resolve())
    code = (
        "from scaffold_bind_craft.openmm_screen_utils import openmm_relax; "
        f"openmm_relax({pdb_file_path!r}, {output_pdb_path!r}, "
        f"use_gpu_relax={bool(use_gpu_relax)!r}, use_faspr_repack={bool(use_faspr_repack)!r})"
    )
    fallback_signature = "[OpenMM-Relax] ERROR; copied input to output"
    env = os.environ.copy()
    if starting_pdb:
        env["BINDCRAFT_STARTING_PDB"] = str(Path(starting_pdb).resolve())
    if target_chains:
        env["BINDCRAFT_TARGET_CHAINS"] = str(target_chains)
    attempts = max(1, int(max_attempts))
    for attempt in range(1, attempts + 1):
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        if proc.stdout:
            sys.stdout.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        combined = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode == 0 and fallback_signature not in combined:
            return
        if attempt == attempts:
            if fallback_signature in combined:
                raise RuntimeError(f"OpenMM relax fell back to copying input after {attempts} attempts")
            raise RuntimeError(f"OpenMM relax failed after {attempts} attempts")
        if os.path.exists(output_pdb_path):
            os.remove(output_pdb_path)
        time.sleep(0.5)


def openmm_relax(
    pdb_file_path: str | Path,
    output_pdb_path: str | Path,
    use_gpu_relax: bool = True,
    openmm_max_iterations: int = 1000,
    openmm_ramp_force_tolerance_kj_mol_nm: float = 2.39,
    openmm_final_force_tolerance_kj_mol_nm: float = 0.8,
    restraint_k_kcal_mol_A2: float = 0.0,
    restraint_ramp_factors: tuple[float, ...] = (0.0,),
    md_steps_per_shake: int = 0.0,
    lj_rep_base_k_kj_mol: float = 0.0,
    lj_rep_ramp_factors: tuple[float, ...] = (0.0,),
    use_faspr_repack: bool = False,
    post_faspr_minimize: bool = True,
) -> str | None:
    pdb_file_path = str(pdb_file_path)
    output_pdb_path = str(output_pdb_path)
    platform_name = None
    deconcat_tmp = None
    reconcat_spec = None
    try:
        original_b_factors = _residue_b_factors(pdb_file_path)
        pdb_for_fixer = pdb_file_path
        starting_pdb = os.environ.get("BINDCRAFT_STARTING_PDB")
        target_chains = os.environ.get("BINDCRAFT_TARGET_CHAINS")
        if starting_pdb and os.path.isfile(starting_pdb) and target_chains:
            lengths = compute_target_segment_lengths(starting_pdb, target_chains)
            if len([length for length in lengths if length > 0]) >= 2:
                import string

                chain_pool = [chain_id for chain_id in string.ascii_uppercase if chain_id not in {"A", "B"}]
                chain_pool += list("0123456789") + list(string.ascii_lowercase)
                new_chain_ids = chain_pool[: len(lengths)]
                tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
                tmp.close()
                deconcat_tmp = tmp.name
                split_chain_into_subchains(pdb_file_path, "A", lengths, new_chain_ids, deconcat_tmp)
                if os.path.isfile(deconcat_tmp) and os.path.getsize(deconcat_tmp) > 0:
                    pdb_for_fixer = deconcat_tmp
                    reconcat_spec = (new_chain_ids, "A")
                else:
                    deconcat_tmp = None

        fixer = PDBFixer(filename=pdb_file_path)
        if pdb_for_fixer != pdb_file_path:
            fixer = PDBFixer(filename=pdb_for_fixer)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)

        forcefield = _get_openmm_forcefield()
        system = forcefield.createSystem(
            fixer.topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=app.HBonds,
        )
        original_sigmas = []
        nonbonded_force_index = -1
        for force_index in range(system.getNumForces()):
            force = system.getForce(force_index)
            if isinstance(force, openmm.NonbondedForce):
                nonbonded_force_index = force_index
                for particle_index in range(force.getNumParticles()):
                    _, sigma, _ = force.getParticleParameters(particle_index)
                    original_sigmas.append(sigma.value_in_unit(unit.nanometer))
                break
        lj_force, lj_index = _create_lj_repulsive_force(
            system,
            lj_rep_base_k_kj_mol,
            lj_rep_ramp_factors,
            original_sigmas,
            nonbonded_force_index,
        )
        restraint_force, restraint_index = _create_backbone_restraint_force(system, fixer, restraint_k_kcal_mol_A2)
        integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
        simulation, platform_name = _create_simulation(fixer.topology, system, integrator, use_gpu_relax)
        simulation.context.setPositions(fixer.positions)

        if restraint_k_kcal_mol_A2 > 0 or lj_rep_base_k_kj_mol > 0:
            if lj_force is not None and lj_index != -1:
                lj_force.setGlobalParameterDefaultValue(lj_index, 0.0)
                lj_force.updateParametersInContext(simulation.context)
                simulation.context.setParameter("k_rep_lj", 0.0)
            if restraint_force is not None and restraint_index != -1:
                restraint_value = _k_kj_per_nm2(restraint_k_kcal_mol_A2)
                restraint_force.setGlobalParameterDefaultValue(restraint_index, restraint_value)
                restraint_force.updateParametersInContext(simulation.context)
                simulation.context.setParameter("k_restraint", restraint_value)
            simulation.minimizeEnergy(
                tolerance=openmm_ramp_force_tolerance_kj_mol_nm * unit.kilojoule_per_mole / unit.nanometer,
                maxIterations=openmm_max_iterations,
            )

        best_energy = None
        best_positions = None
        restraint_factors = restraint_ramp_factors if restraint_k_kcal_mol_A2 > 0 and restraint_ramp_factors else (0.0,)
        lj_factors = lj_rep_ramp_factors if lj_rep_base_k_kj_mol > 0 and lj_rep_ramp_factors else (0.0,)
        ramp_pairs = list(zip_longest(restraint_factors, lj_factors, fillvalue=0.0))
        for index, (restraint_factor, lj_factor) in enumerate(ramp_pairs):
            if lj_force is not None and lj_index != -1:
                lj_value = lj_rep_base_k_kj_mol * lj_factor
                lj_force.setGlobalParameterDefaultValue(lj_index, lj_value)
                lj_force.updateParametersInContext(simulation.context)
                simulation.context.setParameter("k_rep_lj", lj_value)
            if restraint_force is not None and restraint_index != -1:
                restraint_value = _k_kj_per_nm2(restraint_k_kcal_mol_A2 * restraint_factor)
                restraint_force.setGlobalParameterDefaultValue(restraint_index, restraint_value)
                restraint_force.updateParametersInContext(simulation.context)
                simulation.context.setParameter("k_restraint", restraint_value)
            if md_steps_per_shake > 0 and index < 2:
                simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
                simulation.step(int(md_steps_per_shake))
            tolerance = (
                openmm_final_force_tolerance_kj_mol_nm
                if index == len(ramp_pairs) - 1
                else openmm_ramp_force_tolerance_kj_mol_nm
            )
            energy = _chunked_minimize(simulation, tolerance, openmm_max_iterations)
            if best_energy is None or energy < best_energy:
                best_energy = energy
                best_positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)

        if best_positions is not None:
            simulation.context.setPositions(best_positions)
        positions = simulation.context.getState(getPositions=True).getPositions()
        with open(output_pdb_path, "w") as handle:
            app.PDBFile.writeFile(simulation.topology, positions, handle, keepIds=True)

        if use_faspr_repack:
            _run_faspr_pipeline(
                output_pdb_path,
                post_faspr_minimize=post_faspr_minimize,
                force_tolerance_kj_mol_nm=openmm_final_force_tolerance_kj_mol_nm,
            )

        if deconcat_tmp and os.path.isfile(deconcat_tmp):
            biopython_align_all_ca(deconcat_tmp, output_pdb_path)
        else:
            biopython_align_all_ca(pdb_file_path, output_pdb_path)
        _apply_residue_b_factors(output_pdb_path, original_b_factors)
        if reconcat_spec:
            source_ids, dest_id = reconcat_spec
            merge_chains_into_single(output_pdb_path, source_ids, dest_id, output_pdb_path)
        _append_connectivity_records(output_pdb_path)
        clean_pdb(output_pdb_path)
        return platform_name
    except Exception as exc:
        shutil.copy(pdb_file_path, output_pdb_path)
        print(f"[OpenMM-Relax] ERROR; copied input to output for {Path(pdb_file_path).name}")
        print(f"[OpenMM-Relax] ERROR; exception {exc}")
        return platform_name
    finally:
        if deconcat_tmp and os.path.isfile(deconcat_tmp):
            with contextlib.suppress(Exception):
                os.remove(deconcat_tmp)
        gc.collect()


def _compute_sasa_metrics(pdb_file_path: str | Path, binder_chain: str, target_chain: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", str(pdb_file_path))
    model = structure[0]
    sr_complex = ShrakeRupley(probe_radius=1.4, n_points=960, radii_dict=R_CHOTHIA)
    sr_complex.compute(model, level="A")
    binder_sasa_in_complex = _chain_total_sasa(model[binder_chain]) if binder_chain in model else 0.0
    target_sasa_in_complex = _chain_total_sasa(model[target_chain]) if target_chain in model else 0.0

    binder_sasa_monomer = 0.0
    target_sasa_monomer = 0.0
    surface_hydrophobicity_fraction = 0.0
    if binder_chain in model:
        binder_model, binder_only = _single_chain_model(model[binder_chain])
        ShrakeRupley(probe_radius=1.4, n_points=960, radii_dict=R_CHOTHIA).compute(binder_model, level="A")
        binder_sasa_monomer = _chain_total_sasa(binder_only)
        hydrophobic_sasa = 0.0
        for residue in binder_only:
            if Polypeptide.is_aa(residue, standard=True):
                try:
                    aa = seq1(residue.get_resname()).upper()
                except Exception:
                    aa = ""
                if aa in HYDROPHOBIC_AA_SET:
                    hydrophobic_sasa += sum(getattr(atom, "sasa", 0.0) for atom in residue.get_atoms())
        surface_hydrophobicity_fraction = (
            hydrophobic_sasa / binder_sasa_monomer if binder_sasa_monomer > 0.0 else 0.0
        )
    if target_chain in model:
        target_model, target_only = _single_chain_model(model[target_chain])
        ShrakeRupley(probe_radius=1.4, n_points=960, radii_dict=R_CHOTHIA).compute(target_model, level="A")
        target_sasa_monomer = _chain_total_sasa(target_only)

    return (
        surface_hydrophobicity_fraction,
        binder_sasa_in_complex,
        binder_sasa_monomer,
        target_sasa_in_complex,
        target_sasa_monomer,
    )


def _compute_sasa_metrics_with_freesasa(pdb_file_path: str | Path, binder_chain: str, target_chain: str):
    if not HAS_FREESASA:
        return _compute_sasa_metrics(pdb_file_path, binder_chain, target_chain)
    tmp_paths = []
    try:
        structure_complex = freesasa.Structure(str(pdb_file_path))  # type: ignore[name-defined]
        result_complex = freesasa.calc(structure_complex)  # type: ignore[name-defined]
        sel_area = freesasa.selectArea(  # type: ignore[name-defined]
            [f"binder, chain {binder_chain}", f"target, chain {target_chain}"],
            structure_complex,
            result_complex,
        )
        binder_sasa_in_complex = float(sel_area.get("binder", 0.0))
        target_sasa_in_complex = float(sel_area.get("target", 0.0))

        parser = PDBParser(QUIET=True)
        model = parser.get_structure("complex_for_freesasa", str(pdb_file_path))[0]
        binder_sasa_monomer = 0.0
        target_sasa_monomer = 0.0
        surface_hydrophobicity_fraction = 0.0
        if binder_chain in model:
            binder_path = _write_temp_chain(model[binder_chain])
            tmp_paths.append(binder_path)
            binder_structure = freesasa.Structure(binder_path)  # type: ignore[name-defined]
            binder_result = freesasa.calc(binder_structure)  # type: ignore[name-defined]
            binder_sasa_monomer = float(binder_result.totalArea())
            with _suppress_stderr():
                hydro = freesasa.selectArea(  # type: ignore[name-defined]
                    ["hydro, resn ala+val+leu+ile+met+phe+pro+trp+tyr+cys+gly"],
                    binder_structure,
                    binder_result,
                )
            surface_hydrophobicity_fraction = (
                float(hydro.get("hydro", 0.0)) / binder_sasa_monomer if binder_sasa_monomer else 0.0
            )
        if target_chain in model:
            target_path = _write_temp_chain(model[target_chain])
            tmp_paths.append(target_path)
            target_structure = freesasa.Structure(target_path)  # type: ignore[name-defined]
            target_result = freesasa.calc(target_structure)  # type: ignore[name-defined]
            target_sasa_monomer = float(target_result.totalArea())
        return (
            surface_hydrophobicity_fraction,
            binder_sasa_in_complex,
            binder_sasa_monomer,
            target_sasa_in_complex,
            target_sasa_monomer,
        )
    except Exception:
        return _compute_sasa_metrics(pdb_file_path, binder_chain, target_chain)
    finally:
        for path in tmp_paths:
            with contextlib.suppress(Exception):
                os.remove(path)


def _calculate_shape_complementarity(pdb_file_path: str | Path, binder_chain: str, target_chain: str) -> float:
    candidates = [
        os.environ.get("SC_RS_BIN"),
        os.environ.get("SC_BIN"),
        shutil.which("sc"),
        shutil.which("sc-rs"),
        shutil.which("shape-complementarity"),
    ]
    sc_bin = next((candidate for candidate in candidates if candidate and os.path.isfile(candidate)), None)
    if sc_bin is None:
        return 0.70
    try:
        proc = subprocess.run(
            [sc_bin, str(pdb_file_path), str(target_chain), str(binder_chain), "--json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        payload = json.loads(proc.stdout.strip())
        value = float(payload.get("sc", payload.get("sc_value", 0.70)))
        return value if 0.0 <= value <= 1.0 else 0.70
    except Exception:
        return 0.70


def _get_openmm_forcefield():
    global _OPENMM_FORCEFIELD
    if _OPENMM_FORCEFIELD is None:
        _OPENMM_FORCEFIELD = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
    return _OPENMM_FORCEFIELD


def _create_simulation(topology, system, integrator, use_gpu_relax: bool):
    platform_names = []
    if use_gpu_relax:
        platform_names.extend([name.strip() for name in os.environ.get("OPENMM_PLATFORM_ORDER", "OpenCL,CUDA").split(",") if name.strip()])
    platform_names.append("CPU")
    last_error = None
    for platform_name in platform_names:
        try:
            platform = Platform.getPlatformByName(platform_name)
            properties = {}
            if platform_name == "CUDA":
                properties = {"CudaPrecision": "mixed"}
            elif platform_name == "OpenCL":
                properties = {"OpenCLPrecision": "single"}
            return app.Simulation(topology, system, integrator, platform, properties), platform_name
        except Exception as exc:
            last_error = exc
    raise last_error or OpenMMException("No usable OpenMM platform")


def _chunked_minimize(simulation, force_tolerance_kj_mol_nm: float, max_iterations: int):
    tolerance = force_tolerance_kj_mol_nm * unit.kilojoule_per_mole / unit.nanometer
    per_call_iterations = 200 if max_iterations == 0 or max_iterations > 200 else max_iterations
    remaining = max_iterations
    small_improvement_streak = 0
    last_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    while True:
        simulation.minimizeEnergy(tolerance=tolerance, maxIterations=per_call_iterations)
        current_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        try:
            if last_energy - current_energy < 0.1 * unit.kilojoule_per_mole:
                small_improvement_streak += 1
            else:
                small_improvement_streak = 0
        except Exception:
            small_improvement_streak = 3
        last_energy = current_energy
        if max_iterations > 0:
            remaining -= per_call_iterations
            if remaining <= 0:
                break
        if small_improvement_streak >= 3:
            break
    return last_energy


def _create_lj_repulsive_force(system, base_k_kj_mol: float, ramp_factors, original_sigmas, nonbonded_force_index: int):
    if base_k_kj_mol <= 0 or not original_sigmas or not ramp_factors:
        return None, -1

    force = openmm.CustomNonbondedForce("k_rep_lj * (((sigma_particle1 + sigma_particle2) * 0.5 / r)^12)")
    index = force.addGlobalParameter("k_rep_lj", float(base_k_kj_mol * ramp_factors[0]))
    force.addPerParticleParameter("sigma_particle")
    for sigma in original_sigmas:
        force.addParticle([sigma])

    if nonbonded_force_index != -1:
        nonbonded_force = system.getForce(nonbonded_force_index)
        method = nonbonded_force.getNonbondedMethod()
        if method == openmm.NonbondedForce.CutoffPeriodic:
            force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
            force.setCutoffDistance(nonbonded_force.getCutoffDistance())
            force.setUseSwitchingFunction(nonbonded_force.getUseSwitchingFunction())
            if nonbonded_force.getUseSwitchingFunction():
                force.setSwitchingDistance(nonbonded_force.getSwitchingDistance())
        elif method == openmm.NonbondedForce.CutoffNonPeriodic:
            force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffNonPeriodic)
            force.setCutoffDistance(nonbonded_force.getCutoffDistance())
        else:
            force.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
        for exception_index in range(nonbonded_force.getNumExceptions()):
            p1, p2, *_ = nonbonded_force.getExceptionParameters(exception_index)
            force.addExclusion(p1, p2)
    else:
        force.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)

    force.setForceGroup(2)
    system.addForce(force)
    return force, index


def _create_backbone_restraint_force(system, fixer, restraint_k_kcal_mol_A2: float):
    if restraint_k_kcal_mol_A2 <= 0:
        return None, -1
    force = openmm.CustomExternalForce(
        "0.5 * k_restraint * ((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0))"
    )
    index = force.addGlobalParameter("k_restraint", _k_kj_per_nm2(restraint_k_kcal_mol_A2))
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    backbone = {"N", "CA", "C", "O"}
    count = 0
    for atom in fixer.topology.atoms():
        if atom.name in backbone:
            xyz = fixer.positions[atom.index].value_in_unit(unit.nanometer)
            force.addParticle(atom.index, [xyz[0], xyz[1], xyz[2]])
            count += 1
    if count == 0:
        return None, -1
    system.addForce(force)
    return force, index


def _resolve_faspr_binary():
    env_bin = os.environ.get("FASPR_BIN")
    if env_bin and os.path.isfile(env_bin) and os.access(env_bin, os.X_OK):
        return env_bin, os.path.dirname(os.path.abspath(env_bin))
    which = shutil.which("FASPR")
    if which and os.path.isfile(which) and os.access(which, os.X_OK):
        return which, os.path.dirname(os.path.abspath(which))
    return None, None


def _run_faspr(input_pdb_path: str, output_pdb_path: str, timeout: int = 900) -> bool:
    faspr_bin, faspr_dir = _resolve_faspr_binary()
    if not faspr_bin or not faspr_dir:
        return False
    try:
        proc = subprocess.run(
            [faspr_bin, "-i", os.path.abspath(input_pdb_path), "-o", os.path.abspath(output_pdb_path)],
            cwd=faspr_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode == 0 and os.path.isfile(output_pdb_path) and os.path.getsize(output_pdb_path) > 0
    except Exception:
        return False


def _run_faspr_pipeline(
    output_pdb_path: str,
    post_faspr_minimize: bool,
    force_tolerance_kj_mol_nm: float,
) -> None:
    tmp_dir = tempfile.mkdtemp(prefix="faspr_")
    tmp_heavy = os.path.join(tmp_dir, "input_heavy.pdb")
    tmp_faspr_out = os.path.join(tmp_dir, "faspr_out.pdb")
    try:
        try:
            fixer = PDBFixer(filename=output_pdb_path)
            fixer.findMissingResidues()
            fixer.findNonstandardResidues()
            fixer.replaceNonstandardResidues()
            fixer.removeHeterogens(keepWater=False)
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            with open(tmp_heavy, "w") as handle:
                app.PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
        except Exception:
            shutil.copy(output_pdb_path, tmp_heavy)

        if not _run_faspr(tmp_heavy, tmp_faspr_out):
            return
        if post_faspr_minimize:
            _add_hydrogens_and_minimize(
                tmp_faspr_out,
                output_pdb_path,
                force_tolerance_kj_mol_nm=force_tolerance_kj_mol_nm,
                max_iterations=300,
            )
        else:
            fixer = PDBFixer(filename=tmp_faspr_out)
            fixer.findMissingResidues()
            fixer.findNonstandardResidues()
            fixer.replaceNonstandardResidues()
            fixer.removeHeterogens(keepWater=False)
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(7.0)
            with open(output_pdb_path, "w") as handle:
                app.PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
    except Exception:
        return
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _add_hydrogens_and_minimize(
    pdb_in_path: str,
    pdb_out_path: str,
    force_tolerance_kj_mol_nm: float = 0.1,
    max_iterations: int = 500,
) -> None:
    fixer = PDBFixer(filename=pdb_in_path)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    forcefield = _get_openmm_forcefield()
    system = forcefield.createSystem(
        fixer.topology,
        nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
    )
    integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    simulation, _ = _create_simulation(fixer.topology, system, integrator, use_gpu_relax=True)
    simulation.context.setPositions(fixer.positions)
    simulation.minimizeEnergy(
        tolerance=force_tolerance_kj_mol_nm * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=max_iterations,
    )
    positions = simulation.context.getState(getPositions=True).getPositions()
    with open(pdb_out_path, "w") as handle:
        app.PDBFile.writeFile(simulation.topology, positions, handle, keepIds=True)


def biopython_align_all_ca(reference_pdb_path: str | Path, pdb_to_align_path: str | Path) -> None:
    parser = PDBParser(QUIET=True)
    reference_structure = parser.get_structure("reference", str(reference_pdb_path))
    align_structure = parser.get_structure("align", str(pdb_to_align_path))
    ref_atoms = [
        residue["CA"]
        for residue in reference_structure.get_residues()
        if Polypeptide.is_aa(residue, standard=True) and "CA" in residue
    ]
    align_atoms = [
        residue["CA"]
        for residue in align_structure.get_residues()
        if Polypeptide.is_aa(residue, standard=True) and "CA" in residue
    ]
    length = min(len(ref_atoms), len(align_atoms))
    if length == 0:
        return
    superimposer = Superimposer()
    superimposer.set_atoms(ref_atoms[:length], align_atoms[:length])
    superimposer.apply(align_structure.get_atoms())
    io = PDBIO()
    io.set_structure(align_structure)
    io.save(str(pdb_to_align_path))


def clean_pdb(pdb_path: str | Path) -> None:
    with open(pdb_path) as handle:
        lines = handle.readlines()
    cleaned = [line for line in lines if line.startswith(("ATOM", "HETATM", "MODEL", "TER", "SSBOND", "CONECT", "LINK", "END"))]
    if not cleaned or not cleaned[-1].startswith("END"):
        cleaned.append("END\n")
    with open(pdb_path, "w") as handle:
        handle.writelines(cleaned)


def _residue_b_factors(source_pdb: str | Path) -> dict:
    parser = PDBParser(QUIET=True)
    source = parser.get_structure("source", str(source_pdb))
    b_factors = {}
    for residue in source.get_residues():
        if Polypeptide.is_aa(residue, standard=True):
            atom = residue["CA"] if "CA" in residue else next(residue.get_atoms(), None)
            if atom is not None:
                b_factors[(residue.get_parent().id, residue.id)] = atom.get_bfactor()
    return b_factors


def _apply_residue_b_factors(dest_pdb: str | Path, b_factors: dict) -> None:
    if not b_factors:
        return
    parser = PDBParser(QUIET=True)
    dest = parser.get_structure("dest", str(dest_pdb))
    changed = False
    for residue in dest.get_residues():
        b_factor = b_factors.get((residue.get_parent().id, residue.id))
        if b_factor is not None:
            for atom in residue:
                atom.set_bfactor(b_factor)
            changed = True
    if changed:
        io = PDBIO()
        io.set_structure(dest)
        io.save(str(dest_pdb))


def _copy_residue_b_factors(source_pdb: str | Path, dest_pdb: str | Path) -> None:
    _apply_residue_b_factors(dest_pdb, _residue_b_factors(source_pdb))


def _append_connectivity_records(pdb_path: str | Path) -> None:
    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
    tmp.close()
    tmp_path = tmp.name
    try:
        fixer = PDBFixer(filename=str(pdb_path))
        with open(tmp_path, "w") as handle:
            app.PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
        with open(tmp_path) as handle:
            conn_lines = [line for line in handle if line.startswith(("SSBOND", "CONECT", "LINK"))]
        if not conn_lines:
            return
        with open(pdb_path) as handle:
            lines = handle.readlines()
        existing = {line for line in lines if line.startswith(("SSBOND", "CONECT", "LINK"))}
        to_add = [line for line in conn_lines if line not in existing]
        if not to_add:
            return
        end_index = next((index for index, line in enumerate(lines) if line.startswith("END")), None)
        if end_index is None:
            lines.extend(to_add)
        else:
            lines = lines[:end_index] + to_add + lines[end_index:]
        with open(pdb_path, "w") as handle:
            handle.writelines(lines)
    except Exception:
        return
    finally:
        with contextlib.suppress(Exception):
            os.remove(tmp_path)


def _single_chain_model(chain):
    structure = Structure.Structure("single")
    model = Model.Model(0)
    chain_copy = copy.deepcopy(chain)
    model.add(chain_copy)
    structure.add(model)
    return model, chain_copy


def _write_temp_chain(chain) -> str:
    structure = Structure.Structure("single")
    model = Model.Model(0)
    model.add(copy.deepcopy(chain))
    structure.add(model)
    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
    tmp.close()
    io = PDBIO()
    io.set_structure(structure)
    io.save(tmp.name)
    return tmp.name


def compute_target_chain_lengths(starting_pdb_path: str | Path, chains: str) -> list[int]:
    try:
        chain_ids = [chain.strip() for chain in str(chains).split(",") if chain.strip()]
        parser = PDBParser(QUIET=True)
        model = parser.get_structure("start", str(starting_pdb_path))[0]
        lengths = []
        for chain_id in chain_ids:
            if chain_id not in model:
                lengths.append(0)
                continue
            lengths.append(sum(1 for residue in model[chain_id] if Polypeptide.is_aa(residue, standard=True)))
        return lengths
    except Exception:
        return []


def compute_target_segment_lengths(starting_pdb_path: str | Path, chains: str) -> list[int]:
    try:
        chain_ids = [chain.strip() for chain in str(chains).split(",") if chain.strip()]
        parser = PDBParser(QUIET=True)
        model = parser.get_structure("start", str(starting_pdb_path))[0]
        lengths = []
        for chain_id in chain_ids:
            if chain_id not in model:
                continue
            seg_len = 0
            last_resseq = None
            for residue in model[chain_id]:
                if not Polypeptide.is_aa(residue, standard=True):
                    continue
                resseq = residue.id[1]
                if last_resseq is not None and isinstance(resseq, int) and isinstance(last_resseq, int) and resseq - last_resseq > 1:
                    if seg_len > 0:
                        lengths.append(seg_len)
                    seg_len = 1
                else:
                    seg_len += 1
                last_resseq = resseq
            if seg_len > 0:
                lengths.append(seg_len)
        return lengths
    except Exception:
        return []


def split_chain_into_subchains(
    pdb_in_path: str | Path,
    source_chain_id: str,
    subchain_lengths,
    new_chain_ids,
    output_path: str | Path | None = None,
) -> None:
    try:
        lengths = [int(length) for length in subchain_lengths if int(length) > 0]
        chain_names = [str(chain_id) for chain_id in new_chain_ids]
        if not lengths or len(lengths) != len(chain_names):
            return
        cumulative = []
        running = 0
        for length in lengths:
            running += length
            cumulative.append(running)

        def subchain_index_for(count: int) -> int:
            for index, bound in enumerate(cumulative):
                if count <= bound:
                    return index
            return len(cumulative) - 1

        with open(pdb_in_path) as handle:
            lines = handle.readlines()
        out_lines = []
        residue_counter = 0
        current_residue = None
        for line in lines:
            if line.startswith(("ATOM", "HETATM")) and len(line) >= 27 and line[21:22] == source_chain_id:
                try:
                    resseq = int(line[22:26])
                except Exception:
                    resseq = None
                residue_id = (resseq, line[26:27])
                if residue_id != current_residue:
                    residue_counter += 1
                    current_residue = residue_id
                new_chain = chain_names[subchain_index_for(residue_counter)][:1]
                line = line[:21] + new_chain + line[22:]
            out_lines.append(line)
        with open(output_path or pdb_in_path, "w") as handle:
            handle.writelines(out_lines)
    except Exception:
        return


def merge_chains_into_single(
    pdb_in_path: str | Path,
    source_chain_ids,
    dest_chain_id: str = "A",
    output_path: str | Path | None = None,
) -> None:
    try:
        source_ids = {str(chain_id)[:1] for chain_id in source_chain_ids if str(chain_id)}
        if not source_ids:
            return
        with open(pdb_in_path) as handle:
            lines = handle.readlines()
        out_lines = []
        for line in lines:
            if line.startswith(("ATOM", "HETATM")) and len(line) >= 22 and line[21:22] in source_ids:
                line = line[:21] + dest_chain_id[:1] + line[22:]
            out_lines.append(line)
        with open(output_path or pdb_in_path, "w") as handle:
            handle.writelines(out_lines)
    except Exception:
        return


def _chain_total_sasa(chain_entity) -> float:
    return sum(getattr(atom, "sasa", 0.0) for atom in chain_entity.get_atoms())


def _k_kj_per_nm2(k_kcal_A2: float) -> float:
    return k_kcal_A2 * 4.184 * 100.0


def _element(atom) -> str:
    element = (getattr(atom, "element", "") or "").strip().upper()
    return element or atom.get_name().strip().upper()[0]


@contextlib.contextmanager
def _suppress_stderr():
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_fd = os.dup(2)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        try:
            yield
        finally:
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
    except Exception:
        yield
