# scaffold_bind_craft

Modular scaffold binder design pipeline with restartable HPC stages:

1. `scripts/run_colabdesign_rso.py` performs relaxed sequence optimisation with `colabdesign_modified`, fixed residue support, pLDDT early stopping, and RSO save filters for iPTM, pLDDT, PAE, and interface PAE.
2. `scripts/run_proteinmpnn.py` runs ProteinMPNN on accepted RSO structures and designs only the relaxed positions.
3. `scripts/run_esmfold.py` predicts pending ProteinMPNN binder-target sequences with ESMFold.
4. `scripts/run_openmm_screen.py` relaxes ESMFold PDBs with the local OpenMM path, scores interfaces with PyRosetta-free utilities, and writes accepted relaxed structures.

Each script takes the same settings file and creates one subdirectory under `output_dir`. Existing outputs are counted before work starts, so a time-limited HPC job can be resubmitted without restarting completed work.

## Setup
You need 4 environments:
1. Colabdesign_modified
2. ProteinMPNN
3. ESMFold
4. OpenMM


## Usage

Edit `settings/example.json`, then run each stage independently:

```bash
python3 scripts/run_colabdesign_rso.py --settings settings/my_run.json --limit 50
python3 scripts/run_proteinmpnn.py --settings settings/my_run.json --limit 50
python3 scripts/run_esmfold.py --settings settings/my_run.json --limit 100
python3 scripts/run_openmm_screen.py --settings settings/my_run.json --limit 100
```

`--limit` is optional. It caps the amount of new work in the current job but does not change the total target counts in the settings file.

## Output Layout

With the default stage directory names:

```text
output_dir/
  01_rso/
    accepted_pdbs/
    filtered_pdbs/
    rso_metrics.csv
    manifest.jsonl
  02_proteinmpnn/
    seqs/
    scores/
    probs/
    proteinmpnn_metrics.csv
  03_esmfold/
    inputs/
    predictions/
    metrics/
    esmfold_metrics.csv
  04_openmm_screen/
    pdbs/
    relaxed_pdbs/
    accepted/
    screening_metrics.csv
```

## Notes

- RSO uses `rso.fixed_positions` for ColabDesign. If `fixed_positions` is `null`, it is generated as the complement of `rso.design_positions` or top-level `design_positions` within the binder chain. ProteinMPNN uses `proteinmpnn.design_positions`; these positions are one-indexed positions within the designed chain, matching the upstream helper script semantics.
- ESMFold target sequences can be set explicitly with `esmfold.target_sequences`. If omitted, they are extracted from `input_pdb` for `esmfold.target_chains`.
- ESMFold runs in the active Python environment and expects `fair-esm[esmfold]` plus PyTorch. Set `esmfold.gpu_device`, `esmfold.device`, `esmfold.chunk_size`, and `esmfold.num_recycles` as needed for the hardware.
- Screening records ESMFold confidence fields as `esmfold_plddt`, `esmfold_binder_plddt`, `esmfold_target_plddt`, and `esmfold_chain_count`. ESMFold filter rules use `threshold`.
- OpenMM screening is self-contained in `scaffold_bind_craft.openmm_screen_utils` and `scaffold_bind_craft.hbond_utils`. It records structure-specific binder-target H-bonds and approximate buried unsatisfied polar atoms from the relaxed PDB.
