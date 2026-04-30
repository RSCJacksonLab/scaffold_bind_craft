# scaffold_bind_craft

Modular scaffold binder design pipeline with restartable HPC stages:

1. `scripts/run_colabdesign_rso.py` performs relaxed sequence optimisation with `colabdesign_modified`, fixed residue support, pLDDT early stopping, and RSO save filters for iPTM, pLDDT, PAE, and interface PAE.
2. `scripts/run_proteinmpnn.py` runs ProteinMPNN on accepted RSO structures and designs only the relaxed positions.
3. `scripts/run_fast_protenix.py` converts ProteinMPNN FASTA outputs into AlphaFold 3-style JSON inputs and runs pending predictions through `../fast_protenix`.
4. `scripts/run_openmm_screen.py` converts Protenix CIFs to PDB, runs the local OpenMM relax path, scores interfaces with PyRosetta-free utilities, and writes accepted relaxed structures.

Each script takes the same settings file and creates one subdirectory under `output_dir`. Existing outputs are counted before work starts, so a time-limited HPC job can be resubmitted without restarting completed work.

## Setup
You need 4 environments:
1. Colabdesign_modified
2. ProteinMPNN
3. Fast_Protenix
4. OpenMM


## Usage

Edit `settings/example.json`, then run each stage independently:

```bash
python3 scripts/run_colabdesign_rso.py --settings settings/my_run.json --limit 50
python3 scripts/run_proteinmpnn.py --settings settings/my_run.json --limit 50
python3 scripts/run_fast_protenix.py --settings settings/my_run.json --limit 100
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
  03_fast_protenix/
    af3_inputs/
    alphafast_data/
    protenix_msas/
    protenix_output/
    fast_protenix_metrics.csv
  04_openmm_screen/
    pdbs/
    relaxed_pdbs/
    accepted/
    screening_metrics.csv
```

## Notes

- RSO uses `rso.fixed_positions` for ColabDesign. If `fixed_positions` is `null`, it is generated as the complement of `rso.design_positions` or top-level `design_positions` within the binder chain. ProteinMPNN uses `proteinmpnn.design_positions`; these positions are one-indexed positions within the designed chain, matching the upstream helper script semantics.
- Fast Protenix target sequences can be set explicitly with `fast_protenix.target_sequences`. If omitted, they are extracted from `input_pdb` for `fast_protenix.target_chains`.
- Fast Protenix uses `../fast_protenix` by default through `python -m fast_protenix.cli`, so the stage can run from the adjacent checkout without installing a console script. Configure `fast_protenix.db_dir`, `fast_protenix.mmseqs_db_dir`, and `fast_protenix.alphafast_dir` for the AlphaFast preprocessing step.
- Screening records Protenix confidence fields as `protenix_plddt`, `protenix_gpde`, `protenix_ptm`, `protenix_iptm`, `protenix_ranking_score`, and `protenix_has_clash`. Compatibility aliases for selected old `boltz_*` columns are also written for older CSV consumers.
- OpenMM screening is self-contained in `scaffold_bind_craft.openmm_screen_utils` and `scaffold_bind_craft.hbond_utils`. It records structure-specific binder-target H-bonds and approximate buried unsatisfied polar atoms from the relaxed PDB.
