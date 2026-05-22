# scaffold_bind_craft

Modular scaffold binder design pipeline with restartable HPC stages:

1. `scripts/run_colabdesign_rso.py` performs relaxed sequence optimisation with `colabdesign_modified`, fixed residue support, pLDDT early stopping, and RSO save filters for iPTM, pLDDT, PAE, and interface PAE.
2. `scripts/run_proteinmpnn.py` runs ProteinMPNN on accepted RSO structures and designs only the relaxed positions.
3. `scripts/run_chai1_offline.py` predicts pending ProteinMPNN binder-target sequences with Chai-1 offline in single-sequence mode using ESM embeddings, without MSA or template servers.
4. `scripts/run_openmm_screen.py` relaxes Chai-1 PDBs with the local OpenMM path, scores interfaces with PyRosetta-free utilities, and writes accepted relaxed structures.

Each script takes the same settings file and creates one subdirectory under `output_dir`. Existing outputs are counted before work starts, so a time-limited HPC job can be resubmitted without restarting completed work.

## Setup
You need 4 environments:
1. Colabdesign_modified
2. ProteinMPNN
3. Chai-1 offline
4. OpenMM

### Chai-1 Offline Install

Chai-1 requires Linux, Python 3.10 or later, and a CUDA GPU with bfloat16 support. Install it in the prediction environment:

```bash
git clone git@github.com:RSCJacksonLab/chai-1-offline.git

python3.10 -m venv chai1_offline
source chai1_offline/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ./chai-1-offline
python -m pip install biopython huggingface_hub
```

For an offline HPC run, download the Chai-1 assets and ESM-2 embedding weights on the `copyq` queue before prediction:

```bash
qsub environments/00_download_chai1_weights_copyq.pbs
```

Set `chai1.model_dir` or `CHAI_MODEL_DIR` to the same `/scratch/xz2/chai-1_weights/` location for prediction jobs. See `environments/gadi_chai1_offline.md` for Gadi-specific setup.

## Usage

Edit `settings/example.json`, then run each stage independently:

```bash
python3 scripts/run_colabdesign_rso.py --settings settings/my_run.json --limit 50
python3 scripts/run_proteinmpnn.py --settings settings/my_run.json --limit 50
python3 scripts/run_chai1_offline.py --settings settings/my_run.json --limit 100
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
  03_chai1/
    inputs/
    raw_outputs/
    cifs/
    predictions/
    metrics/
    chai1_metrics.csv
  04_openmm_screen/
    pdbs/
    relaxed_pdbs/
    accepted/
    screening_metrics.csv
```

## Notes

- RSO uses `rso.fixed_positions` for ColabDesign. If `fixed_positions` is `null`, it is generated as the complement of `rso.design_positions` or top-level `design_positions` within the binder chain. ProteinMPNN uses `proteinmpnn.design_positions` or top-level `design_positions`; these positions are one-indexed positions within the designed chain, matching the upstream helper script semantics.
- Chai-1 target sequences can be set explicitly with `chai1.target_sequences`. If omitted, they are extracted from `input_pdb` for `chai1.target_chains` or top-level `target_chains`.
- Chai-1 runs in single-sequence mode by default: `use_esm_embeddings=true`, `use_msa_server=false`, and `use_templates_server=false`. Set `chai1.model_dir` to the predownloaded Chai/ESM weight directory, and tune `chai1.gpu_device`, `chai1.device`, `chai1.num_trunk_recycles`, `chai1.num_diffn_timesteps`, and `chai1.num_diffn_samples` as needed for the hardware.
- Screening records Chai-1 confidence fields as `chai1_plddt`, `chai1_binder_plddt`, `chai1_target_plddt`, `chai1_chain_count`, and `chai1_aggregate_score`. Chai-1 filter rules use `threshold`.
- OpenMM screening is self-contained in `scaffold_bind_craft.openmm_screen_utils` and `scaffold_bind_craft.hbond_utils`. It records structure-specific binder-target H-bonds and approximate buried unsatisfied polar atoms from the relaxed PDB.
