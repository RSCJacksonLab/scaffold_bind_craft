# scaffold_bind_craft

Modular scaffold binder design pipeline with restartable HPC stages:

1. `scripts/run_colabdesign_rso.py` performs relaxed sequence optimisation with `colabdesign_modified`, fixed residue support, pLDDT early stopping, and RSO save filters for iPTM, pLDDT, PAE, interface PAE, and optional clashes against a full multimer.
2. `scripts/run_proteinmpnn.py` runs ProteinMPNN on accepted RSO structures and designs only the relaxed positions.
3. `scripts/run_chai1_offline.py` predicts pending ProteinMPNN binder-target sequences with Chai-1 offline in single-sequence mode using ESM embeddings, without MSA or template servers, then optionally filters clashes against a full multimer. Alternatively, `scripts/run_esmfold2.py` performs Step 3 with ESMFold2.
4. Optional `scripts/run_chai1_nise.py` runs a NISE-style select/expand/repredict loop from Chai-1 structures above `nise.threshold` using ProteinMPNN plus Chai-1.
5. `scripts/run_openmm_screen.py` relaxes Chai-1 PDBs with the local OpenMM path, scores interfaces with PyRosetta-free utilities, and writes accepted relaxed structures.
6. `scripts/select_sequences.py` greedily selects a diverse subset of accepted binder sequences for wet-lab testing using Hamming distance.

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

### ESMFold2 Offline Install

ESMFold2 requires Python 3.12. On Gadi, `environments/00_install_esmfold2_copyq.pbs` builds a persistent environment and downloads the full ESMFold2/ESMC checkpoint set while network access is available. GPU jobs then run with Hugging Face and Transformers offline modes forced. See `environments/gadi_esmfold2.md`.

## Usage

Edit `settings/example.json`, then run each stage independently:

```bash
python3 scripts/run_colabdesign_rso.py --settings settings/my_run.json --limit 50
python3 scripts/run_proteinmpnn.py --settings settings/my_run.json --limit 50
python3 scripts/run_chai1_offline.py --settings settings/my_run.json --limit 100
python3 scripts/run_esmfold2.py --settings settings/my_run.json --limit 100
python3 scripts/run_chai1_nise.py --settings settings/my_run.json --limit 100
python3 scripts/run_openmm_screen.py --settings settings/my_run.json --limit 100
python3 scripts/select_sequences.py --settings settings/my_run.json --num-sequences 12
python3 scripts/select_sequences.py --settings settings/my_run.json --num-sequences 12 --metric chai1_aggregate_score
python3 scripts/select_sequences.py --settings settings/my_run.json --num-sequences 12 --metric relaxed_clashes --metric-lower-is-better
python3 scripts/select_sequences.py --settings settings/my_run.json --num-sequences 12 --fasta-heading wetlab_candidate_
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
    filtered_pdbs/
    metrics/
    chai1_metrics.csv
  03a_NISE/
    round_001/
      proteinmpnn/
      chai1/

  03_esmfold2/              # when the alternative Step 3 is used
    cifs/
    predictions/
    filtered_pdbs/
    metrics/
    esmfold2_metrics.csv
    selected_backbones.csv
    nise_metrics.csv
  04_openmm_screen/
    pdbs/
    relaxed_pdbs/
    accepted/
    screening_metrics.csv
  05_sequence_selection/
    selected_sequences.csv
    selected_sequences.fasta
    screening_metrics.csv
```

## Notes

- RSO uses `rso.fixed_positions` for ColabDesign. If `fixed_positions` is `null`, it is generated as the complement of `rso.design_positions` or top-level `design_positions` within the binder chain. ProteinMPNN uses `proteinmpnn.design_positions` or top-level `design_positions`; these positions are one-indexed positions within the designed chain, matching the upstream helper script semantics.
- Set `multimer_clash_filter.enabled=true` and `multimer_clash_filter.reference_pdb` to a full multimer PDB to apply the filter after both RSO and Chai-1. The configured target chain is aligned to the same chain in the reference, and binder heavy atoms within `clash_cutoff` Å of any non-target reference chain are counted. A design is rejected when the count is greater than or equal to `clash_count_threshold` (default `1`, so counts below the threshold pass). Rejected PDBs are retained in each stage's `filtered_pdbs/` directory.
- NISE is disabled by default with `nise.enabled=false`. It starts from `03_chai1/chai1_metrics.csv`, selects completed predictions with `nise.score_field` above `nise.threshold` (default `chai1_aggregate_score >= 0.7`), samples binder variants with ProteinMPNN, repredicts them with Chai-1, and can repeat for `nise.num_iterations`.
- NISE designs are named `NISE-x-y`, where `x` is the 1-based iteration number and `y` is the 1-based ProteinMPNN design number across that iteration.
- Each NISE metrics row and JSON records `rso_name` and `rso_pdb` for the originating RSO backbone, alongside the immediate `parent_name` and `parent_pdb`.
- Set `nise.initial_pdb` to seed NISE directly from an existing two-chain structure; `nise.initial_name` and `nise.initial_score` control its provenance label and initial parent score. Set `screening.prediction_stage` to `"nise"` to screen completed Chai-1 predictions from every NISE iteration.
- Set `proteinmpnn.input_pdb` to run the initial ProteinMPNN stage on one specific structure. Its `proteinmpnn.design_positions` are used only for that initial library; NISE can independently use `nise.design_positions="interface"` for later rounds.
- NISE can set `nise.design_positions` to `"interface"` to choose binder-chain ProteinMPNN design positions from per-backbone SASA loss at the interface. Tune `nise.interface_sasa_threshold` and optionally combine it with `nise.num_random_design_positions`; chosen positions are written to `chosen_design_positions.json`.
- Chai-1 target sequences can be set explicitly with `chai1.target_sequences`. If omitted, they are extracted from `input_pdb` for `chai1.target_chains` or top-level `target_chains`.
- ESMFold2 uses the analogous `esmfold2.target_sequences` setting and reads ProteinMPNN FASTAs identically. `esmfold2.source_path` defaults to `/Users/u7282763/repos/esm`; `esmfold2.model` may be a Hugging Face model name or a local downloaded checkpoint. Set `esmfold2.offline=true` only after the checkpoint is cached. The runner emits standard Step 3 predictions, confidence JSON, a resumable CSV, and applies the same optional multimer clash filter.
- Before accepting an ESMFold2 prediction, Step 3 validates the generated PDB contract required by PDBFixer/OpenMM: one-character chain IDs, matching chain and residue counts, standard amino-acid names, complete backbone atoms, and finite coordinates.
- To feed ESMFold2 predictions into Step 4, set `screening.prediction_stage` to `"esmfold2"`. Available scalar filters are `esmfold2_mean_plddt`, `esmfold2_binder_mean_plddt`, `esmfold2_target_mean_plddt`, `esmfold2_ptm`, `esmfold2_iptm`, `esmfold2_binder_target_iptm`, `esmfold2_mean_pae`, `esmfold2_binder_mean_pae`, `esmfold2_target_mean_pae`, and `esmfold2_interface_mean_pae`. pLDDT is reported on 0–100; PAE is in Å; TM scores are on 0–1. Per-chain pLDDT, directional chain-pair PAE, and directional chain-pair ipTM are also retained in each prediction's metrics JSON.
- Chai-1 runs in single-sequence mode by default: `use_esm_embeddings=true`, `use_msa_server=false`, and `use_templates_server=false`. Set `chai1.model_dir` to the predownloaded Chai/ESM weight directory, and tune `chai1.gpu_device`, `chai1.device`, `chai1.num_trunk_recycles`, `chai1.num_diffn_timesteps`, and `chai1.num_diffn_samples` as needed for the hardware.
- Screening records Chai-1 confidence fields as `chai1_plddt`, `chai1_binder_plddt`, `chai1_target_plddt`, `chai1_chain_count`, and `chai1_aggregate_score`. Chai-1 filter rules use `threshold`.
- OpenMM screening is self-contained in `scaffold_bind_craft.openmm_screen_utils` and `scaffold_bind_craft.hbond_utils`. It records structure-specific binder-target H-bonds and approximate buried unsatisfied polar atoms from the relaxed PDB.
- Sequence selection deduplicates accepted sequences. By default it chooses an initial sequence with the largest total distance to the pool, then repeatedly chooses the sequence with the greatest minimum Hamming distance to the selected set; all accepted sequences must have the same length. Pass `--metric COLUMN` to rank by a recorded numeric screening metric instead (larger is better by default), or add `--metric-lower-is-better` for metrics such as clash counts. Pass `--fasta-heading PREFIX` to name FASTA records `PREFIX1`, `PREFIX2`, and so on in selection order. It also copies the full screening metrics table into the selection directory, changes the chosen designs' status from `accepted` to `selected`, and records their FASTA names in a `design_alias` column.
