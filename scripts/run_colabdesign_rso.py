#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scaffold_bind_craft.config import load_settings, require, stage_dir
from scaffold_bind_craft.io_utils import append_csv, append_jsonl, read_csv_rows, read_jsonl
from scaffold_bind_craft.pdb_utils import parse_position_string

np: Any = None


def ensure_numpy() -> Any:
    global np
    if np is None:
        import numpy as numpy_module

        np = numpy_module
    return np


METRIC_FIELDS = [
    "design_name",
    "seed",
    "status",
    "loss",
    "plddt",
    "iptm",
    "pae",
    "i_pae",
    "output_pdb",
    "reason",
    "elapsed_seconds",
]


def add_colabdesign_to_path(settings: dict[str, Any]) -> None:
    path = settings.get("tools", {}).get("colabdesign_path")
    if path:
        sys.path.insert(0, str(Path(path).expanduser().resolve()))


def metric_from_log(log: dict[str, Any], *keys: str, default: float | None = None) -> float | None:
    ensure_numpy()
    for key in keys:
        if key in log:
            value = log[key]
            try:
                return float(np.mean(value))
            except Exception:
                return default
    return default


def best_log(af_model: Any) -> dict[str, Any]:
    try:
        return dict(af_model._tmp["best"]["aux"]["log"])
    except Exception:
        return {}


def best_plddt(af_model: Any, binder_length: int | None = None) -> float | None:
    ensure_numpy()
    try:
        plddt = np.asarray(af_model._tmp["best"]["aux"]["plddt"], dtype=float)
        if binder_length:
            plddt = plddt[-int(binder_length) :]
        return float(np.mean(plddt))
    except Exception:
        log = best_log(af_model)
        return metric_from_log(log, "plddt", "pLDDT")


def normalise_metric(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def passes_threshold(value: float | None, threshold: float | None, higher: bool) -> bool:
    if threshold is None or value is None:
        return True
    return value >= threshold if higher else value <= threshold


def below_threshold(value: float | None, threshold: float | None) -> bool:
    return threshold is not None and value is not None and value < float(threshold)


def fixed_positions_from_design_positions(settings: dict[str, Any], rso: dict[str, Any]) -> str | None:
    fixed_positions = rso.get("fixed_positions", settings.get("fixed_positions", None))
    if fixed_positions is not None:
        return fixed_positions

    design_positions = parse_position_string(rso.get("design_positions", settings.get("design_positions", None)))
    if not design_positions:
        return None

    binder_chain = str(rso.get("binder_chain", settings.get("binder_chain", "B")))
    chain_ids = [chain.strip() for chain in binder_chain.split(",") if chain.strip()]
    residues = binder_residue_ids(require(settings, "input_pdb"), chain_ids)
    if not residues:
        binder_length = rso.get("binder_length")
        if binder_length is None:
            raise ValueError("Cannot infer RSO fixed_positions complement without binder residues or rso.binder_length")
        residues = [(chain_ids[0], index) for index in range(1, int(binder_length) + 1)]

    design_set = set(design_positions)
    invalid_positions = sorted(pos for pos in design_set if pos < 1 or pos > len(residues))
    if invalid_positions:
        raise ValueError(
            f"design_positions contains positions outside binder length {len(residues)}: {invalid_positions}"
        )
    fixed_residues = [
        (chain_id, resseq)
        for ordinal, (chain_id, resseq) in enumerate(residues, start=1)
        if ordinal not in design_set
    ]
    return compact_chain_positions(fixed_residues)


def binder_residue_ids(pdb_path: str | Path, chain_ids: list[str]) -> list[tuple[str, int]]:
    from Bio.PDB import PDBParser, Polypeptide

    parser = PDBParser(QUIET=True)
    model = parser.get_structure("input", str(pdb_path))[0]
    residues: list[tuple[str, int]] = []
    for chain_id in chain_ids:
        if chain_id not in model:
            continue
        for residue in model[chain_id]:
            if Polypeptide.is_aa(residue, standard=True):
                residues.append((chain_id, int(residue.id[1])))
    return residues


def compact_chain_positions(residues: list[tuple[str, int]]) -> str:
    if not residues:
        return ""
    tokens: list[str] = []
    run_chain, run_start = residues[0]
    run_end = run_start
    for chain_id, resseq in residues[1:]:
        if chain_id == run_chain and resseq == run_end + 1:
            run_end = resseq
            continue
        tokens.append(format_position_run(run_chain, run_start, run_end))
        run_chain, run_start, run_end = chain_id, resseq, resseq
    tokens.append(format_position_run(run_chain, run_start, run_end))
    return ",".join(tokens)


def format_position_run(chain_id: str, start: int, end: int) -> str:
    if start == end:
        return f"{chain_id}{start}"
    return f"{chain_id}{start}-{chain_id}{end}"


def rso_filters_pass(metrics: dict[str, float | None], filters: dict[str, Any]) -> tuple[bool, str]:
    checks = [
        ("plddt", filters.get("min_plddt"), True),
        ("iptm", filters.get("min_iptm"), True),
        ("pae", filters.get("max_pae"), False),
        ("i_pae", filters.get("max_i_pae"), False),
    ]
    failed = []
    for key, threshold, higher in checks:
        if not passes_threshold(metrics.get(key), threshold, higher):
            op = ">=" if higher else "<="
            failed.append(f"{key} {metrics.get(key)} not {op} {threshold}")
    return (not failed, "; ".join(failed))


def initialise_model(settings: dict[str, Any]) -> Any:
    add_colabdesign_to_path(settings)
    from colabdesign import clear_mem, mk_afdesign_model

    rso = require(settings, "rso")
    model_settings = rso.get("model", {})
    clear_mem()
    af_model = mk_afdesign_model(
        protocol=model_settings.get("protocol", "binder"),
        use_multimer=model_settings.get("use_multimer", True),
        data_dir=settings.get("tools", {}).get("alphafold_params_dir"),
        recycle_mode=model_settings.get("recycle_mode", "sample"),
    )
    af_model.prep_inputs(
        pdb_filename=require(settings, "input_pdb"),
        chain=rso.get("target_chains", settings.get("target_chains", "A")),
        hotspot=rso.get("hotspot_residues", settings.get("hotspot_residues", "")),
        binder_chain=rso.get("binder_chain", settings.get("binder_chain", "B")),
        fix_pos=fixed_positions_from_design_positions(settings, rso),
    )

    if "contacts" in rso:
        contacts = rso["contacts"]
        if "intra" in contacts:
            af_model.opt["con"].update(contacts["intra"])
        if "inter" in contacts:
            af_model.opt["i_con"].update(contacts["inter"])

    af_model.opt["weights"].update(rso.get("weights", {}))
    if rso.get("disable_dgram_cce", True):
        af_model.opt["weights"]["dgram_cce"] = 0
    return af_model


def collect_metrics(af_model: Any, binder_length: int | None) -> dict[str, float | None]:
    log = best_log(af_model)
    return {
        "loss": normalise_metric(metric_from_log(log, "loss")),
        "plddt": normalise_metric(best_plddt(af_model, binder_length)),
        "iptm": normalise_metric(metric_from_log(log, "i_ptm", "iptm", "i_pTM")),
        "pae": normalise_metric(metric_from_log(log, "pae", "pAE")),
        "i_pae": normalise_metric(metric_from_log(log, "i_pae", "i_pAE")),
    }


def run_one(settings: dict[str, Any], seed: int, accepted_dir: Path, rejected_dir: Path, traj_dir: Path) -> dict[str, Any]:
    start = time.time()
    rso = settings["rso"]
    binder_length = rso.get("binder_length")
    early = rso.get("early_stop", {})
    filters = rso.get("save_filters", {})
    af_model = initialise_model(settings)

    np.random.seed(seed)
    if hasattr(af_model, "set_seed"):
        af_model.set_seed(seed)
    name_prefix = f"rso_{seed}"

    initial_iters = int(early.get("initial_logits_iters", 75))
    if initial_iters > 0:
        af_model.design_logits(iters=initial_iters, e_soft=early.get("initial_e_soft", 0.9), ramp_recycles=False, save_best=True)
        initial_plddt = best_plddt(af_model, binder_length)
        if below_threshold(initial_plddt, early.get("min_initial_plddt", 0.0)):
            metrics = collect_metrics(af_model, binder_length)
            return {
                "design_name": name_prefix,
                "seed": seed,
                "status": "early_stopped",
                **metrics,
                "output_pdb": "",
                "reason": f"initial plddt {initial_plddt:.4f} < {early.get('min_initial_plddt')}",
                "elapsed_seconds": round(time.time() - start, 2),
            }

    af_model.clear_best()
    af_model.design_logits(
        iters=int(rso.get("logit_iters", 300)),
        e_soft=float(rso.get("logit_e_soft", 1.0)),
        ramp_recycles=bool(rso.get("ramp_recycles", False)),
        save_best=True,
    )
    metrics = collect_metrics(af_model, binder_length)
    if below_threshold(metrics["plddt"], early.get("min_logit_plddt", 0.0)):
        return {
            "design_name": name_prefix,
            "seed": seed,
            "status": "early_stopped",
            **metrics,
            "output_pdb": "",
            "reason": f"logit plddt {metrics['plddt']:.4f} < {early.get('min_logit_plddt')}",
            "elapsed_seconds": round(time.time() - start, 2),
        }

    soft_iters = int(rso.get("soft_iters", 0))
    if soft_iters > 0:
        af_model.design_soft(
            iters=soft_iters,
            e_temp=float(rso.get("soft_e_temp", 1e-2)),
            ramp_recycles=bool(rso.get("ramp_recycles", False)),
            save_best=True,
        )
        metrics = collect_metrics(af_model, binder_length)
        if below_threshold(metrics["plddt"], early.get("min_soft_plddt", 0.0)):
            return {
                "design_name": name_prefix,
                "seed": seed,
                "status": "early_stopped",
                **metrics,
                "output_pdb": "",
                "reason": f"soft plddt {metrics['plddt']:.4f} < {early.get('min_soft_plddt')}",
                "elapsed_seconds": round(time.time() - start, 2),
            }

    hard_iters = int(rso.get("hard_iters", 0))
    if hard_iters > 0:
        af_model.clear_best()
        af_model.design_hard(
            iters=hard_iters,
            temp=float(rso.get("hard_temp", 1e-2)),
            dropout=bool(rso.get("dropout", False)),
            ramp_recycles=bool(rso.get("ramp_recycles", False)),
            save_best=True,
        )

    semigreedy_iters = int(rso.get("semigreedy_iters", 0))
    if semigreedy_iters > 0:
        af_model.design_pssm_semigreedy(
            soft_iters=0,
            hard_iters=semigreedy_iters,
            tries=int(rso.get("semigreedy_tries", 10)),
            dropout=bool(rso.get("dropout", False)),
            num_models=int(rso.get("semigreedy_num_models", 1)),
            ramp_models=bool(rso.get("semigreedy_ramp_models", False)),
            save_best=True,
        )

    metrics = collect_metrics(af_model, binder_length)
    passed, reason = rso_filters_pass(metrics, filters)
    status = "accepted" if passed else "filtered"
    target_dir = accepted_dir if passed else rejected_dir
    design_name = f"rso_{seed}_plddt{metrics['plddt']}_iptm{metrics['iptm']}"
    pdb_path = target_dir / f"{design_name}.pdb"
    af_model.save_pdb(str(pdb_path), get_best=True)

    if rso.get("save_trajectory", False):
        import pickle

        for key in ("traj", "log"):
            try:
                with (traj_dir / f"{design_name}_{key}.pkl").open("wb") as handle:
                    pickle.dump(af_model._tmp[key], handle)
            except Exception as exc:
                print(f"Warning: failed to save {key} for {design_name}: {exc}")

    return {
        "design_name": design_name,
        "seed": seed,
        "status": status,
        **metrics,
        "output_pdb": str(pdb_path),
        "reason": reason,
        "elapsed_seconds": round(time.time() - start, 2),
    }


def completed_accepted(metrics_csv: Path) -> int:
    return sum(1 for row in read_csv_rows(metrics_csv) if row.get("status") == "accepted")


def used_seeds(manifest_path: Path) -> set[int]:
    seeds: set[int] = set()
    for row in read_jsonl(manifest_path):
        try:
            seeds.add(int(row["seed"]))
        except Exception:
            continue
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description="Run relaxed scaffold sequence optimisation with colabdesign_modified.")
    parser.add_argument("--settings", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int, default=None, help="Maximum accepted RSO structures to add in this run.")
    args = parser.parse_args()

    ensure_numpy()
    settings = load_settings(args.settings, args.output_dir)
    rso = require(settings, "rso")
    out = stage_dir(settings, "rso")
    accepted_dir = out / "accepted_pdbs"
    rejected_dir = out / "filtered_pdbs"
    traj_dir = out / "trajectories"
    for path in (accepted_dir, rejected_dir, traj_dir):
        path.mkdir(parents=True, exist_ok=True)

    metrics_csv = out / "rso_metrics.csv"
    manifest_path = out / "manifest.jsonl"
    target = int(rso.get("num_structures", settings.get("num_structures", 0)))
    if target <= 0:
        raise ValueError("rso.num_structures must be greater than zero")

    already = completed_accepted(metrics_csv)
    to_add = max(0, target - already)
    if args.limit is not None:
        to_add = min(to_add, max(0, int(args.limit)))
    print(f"RSO target={target}, accepted_existing={already}, accepted_to_add={to_add}")
    if to_add == 0:
        return 0

    rng = np.random.default_rng(int(rso.get("seed", 0)) or None)
    seen = used_seeds(manifest_path)
    attempts = 0
    accepted_this_run = 0
    max_attempts = int(rso.get("max_attempts", max(100, to_add * 20)))
    while accepted_this_run < to_add and attempts < max_attempts:
        seed = int(rng.integers(1, 2_147_483_647))
        if seed in seen:
            continue
        seen.add(seed)
        attempts += 1
        try:
            row = run_one(settings, seed, accepted_dir, rejected_dir, traj_dir)
        except Exception as exc:
            row = {
                "design_name": f"rso_{seed}",
                "seed": seed,
                "status": "failed",
                "loss": None,
                "plddt": None,
                "iptm": None,
                "pae": None,
                "i_pae": None,
                "output_pdb": "",
                "reason": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": 0,
            }
            print(f"RSO seed {seed} failed: {exc}")
        append_csv(metrics_csv, row, METRIC_FIELDS)
        append_jsonl(manifest_path, row)
        if row["status"] == "accepted":
            accepted_this_run += 1
        print(f"{row['design_name']}: {row['status']} ({accepted_this_run}/{to_add} accepted this run)")

    if accepted_this_run < to_add:
        print(f"Stopped after {attempts} attempts with {accepted_this_run}/{to_add} accepted.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
