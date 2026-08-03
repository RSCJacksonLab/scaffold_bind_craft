#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scaffold_bind_craft.config import load_settings, stage_dir
from scaffold_bind_craft.io_utils import read_csv_rows


def hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError(f"Hamming distance requires equal-length sequences, got {len(left)} and {len(right)}")
    return sum(a != b for a, b in zip(left, right))


def select_diverse_sequences(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count < 1:
        raise ValueError("count must be at least 1")
    unique: dict[str, dict[str, str]] = {}
    for row in rows:
        sequence = row.get("sequence", "").strip().upper()
        if sequence:
            unique.setdefault(sequence, row)
    candidates = sorted(unique)
    if not candidates:
        raise ValueError("No non-empty accepted sequences were found")
    lengths = {len(sequence) for sequence in candidates}
    if len(lengths) != 1:
        raise ValueError(f"Accepted sequences have different lengths: {sorted(lengths)}")
    if count > len(candidates):
        raise ValueError(f"Requested {count} sequences, but only {len(candidates)} unique accepted sequences are available")

    first = max(candidates, key=lambda seq: (sum(hamming_distance(seq, other) for other in candidates), seq))
    selected = [first]
    remaining = set(candidates) - {first}
    while len(selected) < count:
        next_sequence = max(
            remaining,
            key=lambda seq: (
                min(hamming_distance(seq, chosen) for chosen in selected),
                sum(hamming_distance(seq, chosen) for chosen in selected),
                seq,
            ),
        )
        selected.append(next_sequence)
        remaining.remove(next_sequence)

    result = []
    for rank, sequence in enumerate(selected, start=1):
        source = unique[sequence]
        min_distance = "" if rank == 1 else str(min(hamming_distance(sequence, prior) for prior in selected[:rank - 1]))
        result.append({
            "selection_rank": str(rank),
            "design_name": source.get("design_name", ""),
            "sequence": sequence,
            "min_distance_to_previous": min_distance,
            "selection_metric": "",
            "selection_metric_value": "",
            "relaxed_pdb": source.get("relaxed_pdb", ""),
        })
    return result


def select_by_metric(
    rows: list[dict[str, str]],
    count: int,
    metric: str,
    lower_is_better: bool = False,
) -> list[dict[str, str]]:
    """Select unique accepted sequences by a numeric screening metric."""
    if count < 1:
        raise ValueError("count must be at least 1")

    ranked: list[tuple[float, str, str, dict[str, str]]] = []
    for row in rows:
        sequence = row.get("sequence", "").strip().upper()
        if not sequence:
            continue
        raw_value = row.get(metric, "").strip()
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        ranked.append((value, row.get("design_name", ""), sequence, row))

    if not ranked:
        raise ValueError(f"No accepted sequences have a numeric value for metric {metric!r}")

    # Sort best-to-worst, then use design name and sequence for deterministic ties.
    ranked.sort(key=lambda item: ((item[0] if lower_is_better else -item[0]), item[1], item[2]))
    unique: list[tuple[float, str, dict[str, str]]] = []
    seen_sequences: set[str] = set()
    for value, _design_name, sequence, row in ranked:
        if sequence in seen_sequences:
            continue
        seen_sequences.add(sequence)
        unique.append((value, sequence, row))

    if count > len(unique):
        raise ValueError(
            f"Requested {count} sequences, but only {len(unique)} unique accepted sequences "
            f"have a numeric {metric!r} value"
        )

    result: list[dict[str, str]] = []
    for rank, (value, sequence, source) in enumerate(unique[:count], start=1):
        result.append({
            "selection_rank": str(rank),
            "design_name": source.get("design_name", ""),
            "sequence": sequence,
            "min_distance_to_previous": "",
            "selection_metric": metric,
            "selection_metric_value": str(value),
            "relaxed_pdb": source.get("relaxed_pdb", ""),
        })
    return result


def write_annotated_screening_metrics(
    input_csv: Path,
    selected: list[dict[str, str]],
    output_dir: Path,
) -> Path:
    """Copy the screening table and mark the selected accepted designs."""
    output_path = output_dir / "screening_metrics.csv"
    selected_designs = {
        (row["design_name"], row["sequence"].strip().upper())
        for row in selected
    }
    with input_csv.open(newline="") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or [])
        if "status" not in fields:
            raise ValueError(f"Screening metrics CSV has no status column: {input_csv}")
        rows = list(reader)

    for row in rows:
        identity = (row.get("design_name", ""), row.get("sequence", "").strip().upper())
        if row.get("status", "").lower() == "accepted" and identity in selected_designs:
            row["status"] = "selected"

    with output_path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_outputs(
    selected: list[dict[str, str]],
    input_csv: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "selected_sequences.csv"
    fasta_path = output_dir / "selected_sequences.fasta"
    fields = [
        "selection_rank",
        "design_name",
        "sequence",
        "min_distance_to_previous",
        "selection_metric",
        "selection_metric_value",
        "relaxed_pdb",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    with fasta_path.open("w") as handle:
        for row in selected:
            name = row["design_name"] or f"selected_{row['selection_rank']}"
            handle.write(f">{name} selection_rank={row['selection_rank']}\n{row['sequence']}\n")
    screening_csv_path = write_annotated_screening_metrics(input_csv, selected, output_dir)
    return csv_path, fasta_path, screening_csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Greedily select diverse accepted binder sequences for wet-lab testing.")
    parser.add_argument("--settings", required=True)
    parser.add_argument("-n", "--num-sequences", required=True, type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--input-csv", help="Override the OpenMM screening metrics CSV.")
    parser.add_argument("--selection-dir", help="Override the selection output directory.")
    parser.add_argument(
        "--metric",
        help="Select the best accepted sequences by this numeric screening-metrics column instead of diversity.",
    )
    parser.add_argument(
        "--metric-lower-is-better",
        action="store_true",
        help="Rank smaller metric values first (the default ranks larger values first).",
    )
    args = parser.parse_args()

    settings = load_settings(args.settings, args.output_dir)
    input_csv = Path(args.input_csv).expanduser().resolve() if args.input_csv else stage_dir(settings, "openmm") / "screening_metrics.csv"
    if not input_csv.exists():
        raise FileNotFoundError(f"Screening metrics CSV not found: {input_csv}")
    accepted = [row for row in read_csv_rows(input_csv) if row.get("status", "").lower() == "accepted"]
    if args.metric:
        selected = select_by_metric(
            accepted,
            args.num_sequences,
            args.metric,
            lower_is_better=args.metric_lower_is_better,
        )
    else:
        if args.metric_lower_is_better:
            parser.error("--metric-lower-is-better requires --metric")
        selected = select_diverse_sequences(accepted, args.num_sequences)
    selection_dir = Path(args.selection_dir).expanduser().resolve() if args.selection_dir else Path(settings["output_dir"]) / "05_sequence_selection"
    csv_path, fasta_path, screening_csv_path = write_outputs(selected, input_csv, selection_dir)
    unique_count = len({row.get("sequence", "").strip().upper() for row in accepted if row.get("sequence", "").strip()})
    print(f"Selected {len(selected)} of {len(accepted)} accepted rows ({unique_count} unique sequences)")
    print(f"CSV: {csv_path}")
    print(f"FASTA: {fasta_path}")
    print(f"Annotated screening metrics: {screening_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
