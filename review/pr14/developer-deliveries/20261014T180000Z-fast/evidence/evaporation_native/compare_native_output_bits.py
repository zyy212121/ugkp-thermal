#!/usr/bin/env python3
"""Compare actual native CPU snapshots; no solver or production state mutation."""
from pathlib import Path
import hashlib
import json
import struct

HERE = Path(__file__).resolve().parent


def load(name):
    path = HERE / ("boundary_" + name + "_state.json.snapshots.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len({row["label"] for row in rows}) == len(rows)
    return path, {row["label"]: row for row in rows}


def exact(value):
    if isinstance(value, float):
        return ("binary64", struct.pack(">d", value).hex())
    if isinstance(value, dict):
        return tuple((key, exact(item)) for key, item in sorted(value.items()) if key != "label")
    if isinstance(value, list):
        return tuple(map(exact, value))
    return (type(value).__name__, value)


def main():
    original_path, original = load("original")
    candidate_path, candidate = load("signed_open_edge")
    shared = []
    promoted = []
    for label, value in original.items():
        same = exact(value) == exact(candidate[label])
        assert same, label
        shared.append({"label": label, "all_serialized_fields_exact_binary64_match": same})
    for mode in ("checkpoint_aux", "checkpoint_bulk"):
        for halves in (0, 1, 10, 30):
            old = f"{mode}-complete-half{halves}"
            new = f"{mode}-advance-half{halves}"
            same = exact(original[old]) == exact(candidate[new])
            assert same, (old, new)
            promoted.append({"original_terminal": old, "candidate_ordinary": new,
                             "all_serialized_fields_exact_binary64_match": same})
    inputs = [Path(__file__).resolve(), original_path, candidate_path,
              HERE / "checkpoint_film_boundary_probe.cpp",
              HERE / "checkpoint_film_packet_values.inc",
              HERE / "checkpoint_json_type_names.inc",
              HERE / "signed_open_edge_execution_receipt.json",
              HERE / "checkpoint_film_probe_execution_receipt.json",
              HERE / "evaporation_packet_sequence_receipt.json"]
    report = {
        "scope": "Independent read-only comparison of native operator snapshots. No full GPU owner replay.",
        "inputs_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
        "shared_successes": shared,
        "new_ordinary_equals_old_terminal": promoted,
        "shared_success_count": len(shared),
        "new_ordinary_equals_old_terminal_count": len(promoted),
        "float_comparison": "IEEE-754 binary64 bytes; integer/string/bool fields exact; labels excluded, checksum and every serialized output field included.",
    }
    (HERE / "independent_native_output_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(shared)} unchanged successes; {len(promoted)} new ordinary outputs equal original terminal at same dt")


if __name__ == "__main__":
    main()
