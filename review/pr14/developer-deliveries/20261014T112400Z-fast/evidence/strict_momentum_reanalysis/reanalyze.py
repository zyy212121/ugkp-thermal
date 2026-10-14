#!/usr/bin/env python3
"""Offline analysis of immutable FAST2 GPU logs; never launches a solver.

Run from any directory: python3 reanalyze.py --root /path/to/repository
Only RESULT.json and INPUT_HASHES.json beside this script are written.
"""
import argparse
import collections
import decimal
import hashlib
import json
from pathlib import Path

decimal.getcontext().prec = 80
D = decimal.Decimal
ZERO = D(0)
CASES = {"SST-SSPRK2": 2, "SST-SSPRK3": 3, "dry-ablation": 1}
DELIVERY = Path("review/pr14/reviewer-deliveries/20261014T072053Z-fast")


def exact(value):
    return D.from_float(float(value))


def scalar(token):
    return float.fromhex(token)


def stock(state):
    return sum((exact(row["momentum"][0]) for row in state["gas"]), ZERO)


def tolerance(before, after):
    # Same binary64 literals and scale as the original external bulk-P audit.
    return exact(1e-14) + exact(1e-10) * max(abs(before), abs(after))


def analyze(root, name, stage_count, inputs):
    folder = root / DELIVERY / "artifacts/extra" / name
    log = root / DELIVERY / "logs" / ("ACTUAL-WEIGHTED-LEDGER-" + name + ".log")
    paths = [log, folder / "actual-weighted-final-fluxes.json",
             folder / "complete-independent-budget-audit.json"]
    paths += [folder / (f"checkpoint-{i}-S207-decoded.json") for i in range(3)]
    paths += [folder / f"checkpoint-{i}/state.chmt" for i in range(3)]
    paths += [root / DELIVERY / "logs" / ("ACTUAL-WEIGHTED-LEDGER-" + name + ".spec.json")]
    # The archived case folders contain deltas; inherited originals remain in
    # their earlier immutable delivery. Do not pretend missing files are present.
    paths += sorted(p for p in (root / DELIVERY / "inputs/native/extra" / name).rglob("*") if p.is_file())
    for path in paths:
        inputs[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    states = [json.loads((folder / f"checkpoint-{i}-S207-decoded.json").read_text())
              for i in range(3)]
    retained = json.loads((folder / "actual-weighted-final-fluxes.json").read_text())
    assert retained["alloutput_comparison"]["all_files_byte_equal"]
    original = json.loads((folder / "complete-independent-budget-audit.json").read_text())
    dt, faces, packets, gas = {}, collections.defaultdict(dict), collections.defaultdict(dict), collections.defaultdict(dict)
    micro_flux = collections.defaultdict(lambda: ZERO)
    macro_flux = collections.defaultdict(lambda: ZERO)
    rounded_boundary = collections.defaultdict(lambda: ZERO)
    rounded_exchange = collections.defaultdict(lambda: ZERO)
    float_boundary = float_exchange = 0.0
    checkpoints = {}
    # BoundaryKind enum: NoSlip=3, Interface=6. In these specific inputs only.
    # Persistent packet face IDs are NOT physical gas mesh face IDs.
    kind_counts = collections.Counter()
    for line in log.read_text().splitlines():
        a = line.split(",")
        if a[0] == "LEDGER_STAGE":
            key = tuple(map(int, a[1:4]))
            assert key not in dt
            dt[key] = scalar(a[4])
        elif a[0] == "LEDGER_FINAL_FLUX":
            key = tuple(map(int, a[1:4]))
            face, neighbour, kind = map(int, a[4:7])
            assert face not in faces[key]
            faces[key][face] = (neighbour, kind)
            if neighbour < 0:
                assert kind in (3, 6), "This scoped replay has no periodic/inlet/other boundary model."
                kind_counts[kind] += 1
                # Exact decimal multiplication of the logged binary64 operands.
                flux = exact(scalar(a[9])) * exact(dt[key])
                micro_flux[key[:2]] += flux
                macro_flux[key[0]] += flux
                if kind == 3:
                    # Reproduce the actual double multiply and accumulation order.
                    amount = scalar(a[9]) * dt[key]
                    float_boundary += amount
                    rounded_boundary[key[0]] += exact(amount)
        elif a[0] == "LEDGER_PAIRED":
            key = tuple(map(int, a[1:4]))
            identity = tuple(map(int, a[4:7]))
            assert identity not in packets[key]
            assert identity[2] in (0, 3), "Only GasSolid/PoreGas channels covered."
            amount = scalar(a[11])
            packets[key][identity] = amount
            float_exchange += amount
            rounded_exchange[key[0]] += exact(amount)
        elif a[0] == "F003" and a[4] == "gas" and a[6] == "momentum_x":
            key = int(a[1]), int(a[3])
            cell = int(a[5])
            assert cell not in gas[key]
            gas[key][cell] = scalar(a[7])
        elif a[0] == "F003" and a[4] == "accepted" and a[6] == "end" and int(a[3]) == 50:
            checkpoints[int(a[1])] = (float_boundary, float_exchange)
    expected_micro = {(w, m) for w in (1, 2) for m in range(1, 51)}
    assert set(gas) == expected_micro
    assert set(dt) == {(w, m, s) for w, m in expected_micro for s in range(stage_count)}
    assert set(dt) == set(faces) == set(packets)
    for key in dt:
        assert set(faces[key]) == set(range(36))
        assert sum(n < 0 for n, _ in faces[key].values()) == 24
        assert {f for f, (n, k) in faces[key].items() if n < 0 and k == 6} == {12, 13, 14, 15}
        assert len(packets[key]) == 8
    rows = []
    previous = stock(states[0])
    for key in sorted(gas):
        assert set(gas[key]) == set(range(8))
        current = sum((exact(v) for v in gas[key].values()), ZERO)
        residual = current - previous + micro_flux[key]
        rows.append({"window": key[0], "microstep": key[1],
                     "gas_Px_before": str(previous), "gas_Px_after": str(current),
                     "actual_allboundary_impulse_x": str(micro_flux[key]),
                     "physical_flux_residual_x": str(residual),
                     "original_threshold": str(tolerance(previous, current)),
                     "within_original_threshold": abs(residual) <= tolerance(previous, current)})
        previous = current
    windows = []
    for w in (1, 2):
        a, b = states[w - 1], states[w]
        before, after = stock(a), stock(b)
        boundary = exact(b["budget"]["boundaryMomentum"][0]) - exact(a["budget"]["boundaryMomentum"][0])
        exchange = exact(b["budget"]["exchangeMomentum"][0][0]) - exact(a["budget"]["exchangeMomentum"][0][0])
        replay_boundary, replay_exchange = checkpoints[w]
        assert replay_boundary.hex() == float(b["budget"]["boundaryMomentum"][0]).hex()
        assert replay_exchange.hex() == float(b["budget"]["exchangeMomentum"][0][0]).hex()
        observed = after - before + boundary - exchange
        assert observed == D(original["checks"][w - 1]["gas_momentum"][0]["residual"])
        physical = after - before + macro_flux[w]
        rounded_net = rounded_boundary[w] - rounded_exchange[w]
        windows.append({"window": w, "gas_Px_change": str(after - before),
            "actual_allboundary_impulse_x": str(macro_flux[w]),
            "physical_flux_residual_x": str(physical),
            "original_threshold": str(tolerance(before, after)),
            "physical_flux_within_original_threshold": abs(physical) <= tolerance(before, after),
            "retained_strict_budget_residual_x": str(observed),
            "retained_strict_budget_pass": original["checks"][w - 1]["gas_momentum"][0]["pass"],
            "native_boundary_budget_replay_hex": replay_boundary.hex(),
            "native_exchange_budget_replay_hex": replay_exchange.hex(),
            "both_cumulative_budget_replays_bit_exact": True,
            "exact_sum_of_rounded_boundary_minus_packet_amounts": str(rounded_net),
            "cumulative_addition_roundoff_contribution": str(boundary - exchange - rounded_net),
            "product_packet_representation_contribution": str(rounded_net - macro_flux[w]),
            "total_budget_representation_contribution": str(boundary - exchange - macro_flux[w])})
    return {"case": name, "classification": "OFFLINE_REANALYSIS_OF_EXISTING_GPU_EVIDENCE",
        "new_GPU_runs": 0, "scope": "CHMT gas bulk x momentum only; no whole-system momentum closure",
        "coverage": {"windows": 2, "microsteps": len(gas), "stages": len(dt),
            "cells_each_accepted_microstep": 8, "final_faces_each_stage": 36,
            "physical_boundary_faces_each_stage": 24, "interface_faces_each_stage": 4,
            "no_slip_faces_each_stage": 20, "paired_packets_each_stage": 8,
            "recorded_boundary_kind_counts": dict(kind_counts),
            "packet_kinds": ["GasSolid", "PoreGas"],
            "observer_original_output_bytes_equal": True},
        "windows_result": windows, "accepted_microstep_results": rows,
        "maximum_absolute_microstep_flux_residual_x": str(max(abs(D(q["physical_flux_residual_x"])) for q in rows)),
        "original_strict_failures_preserved": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    inputs = {}
    for rel in (DELIVERY / "SOURCE.json", DELIVERY / "SHA256SUMS",
                DELIVERY / "patches/D203-actual-stage-ledger.patch",
                Path("applications/CHMT/gpu/Backend.cu"),
                Path("applications/CHMT/core/Types.H"),
                Path("applications/CHMT/coupling/ExchangeLedger.H"),
                Path("applications/CHMT/coupling/IntervalAudit.H")):
        inputs[str(rel)] = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    result = {"decimal_precision": 80, "input_delivery": str(DELIVERY),
              "criteria_changed": False, "production_code_changed": False,
              "cases": [analyze(root, n, s, inputs) for n, s in CASES.items()]}
    out = Path(__file__).resolve().parent
    (out / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "INPUT_HASHES.json").write_text(json.dumps(inputs, indent=2, sort_keys=True) + "\n")
    for case in result["cases"]:
        print(case["case"], [(r["physical_flux_residual_x"], r["retained_strict_budget_residual_x"])
                             for r in case["windows_result"]])


if __name__ == "__main__":
    main()
