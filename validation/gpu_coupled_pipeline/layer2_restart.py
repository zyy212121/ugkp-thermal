#!/usr/bin/env python3
"""Resume a native material case at an accepted checkpoint and compare its endpoint."""
import argparse
import json
import os
from pathlib import Path
import layer1
import layer2_native


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--checkpoint", type=int, default=1)
    p.add_argument("--scenario", default="dry")
    p.add_argument("--inspector", type=Path, required=True)
    a = p.parse_args()
    a.reference = a.reference.resolve(); a.binary = a.binary.resolve(); a.work = a.work.resolve()
    checkpoint = a.reference / "chmtOutput" / f"checkpoint-{a.checkpoint}"
    assert (checkpoint / "state.chmt").is_file()
    a.work.mkdir(parents=True, exist_ok=False)
    case = a.work / "case"
    layer1.matrix.copy_input(a.reference, case)
    props = case / "constant/chmtProperties"
    props.write_text(props.read_text() + f'\nrestartDirectory "{checkpoint}";\n')
    layer1.matrix.ENV.update(CHMT_CUDA_STACK_BYTES="65536", OMP_NUM_THREADS="4")
    report = dict(status="RUNNING", binary=str(a.binary), binary_sha256=layer1.digest(a.binary),
                  reference=str(a.reference), checkpoint=str(checkpoint),
                  checkpoint_sha256=layer1.digest(checkpoint / "state.chmt"))
    def save():
        (a.work / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    try:
        report["elapsed_seconds"] = layer1.matrix.run([a.binary, "-case", case], case / "run.log")
        final = max(int(p.stem.split("-")[-1]) for p in (a.reference / "chmtOutput").glob("gas-*.csv"))
        assert final > a.checkpoint
        report["final_scaled_csv_errors"] = {}
        for kind in ("accepted", "gas", "material", "exchange"):
            name = f"{kind}-{final}.csv"
            report["final_scaled_csv_errors"][kind] = layer1.compare_csv(
                a.reference / "chmtOutput" / name, case / "chmtOutput" / name)
        def payload(directory):
            wire = layer1.matrix.terminal(directory).read_bytes()
            header = layer1.matrix.HEADER.unpack_from(wire)
            return wire[layer1.matrix.HEADER.size + header[-3]:]
        report["checkpoint_payload_bitwise_equal"] = payload(case) == payload(a.reference)
        assert report["checkpoint_payload_bitwise_equal"], "Restart changed endpoint checkpoint inventories/geometry"
        report["audit"] = layer2_native.audit(case, a.scenario,
            layer2_native.checkpoint_material(case, a.inspector.resolve()))
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAILED", error=repr(error)); save(); raise
    save(); print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
