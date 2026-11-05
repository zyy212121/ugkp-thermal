#!/usr/bin/env python3
"""Native CPU/GPU interface parity with actual SSPRK3 stages and history.

Prepare the versioned mixture and moving-wall fixtures without an existing run.
The moving case uses the supported constantTransport wall, constant viscosity,
and four gas microsteps per coupling window. Solver tolerances are unchanged.
Run from a sourced OpenFOAM 10 environment with separately built CHMT binaries.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("layer1_evidence", ROOT/"validation/gpu_coupled_pipeline/layer1.py")
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)
matrix = evidence.matrix
CASES = {"mixture": "chmt-mixture", "moving-microsteps": "chmt-moving-short"}


def input_hashes(case):
    return {str(path.relative_to(case)): evidence.digest(path)
            for name in ("0", "constant", "system")
            for path in sorted((case/name).rglob("*")) if path.is_file()}


def prepare(work):
    work.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((matrix.HERE/"fixtures.json").read_text())
    provenance = {}
    for name, fixture in CASES.items():
        baseline, candidate = (work/(name+"-"+kind) for kind in ("baseline", "candidate"))
        matrix.prepare(matrix.HERE/"fixtures"/fixture, baseline, metadata[fixture])
        schemes = baseline/"system/fvSchemes"
        text, count = re.subn(r"(ddtSchemes\s*\{\s*default\s+)Euler;", r"\1SSPRK3;", schemes.read_text())
        assert count == 1, (schemes, "expected fixture Euler integrator")
        schemes.write_text(text)
        overrides = {"ddtSchemes/default": "SSPRK3"}
        if name == "moving-microsteps":
            props = baseline/"constant/chmtProperties"
            interval = float(re.search(r"couplingInterval\s+([^;]+)", props.read_text())[1])
            matrix.entry(props, "gasMaxDt", interval/4)
            text, count = re.subn(r"\bmodel\s+reactingSst;", "model constantTransport;", props.read_text())
            assert count == 1, props
            text, count = re.subn(r"\benableSst\s+true;", "enableSst false;", text)
            assert count == 1, props
            props.write_text(text)
            # The analytic wall model requires constant molecular viscosity.
            (baseline/"constant/gasTransportProperties").unlink()
            overrides.update(gasMaxDt="couplingInterval/4", wallModel="constantTransport",
                             enableSst=False, molecularViscosity="constant")
        matrix.copy_input(baseline, candidate)
        hashes = input_hashes(baseline)
        assert hashes == input_hashes(candidate), name
        provenance[name] = dict(source_fixture=str(Path("validation/molecular_transport/fixtures")/fixture),
                                overrides=overrides, input_sha256=hashes)
    (work/"input-provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")


def execute(work, baseline, candidate):
    binaries = {"baseline": baseline.resolve(), "candidate": candidate.resolve()}
    provenance = json.loads((work/"input-provenance.json").read_text())
    report = dict(binaries={kind: dict(path=str(binary), sha256=evidence.digest(binary))
                            for kind, binary in binaries.items()}, cases={})
    for name in CASES:
        runs = {}
        for kind, binary in binaries.items():
            case = work/(name+"-"+kind)
            assert input_hashes(case) == provenance[name]["input_sha256"], (case, "input changed since preparation")
            assert not (case/"chmtOutput").exists(), (case, "use a fresh run directory")
            elapsed = matrix.run([binary, "-case", case], case/"run.log")
            output = case/"chmtOutput"
            exchanges = sorted((path for path in output.glob("exchange-*.csv") if evidence.rows(path)),
                               key=lambda path: int(path.stem.split("-")[-1]))
            counts = [len({row["step"] for row in evidence.rows(path)}) for path in exchanges]
            assert counts and min(counts) >= 4, (case, counts)
            stage_snapshots = 0
            if name == "moving-microsteps":
                # For the four equal microsteps, the last SSPRK3 RHS is at the
                # midpoint of the last microstep, not at its accepted endpoint.
                assert all(count == 4 for count in counts), (case, counts)
                props = (case/"constant/chmtProperties").read_text()
                dt = float(re.search(r"\bgasMaxDt\s+([^;]+)", props)[1])
                for path in output.glob("wall-layer-*.csv"):
                    wall = evidence.rows(path)
                    if not wall or float(wall[0]["accepted_time"]) == 0:
                        continue
                    stage_snapshots += 1
                    for row in wall:
                        accepted, stage = float(row["accepted_time"]), float(row["stage_time"])
                        expected = accepted-dt/2
                        tolerance = 32*math.ulp(max(abs(accepted), abs(dt)))
                        assert abs(stage-expected) <= tolerance, (path, stage, expected)
                assert stage_snapshots == len(exchanges), (case, stage_snapshots)
            runs[kind] = dict(elapsed_seconds=elapsed, coupling_windows=len(exchanges),
                              microsteps_per_window=counts, checked_rk_stage_snapshots=stage_snapshots,
                              audit=evidence.audit_chmt(case, name == "mixture"))
        parity = evidence.compare_chmt(work/(name+"-baseline"), work/(name+"-candidate"))
        report["cases"][name] = dict(runs=runs, parity=parity)
        (work/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps(dict(case=name, parity=parity, runs=runs)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run-only", action="store_true")
    args = parser.parse_args()
    if not args.prepare_only and not (args.baseline and args.candidate):
        parser.error("--baseline and --candidate are required to execute")
    args.work = args.work.resolve()
    if not args.run_only:
        prepare(args.work)
    if not args.prepare_only:
        execute(args.work, args.baseline, args.candidate)


if __name__ == "__main__":
    main()
