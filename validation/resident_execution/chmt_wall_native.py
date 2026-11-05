#!/usr/bin/env python3
"""CPU-interface baseline versus GPU-interface CHMT with real wall-model stages.

Uses the existing moving-wall fixture, with four gas microsteps per coupling
window. Models are ordinary supported input choices, never solver branches.
Run in a sourced OpenFOAM 10 environment with separately built binaries.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("layer1_evidence", ROOT/"validation/gpu_coupled_pipeline/layer1.py")
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)
matrix = evidence.matrix


def prepare(work):
    work.mkdir(parents=True, exist_ok=False)
    meta = json.loads((matrix.HERE/"fixtures.json").read_text())["chmt-moving-short"]
    for model in ("constantTransport", "reactingSst"):
        for kind in ("baseline", "candidate"):
            case = work/(model+"-"+kind)
            matrix.prepare(matrix.HERE/"fixtures/chmt-moving-short", case, meta)
            props = case/"constant/chmtProperties"
            interval = float(re.search(r"couplingInterval\s+([^;]+)", props.read_text())[1])
            matrix.entry(props, "gasMaxDt", interval/4)
            if model == "constantTransport":
                props.write_text(props.read_text().replace("model reactingSst;", "model constantTransport;").replace("enableSst true;", "enableSst false;"))
                # The analytic constant-transport wall model requires constant viscosity.
                (case/"constant/gasTransportProperties").unlink()
    (work/"input-provenance.json").write_text(json.dumps(dict(
        source_fixture="validation/molecular_transport/fixtures/chmt-moving-short",
        overrides=dict(gas_max_dt="couplingInterval/4", constant_transport="constantTransport with enableSst=false and constant molecular viscosity"),
        physics="Existing two-species surface conversion and coupled recession; original tolerances retained"), indent=2)+"\n")


def execute(work, baseline, candidate, models):
    report = dict(binaries={kind:dict(path=str(binary), sha256=evidence.digest(binary))
        for kind,binary in (("baseline", baseline), ("candidate",candidate))}, cases={})
    for model in models:
        runs = {}
        for kind,binary in (("baseline",baseline), ("candidate",candidate)):
            case = work/(model+"-"+kind)
            elapsed = matrix.run([binary, "-case",case],case/"run.log")
            output = case/"chmtOutput"
            exchanges = [p for p in output.glob("exchange-*.csv") if evidence.rows(p)]
            micro_counts = [len({r["step"] for r in evidence.rows(p)}) for p in exchanges]
            assert exchanges and min(micro_counts)>=4, (case, micro_counts)
            wall_outputs = list(output.glob("wall-layer-*.csv"))
            assert wall_outputs and any(evidence.rows(p) for p in wall_outputs), case
            runs[kind] = dict(elapsed_seconds=elapsed, coupling_windows=len(exchanges),
                microsteps_per_window=micro_counts, wall_output_snapshots=len(wall_outputs),
                audit=evidence.audit_chmt(case,False))
        parity = evidence.compare_chmt(work/(model+"-baseline"),work/(model+"-candidate"))
        report["cases"][model] = dict(runs=runs, parity=parity)
        (work/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        print(json.dumps(dict(model=model, parity=parity, runs=runs)),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work",required=True,type=Path)
    ap.add_argument("--baseline",type=Path)
    ap.add_argument("--candidate",type=Path)
    ap.add_argument("--models",nargs="+",choices=("constantTransport","reactingSst"),default=["constantTransport","reactingSst"])
    ap.add_argument("--prepare-only",action="store_true")
    ap.add_argument("--run-only",action="store_true")
    args=ap.parse_args()
    args.work=args.work.resolve()
    if not args.run_only:prepare(args.work)
    if not args.prepare_only:
        assert args.baseline and args.candidate, "Both native binaries are required"
        execute(args.work,args.baseline,args.candidate,args.models)

if __name__=="__main__":main()
