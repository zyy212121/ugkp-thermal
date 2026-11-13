#!/usr/bin/env python3
"""Compare native CHMT accepted histories after resident buffering and growth."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("history_evidence",ROOT/"validation/gpu_coupled_pipeline/layer1.py")
evidence=importlib.util.module_from_spec(spec);spec.loader.exec_module(evidence)
matrix=evidence.matrix

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--baseline",required=True,type=Path)
    parser.add_argument("--candidate",required=True,type=Path)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    meta=json.loads((matrix.HERE/"fixtures.json").read_text())["chmt-moving-short"]
    runs={}
    for kind,binary in (("baseline",args.baseline),("candidate",args.candidate)):
        case=out/kind;matrix.prepare(matrix.HERE/"fixtures/chmt-moving-short",case,meta)
        props=case/"constant/chmtProperties"
        props.write_text(props.read_text().replace("model reactingSst;","model equilibriumOde;"))
        interval=float(re.search(r"couplingInterval\s+([^;]+)",(case/"constant/scheme").read_text())[1])
        matrix.entry(case/"constant/scheme","maxDeltaT",interval/7)
        matrix.run([binary.resolve(),"-case",case],case/"run.log")
        exchanges=[p for p in (case/"chmtOutput").glob("exchange-*.csv") if evidence.rows(p)]
        micro=[len({row["step"] for row in evidence.rows(p)}) for p in exchanges]
        assert micro and min(micro)>=7,(case,micro)
        wall=list((case/"chmtOutput").glob("wall-layer-*.csv"))
        assert wall and any(evidence.rows(p) for p in wall),case
        runs[kind]={"binary_sha256":evidence.digest(binary),"microsteps_per_window":micro,
            "wall_output_snapshots":len(wall),"audit":evidence.audit_chmt(case,False)}
    parity=evidence.compare_chmt(out/"baseline",out/"candidate")
    assert parity["checkpoint_payload_bitwise_equal"],parity
    assert all(error<=2e-12 for error in parity["maximum_scaled_csv_errors"].values()),parity
    result={"runs":runs,"parity":parity}
    (out/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2,allow_nan=False))
if __name__=="__main__":main()
