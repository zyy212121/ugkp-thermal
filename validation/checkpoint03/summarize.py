#!/usr/bin/env python3
"""Summarize an already executed checkpoint. Never starts or retries a test."""
from pathlib import Path
import argparse, json, statistics
def main():
    p=argparse.ArgumentParser();p.add_argument("batch",type=Path);a=p.parse_args();out=a.batch.resolve()
    status=json.loads((out/"batch-status.json").read_text())
    steps=status.get("steps",[])
    lines=["CHECKPOINT03 EXECUTION REVIEW",status["status"],
           "This report is evidence for review, not automatic acceptance or a production speedup claim.",
           "All CP3 operator/native evidence is paired new execution; historical CP2 warnings are separate.",
           ""]
    for result in ("RUN_PASS","FAIL","SKIP_DEPENDENCY"):
        selected=[x for x in steps if x.get("result")==result]
        lines.append(f"{result}: {len(selected)}")
        if result!="RUN_PASS":
            lines.extend("  "+x["id"]+" "+str(x.get("error",x.get("reason",""))) for x in selected)
    lines+=["","CORRECTNESS"]
    for row in steps:
        if row["kind"]=="correctness" and row.get("result")=="RUN_PASS":
            lines.append(row["id"]+": "+str(row["payload"].get("checks","unreported"))+" observations/case records")
            if row.get("payload",{}).get("counts"):
                lines.append("  Parameter-call counts are correctness-only instrumentation; performance uses a separate binary.")
    lines+=["","NATIVE"]
    for row in steps:
        if row["kind"]=="native":
            child=row.get("native",{})
            lines.append(row["id"]+": "+row.get("result","NOT_RUN"))
            for run in child.get("runs",[]):
                lines.append("  seed="+str(run.get("seed"))+" exit="+str(run.get("exit"))+
                             " elapsed="+str(run.get("seconds"))+" (not a benchmark)")
                lines.extend("  warning: "+x for x in run.get("warning_lines",[]))
    lines+=["","PAIRED PERFORMANCE: baseline seconds / candidate seconds"]
    performance={}
    for row in steps:
        if row["kind"]=="performance":
            performance[(row["precision"],row["family"],row["repeat"],row["variant"])]=row
    for bits in (64,32):
        for family in ("pressure","face","coupling","bath"):
            ratios=[]
            lines.append(f"fp{bits} {family} "+("CUDA kernel event timing" if family=="pressure" else "HOST ARITHMETIC ONLY"))
            for repeat in range(3):
                pair=[performance.get((bits,family,repeat,v)) for v in ("baseline","candidate")]
                if not all(row and row.get("result")=="RUN_PASS" for row in pair):
                    lines.append(f"  pair{repeat}: no complete successful pair");continue
                left,right=pair
                lt,rt=left["timing"],right["timing"]
                ratio=lt["median_seconds"]/rt["median_seconds"];ratios.append(ratio)
                lines.append(f"  pair{repeat} order="+("AB" if repeat%2==0 else "BA")+
                    f" baseline={lt['median_seconds']:.9g} [{lt['min_seconds']:.9g},{lt['max_seconds']:.9g}]"+
                    f" candidate={rt['median_seconds']:.9g} [{rt['min_seconds']:.9g},{rt['max_seconds']:.9g}]"+
                    f" observed_ratio={ratio:.6g}")
                for row in pair:
                    payload=row["payload"]
                    details={key:payload.get(key) for key in
                        ("iterations","worksize","setup/reset_excluded","actual_shared_bytes_per_block",
                         "parameter_struct_bytes","cache_bytes","global_cache_bytes") if key in payload}
                    lines.append("    "+row["variant"]+" workload/storage "+json.dumps(details,sort_keys=True))
            if ratios:lines.append("  Median observed paired ratio="+format(statistics.median(ratios),".6g")+
                                  "; this number alone does not establish a speedup.")
    lines+=["","LIMITS",
        "Pressure device allocation/reset transfers are excluded from event timings.",
        "Face/coupling/bath timings are host arithmetic, not GPU throughput or full-solver acceleration.",
        "Coupling extra production scratch is 12B/particle FP32 and16B/particle FP64; fixture allocations are separate.",
        "Finite-bath public API does not expose actual nonlinear iteration counts. No per-iteration speed claim.",
        "Inspect all device-wide clocks/load/power/temperature traces before interpreting observed ratios.",
        "WSL readings may be unavailable; device memory telemetry is not attributable per-process peak memory.",
        "Short native cases do not prove production steady-state convergence.",
        "Historical CP2 CHT had8 DO budget warnings;4.909006e-7 exceeded requested1e-8. Not a new CP3 pass.",
        "Failures/skips are never passes. No repairs, retries, commit, push or merge are performed here."]
    (out/"REPORT.txt").write_text("\n".join(lines)+"\n")
if __name__=="__main__":main()
