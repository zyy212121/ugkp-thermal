#!/usr/bin/env python3
"""Read only the new batch evidence; never execute solver code."""
from pathlib import Path
import argparse, json, re, statistics
def summarize(batch):
    batch=Path(batch); status=json.loads((batch/"batch-status.json").read_text())
    records=[]
    for folder in sorted((batch/"results").glob("step-*")):
        file=folder/"execution.json"
        if not file.exists():continue
        step=json.loads(file.read_text())
        if not step["id"].startswith("performance-radiation-"):continue
        if step.get("result") not in ("RUN_PASS","REUSED_PASS"):continue
        if step.get("result")=="RUN_PASS" and step.get("exit")!=0:continue
        for line in (folder/"stdout.log").read_text(errors="replace").splitlines():
            if not line.startswith("PASS precision="):continue
            fields=dict(re.findall(r"([A-Za-z_]+)=([^\s]+)",line))
            record={"step":step["id"],"process_wall_seconds":step.get("wall_seconds",step.get("original_wall_seconds")),
                    "execution_result":step["result"],"source_batch":step.get("original_batch",str(batch))}
            for key,value in fields.items():
                try: record[key]=float(value)
                except ValueError: record[key]=value
            records.append(record)
    groups={}
    for row in records:
        key=(int(row["precision"]),int(row["concurrency"]),int(row["cells"]),int(row["angles"]),row["topology"])
        groups.setdefault(key,[]).append(row)
    result={"batch_status":status["status"],"performance_groups":[],"paired_ratios":[],
        "interpretation":["All calls use solve(false): run0 is the first cold solve and run>0 repeats cold-start solves in an already constructed process; this is not a warmStart=true benchmark.",
            "Only identical cells/angles/topology/concurrency/requested tolerance may be paired.",
            "Allocation delta is constructor cudaMemGetInfo, not peak VRAM.",
            "GPU.csv is observed device-wide memory, not per-process memory.",
            "No speedup claim is implied by this descriptive summary.",
            "Outer FP32 source_metric=0 can reflect unchanged stored floats, not a measured physical residual1e-12; inspect DIAG_LINEAR publishedResidual/publishedTarget, sticky warnings and closure.",
            "published_delta compares repeats/scheduling against the first solve within one precision, not FP32-versus-FP64 fields."]}
    comparison_file=batch/"fixtures/performance-comparability.json"
    comparison=json.loads(comparison_file.read_text()) if comparison_file.exists() else {"ratio_allowed":False,"reason":"Cross-batch conditions were not recorded."}
    result["cross_batch_comparability"]=comparison
    stats={}
    for key,rows in sorted(groups.items()):
        repeated_cold=[row["seconds"] for row in rows if row["run"]>0]
        cold=[row["seconds"] for row in rows if row["run"]==0]
        row={"precision":key[0],"concurrency":key[1],"cells":key[2],"angles":key[3],"topology":key[4],
            "requested_tolerances":sorted(set(r["requested_tolerance"] for r in rows)),
            "execution_origins":sorted(set(r["execution_result"] for r in rows)),"source_batches":sorted(set(r["source_batch"] for r in rows)),
            "repeated_cold_n":len(repeated_cold),"first_cold_n":len(cold),"repeated_cold_median_seconds":statistics.median(repeated_cold) if repeated_cold else None,
            "repeated_cold_range_seconds":[min(repeated_cold),max(repeated_cold)] if repeated_cold else None,"first_cold_seconds":cold,
            "allocated_bytes_observed":sorted(set(r["allocated"] for r in rows)),
            "maximum_closure":max(r["closure"] for r in rows),
            "maximum_published_difference":max(r["published_delta"] for r in rows),
            "maximum_reported_source_metric":max(r["source_metric"] for r in rows),
            "iteration_range":[min(r["iterations"] for r in rows),max(r["iterations"] for r in rows)]}
        result["performance_groups"].append(row);stats[key]=row
    for key,a in sorted(stats.items()):
        if not comparison.get("ratio_allowed",False):continue
        b=stats.get((32,)+key[1:])
        if key[0]!=64 or not b or not a["repeated_cold_median_seconds"] or not b["repeated_cold_median_seconds"]:continue
        if a["requested_tolerances"]!=b["requested_tolerances"]:continue
        result["paired_ratios"].append({"concurrency":key[1],"cells":key[2],"angles":key[3],"topology":key[4],
            "median_time64_over_time32":a["repeated_cold_median_seconds"]/b["repeated_cold_median_seconds"],
            "caution":"Descriptive ratio only; inspect repeated cold-start ranges, residuals, clocks, allocation and warnings."})
    (batch/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    return result
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("batch",type=Path)
    print(json.dumps(summarize(p.parse_args().batch),indent=2))
