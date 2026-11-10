"""Actual gasUGKP/CHT/CHMT split restart, including the previous binary.
Inputs and outputs are confined to fresh synthetic case directories.
"""
from pathlib import Path
import argparse,importlib.util,json,shutil,subprocess,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("inlet",ROOT/"validation/pr41/native_injection.py")
inlet=importlib.util.module_from_spec(spec);spec.loader.exec_module(inlet)

def execute(binary,case,label,runs):
    start=time.perf_counter()
    with (case/(label+".log")).open("w") as log:
        result=subprocess.run([str(binary),"-case",str(case)],stdout=log,stderr=subprocess.STDOUT)
    runs.append(dict(solver=binary.name,label=label,exit=result.returncode,
                     seconds=time.perf_counter()-start,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()))
    assert result.returncode==0,(case,label,result.returncode)

def positions(case,solver):
    if solver=="CHMT":
        import csv
        output=case/"chmtOutput"
        frame=max(int(p.stem.split("-")[1]) for p in output.glob("particles-*.csv"))
        with (output/f"particles-{frame}.csv").open() as file:rows=list(csv.DictReader(file))
        xyz=np.array([[float(row[k]) for k in ("x","y","z")] for row in rows])
    else:
        frame=inlet.m.times(case)[-1]
        if solver=="gasUGKP":xyz=np.array(inlet.g.parcel(frame/"gpuResidentStrictParticles.dat"),dtype=object)[:,:3].astype(float)
        else:
            chunks=[data for _,data in inlet.wire.iter_chunks(frame/"gpuResidentStrictParticles.dat")]
            xyz=np.concatenate([np.column_stack([data[k] for k in ("px","py","pz")]) for data in chunks])
    return xyz[np.lexsort((np.round(xyz[:,2],12),np.round(xyz[:,1],12)))]

def resumed_case(first,target,solver):
    target.mkdir()
    for item in ("0","constant","system"):shutil.copytree(first/item,target/item)
    if solver=="CHMT":
        checkpoint=first/"chmtOutput/checkpoint-1"
        inlet.g.set_entry(target/"constant/chmtProperties","restartDirectory",'"'+str(checkpoint)+'"')
    else:
        frame=inlet.m.times(first)[-1];shutil.copytree(frame,target/frame.name)
        inlet.g.set_entry(target/"system/controlDict","startFrom","latestTime")
    inlet.g.set_entry(target/"system/controlDict","endTime",2*inlet.DT)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--before-directory",type=Path)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report={"runs":[],"comparisons":[]}
    def save():(out/"results.json").write_text(json.dumps(report,indent=2)+"\n")
    try:
        for solver in ("gasUGKP","CHT","CHMT"):
            binary=ROOT/"bin"/solver
            continuous=out/(solver+"-continuous")
            expected=inlet.prepare(continuous,solver);expected["expected_mass"]*=2
            inlet.g.set_entry(continuous/"system/controlDict","endTime",2*inlet.DT)
            execute(binary,continuous,"run",report["runs"])
            reference=positions(continuous,solver)
            for old in ([False,True] if args.before_directory else [False]):
                tag="old" if old else "new"
                first=out/(solver+"-"+tag+"-first");inlet.prepare(first,solver)
                execute(args.before_directory/solver if old else binary,first,"run",report["runs"])
                resumed=out/(solver+"-"+tag+"-resumed");resumed_case(first,resumed,solver)
                execute(binary,resumed,"run",report["runs"])
                xyz=positions(resumed,solver)
                assert xyz.shape==reference.shape,(solver,tag,xyz.shape,reference.shape)
                maximum=float(np.max(np.abs(xyz-reference)))
                assert maximum<1e-10,(solver,tag,maximum)
                check=inlet.inspect(resumed,solver,expected,out)
                report["comparisons"].append(dict(solver=solver,first_version=tag,position_max_error=maximum,**check))
                save()
        report["status"]="PASS"
    except Exception as error:
        report["status"]="FAIL";report["error"]=repr(error);save();raise
    save();print(json.dumps(report,indent=2))
if __name__=="__main__":main()
