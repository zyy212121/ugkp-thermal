"""Exercise the actual steady driver: sustained inlet coverage and restart.
Run after sourcing OpenFOAM. All cases are fresh synthetic fixture copies.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,re,shutil,subprocess,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("inlet_native",ROOT/"validation/pr41/native_injection.py")
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)

def metadata(path):
    fields={}
    for line in path.read_text().splitlines():
        key,value=line.split();fields[key]=int(value) if key=="sampleCount" else float(value)
    return fields

def particles(case):
    frame=native.m.times(case)[-1]
    rows=np.asarray(native.g.parcel(frame/"gpuResidentStrictParticles.dat"),dtype=object)
    xyz=np.asarray(rows[:,:3],dtype=float)
    order=np.lexsort((xyz[:,2],xyz[:,1],xyz[:,0]))
    residual=(frame/"gpuResidentStrictSourceResidual.dat").read_text().split()
    return frame,xyz[order],sum(float(v) for v in rows[:,9]),sum(map(float,residual[3::2]))

def setup(case,steps):
    native.prepare(case,"gasUGKPSteady")
    control=case/"system/controlDict"
    for key,value in dict(endTime=steps,writeInterval=32,windowSamples=32,minimumWindows=100000).items():
        native.g.set_entry(control,key,value)
    native.g.set_entry(case/"constant/particleProperties","parcelMass",1e-10)

def execute(binary,case,label,report):
    start=time.perf_counter()
    with (case/(label+".log")).open("w") as log:
        result=subprocess.run([str(binary),"-case",str(case)],stdout=log,stderr=subprocess.STDOUT)
    record=dict(label=label,exit=result.returncode,wall_seconds=time.perf_counter()-start,
                binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    report["runs"].append(record)
    assert result.returncode==0,(case,label,result.returncode)
    return record

def resume(case,steps):
    control=case/"system/controlDict"
    native.g.set_entry(control,"startFrom","latestTime")
    native.g.set_entry(control,"endTime",steps)

def compare(a,b):
    fa,xa,ma,ra=particles(a);fb,xb,mb,rb=particles(b)
    assert xa.shape==xb.shape,(a,b,xa.shape,xb.shape)
    maximum=float(np.max(np.abs(xa-xb)))
    assert maximum<1e-10,(a,b,maximum)
    assert abs(ma+ra-mb-rb)<1e-16
    return dict(count=len(xa),maximum_position_error=maximum,mass_a=ma,mass_b=mb,
                residual_a=ra,residual_b=rb,metadata_a=metadata(fa/"steadyState.dat"),
                metadata_b=metadata(fb/"steadyState.dat"))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--before-binary",type=Path)
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=ROOT/"bin/gasUGKPSteady"
    report={"runs":[]}
    def save():(out/"results.json").write_text(json.dumps(report,indent=2)+"\n")
    try:
        continuous=out/"continuous";setup(continuous,64);execute(binary,continuous,"run",report)
        split=out/"split";setup(split,32);execute(binary,split,"first",report)
        checkpoint=out/"checkpoint32";shutil.copytree(split,checkpoint)
        resume(split,64);execute(binary,split,"resumed",report)
        report["continuous_vs_restart"]=compare(continuous,split)
        frame,xyz,mass,residual=particles(continuous)
        unique=len(np.unique(np.round(xyz[:,1:]/[.2,.01],9),axis=0))
        assert unique>.9*len(xyz),(unique,len(xyz))
        assert metadata(frame/"steadyState.dat")["sampleCount"]==64
        assert metadata(frame/"steadyState.dat")["boundaryTime"]==0
        expected=64*.001*2800*.002*1e-7
        assert abs(mass+residual-expected)<expected*1e-9
        report["coverage"]=dict(parcels=len(xyz),unique_transverse_positions=unique,
            normalized_rounding_digits=9,mass=mass,residual=residual,expected_mass=expected)
        # Existing-format checkpoint, sampleCount larger than Foam::label on DPInt32.
        high=out/"large-count";shutil.copytree(checkpoint,high)
        path=high/"32/steadyState.dat"
        path.write_text(re.sub(r"sampleCount \d+","sampleCount "+str(2**40),path.read_text()))
        resume(high,64);execute(binary,high,"resumed",report)
        highmeta=metadata(high/"64/steadyState.dat")
        assert highmeta["sampleCount"]==2**40+32,highmeta
        report["large_count_restart"]=highmeta
        # Old cases without clock metadata remain runnable, with a fresh epoch.
        missing=out/"without-clock-metadata";shutil.copytree(checkpoint,missing)
        (missing/"32/steadyState.dat").unlink()
        resume(missing,64);execute(binary,missing,"resumed",report)
        missingmeta=metadata(missing/"64/steadyState.dat")
        assert missingmeta["sampleCount"]==32 and missingmeta["boundaryTime"]==0,missingmeta
        report["missing_metadata_restart"]=dict(metadata=missingmeta,
            continuity="Fresh deterministic epoch; historical stream cannot be recovered without metadata.")
        if args.before_binary:
            before=args.before_binary.resolve()
            old=out/"old-version";setup(old,64);execute(before,old,"run",report)
            _,oldxyz,oldmass,oldresidual=particles(old)
            oldunique=len(np.unique(np.round(oldxyz[:,1:]/[.2,.01],9),axis=0))
            assert oldunique<unique
            assert abs(oldmass+oldresidual-mass-residual)<expected*1e-9
            report["old_coverage"]=dict(parcels=len(oldxyz),unique_transverse_positions=oldunique)
            legacy=out/"old-checkpoint";setup(legacy,32);execute(before,legacy,"old-first",report)
            resume(legacy,64);execute(binary,legacy,"new-resumed",report)
            legacy_meta=metadata(legacy/"64/steadyState.dat")
            assert legacy_meta["sampleCount"]==64 and legacy_meta["boundaryTime"]==0
            report["old_format_restart"]=legacy_meta
        report["status"]="PASS"
    except Exception as error:
        report["status"]="FAIL";report["error"]=repr(error);save();raise
    save();print(json.dumps(report,indent=2))
if __name__=="__main__":main()
