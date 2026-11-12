#!/usr/bin/env python3
"""Actual-state restart matrix. Run only inside the declared frozen batch."""
from pathlib import Path
import argparse, importlib.util, json, os, shutil, subprocess, time, traceback
from acceptance_support import BranchCollector
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ROOT=Path(os.environ.get("UGKP_SOURCE_ROOT",Path(__file__).resolve().parents[3])).resolve()
native=module("dual_native",Path(__file__).with_name("native_dual.py"))
steady=module("steady_native",ROOT/"validation/pr41/native_steady_restart.py")
def write_precision(case,bits):
    native.configure_precision(case,bits)
def outputs(case,solver):
    if solver=="CHMT": return sorted(str(p.relative_to(case)) for p in (case/"chmtOutput").glob("checkpoint-*"))
    return sorted(p.name for p in native.inlet.m.times(case))
def latest_checkpoint(case,solver):
    if solver=="CHMT": return max((case/"chmtOutput").glob("checkpoint-*"),key=lambda p:int(p.name.split("-")[1]))
    return native.inlet.m.times(case)[-1]
def precision_record(case,solver):
    p=latest_checkpoint(case,solver)/"ugkpPrecision"
    text=p.read_text(); values={}
    for line in text.splitlines():
        fields=line.rstrip(";").split()
        if len(fields)==2: values[fields[0]]=fields[1]
    return values
def clone_restart(first,target,solver):
    if solver=="gasUGKPSteady":
        shutil.copytree(first,target);steady.resume(target,64)
    else:
        native.restart.resumed_case(first,target,solver)
        if solver=="CHMT":
            checkpoint=target/"restart-input";shutil.copytree(latest_checkpoint(first,solver),checkpoint)
            native.inlet.g.set_entry(target/"constant/chmtProperties","restartDirectory",'"'+str(checkpoint)+'"')
    return target/"restart-input" if solver=="CHMT" else native.inlet.m.times(target)[-1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--fixtures",type=Path,required=True);args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report={"status":"RUNNING","runs":[],"checks":[]}
    def save(): (out/"results.json").write_text(json.dumps(report,indent=2)+"\n")
    def launch(case,solver,bits,label,expect="success",default=False):
        if not default: write_precision(case,bits)
        else: native.configure_precision(case,bits,select=False)
        previous=outputs(case,solver);start=time.perf_counter()
        log=case/(label+".log")
        with log.open("w") as stream:
            proc=subprocess.run([str(ROOT/"bin"/solver),"-case",str(case)],env=native.ENV,
                                stdout=stream,stderr=subprocess.STDOUT)
        text=log.read_text(errors="replace")
        rec={"case":str(case),"solver":solver,"selected_precision":bits,"default_selection":default,
             "exit":proc.returncode,"log":str(log),"seconds":time.perf_counter()-start,"expectation":expect,
             "warning_lines":[s for s in text.splitlines() if "warning" in s.lower()]}
        report["runs"].append(rec);save()
        if expect=="reject_32_to_64":
            assert proc.returncode==2,(case,proc.returncode)
            assert "restart precision 32 is lower than selected precision 64" in text
            assert "restart stopped." in text
            assert outputs(case,solver)==previous,(case,"state advanced after expected rejection")
        else:
            assert proc.returncode==0,(case,proc.returncode)
            if expect=="warn_64_to_32":
                assert "restart precision 64 is higher than selected precision 32" in text
                assert "continuing at precision 32." in text
            if expect=="warn_unknown":
                assert "input state precision is unknown" in text
                assert "continuing with the existing reader at precision "+str(bits) in text
            record=precision_record(case,solver)
            assert record["computePrecision"]==str(bits) and record["restartPrecision"]==str(bits),record
            rec["published_precision"]=record;rec["fields"]=native.finite_outputs(case,solver)
            if solver=="gasUGKPSteady":
                clock=steady.metadata(latest_checkpoint(case,solver)/"steadyState.dat")
                rec["steady_clock"]=clock
        save();return rec
    collector=BranchCollector(report,out,save)
    for solver in ("gasUGKP","gasUGKPSteady","CHT","CHMT"):
        firsts={};successes={}
        for bits in (64,32):
            def produce():
                first=out/(solver+"-first-"+str(bits))
                if solver=="gasUGKPSteady": steady.setup(first,32)
                else: native.inlet.prepare(first,solver)
                launch(first,solver,bits,"first",default=(bits==32))
                return first
            firsts[bits]=collector.attempt(solver+"-first-"+str(bits),produce)
        for producer,consumer in ((64,64),(32,32),(64,32),(32,64)):
            key=solver+"-"+str(producer)+"-to-"+str(consumer)
            def resume_known():
                case=out/(solver+"-"+str(producer)+"-to-"+str(consumer))
                clone_restart(firsts[producer],case,solver)
                expect="reject_32_to_64" if producer<consumer else ("warn_64_to_32" if producer>consumer else "success")
                rec=launch(case,solver,consumer,"resumed",expect)
                if expect!="reject_32_to_64":
                    native.PROBE=args.fixtures/("fp"+str(consumer))/"source_residual"
                    xyz,mass,residual=native.particle_inventory(case,solver,out)
                    expected=(64*.001*2800*.002*1e-7 if solver=="gasUGKPSteady" else
                        2*(.001*(2500 if solver=="CHMT" else 2800)*(1 if solver in ("CHMT","CHT") else .002)*1e-7))
                    relative=abs(mass+residual-expected)/expected
                    assert relative<=native.MASS_RTOL,(solver,producer,consumer,relative)
                    rec.update(mass=mass,source_residual=residual,expected_mass=expected,
                               mass_relative_error=relative,mass_relative_tolerance=native.MASS_RTOL)
                    if solver=="gasUGKPSteady": assert rec["steady_clock"]["sampleCount"]==64
                    return case
                save()
            successes[(producer,consumer)]=collector.attempt(key,resume_known,(solver+"-first-"+str(producer),))
        for producer,consumer in ((64,64),(32,32),(64,32),(32,64)):
            def resume_unknown():
                case=out/(solver+"-unknown-"+str(producer)+"-to-"+str(consumer))
                checkpoint=clone_restart(firsts[producer],case,solver)
                (checkpoint/"ugkpPrecision").unlink()
                launch(case,solver,consumer,"unknown","warn_unknown")
            collector.attempt(solver+"-unknown-"+str(producer)+"-to-"+str(consumer),resume_unknown,(solver+"-first-"+str(producer),))
        def compare_fields():
            compare=native.field_comparison(successes[(64,32)],successes[(32,32)],solver)
            report["checks"].append({"solver":solver,"cross_precision_fields":compare})
        collector.attempt(solver+"-cross-fields",compare_fields,(solver+"-64-to-32",solver+"-32-to-32"))
    if collector.finish(): raise SystemExit(1)
if __name__=="__main__":main()
