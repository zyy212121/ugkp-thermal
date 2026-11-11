"""Acceptance-only collection of failures and their existing dependencies."""
import traceback
from pathlib import Path
FATAL_GPU = ("cudaerrorillegaladdress", "illegal memory access", "cudaerrorlaunchfailure",
             "unspecified launch failure", "device-side assert", "cudaerrorassert",
             "cudaerrorunknown", "cudaerrorinitializationerror", "no cuda-capable device",
             "driver shutting down", "gpu has fallen off", "cuda out of memory",
             "cudaerrormemoryallocation", "cudaerrordeviceunavailable")
FATAL_SYSTEM = ("no space left on device", "input/output error", "error while loading shared libraries")
def failed_environment_scope(text, code=None, gpu=True):
    # Called only after an original step already failed. This never changes PASS.
    lower=text.lower()
    for token in FATAL_SYSTEM:
        if token in lower: return {"scope":"all", "reason":token}
    for token in FATAL_GPU:
        if token in lower: return {"scope":"gpu", "reason":token}
    if gpu and code in (-11,-7,-4):
        return {"scope":"gpu", "reason":"Failed GPU process signal %s; dependent GPU execution is invalid" % code}
    return None
class BranchCollector:
    def __init__(self, report, output, save):
        self.report,self.output,self.save=report,Path(output),save
        report["branches"]=[];self.values={};self.outcomes={}
    def attempt(self, key, action, dependencies=()):
        rec={"id":key,"dependencies":list(dependencies)}
        missing=[d for d in dependencies if self.outcomes.get(d)!="RUN_PASS"]
        if self.report.get("environment_failure") or missing:
            rec.update(result="SKIP_DEPENDENCY",reason=self.report.get("environment_failure") or {"failed_dependencies":missing})
        else:
            first_run=len(self.report.get("runs",[]))
            try:
                self.values[key]=action();rec["result"]="RUN_PASS"
            except Exception as error:
                rec.update(result="FAIL",error=repr(error),traceback=traceback.format_exc())
                for run in self.report.get("runs",[])[first_run:]:
                    log=Path(run.get("log", ""))
                    text=log.read_text(errors="replace") if log.is_file() else ""
                    problem=failed_environment_scope(text,run.get("exit"))
                    if problem:self.report["environment_failure"]=problem;break
                if not self.report.get("environment_failure"):
                    problem=failed_environment_scope(repr(error))
                    if problem:self.report["environment_failure"]=problem
        self.outcomes[key]=rec["result"];self.report["branches"].append(rec);self.save()
        return self.values.get(key)
    def finish(self):
        self.report["status"]="PASS" if all(x=="RUN_PASS" for x in self.outcomes.values()) else "FAIL"
        self.save()
        return 0 if self.report["status"]=="PASS" else 1
