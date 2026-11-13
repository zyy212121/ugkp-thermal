"""Compile/link the loader fixture only; never execute it."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess
def main():
 p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--bits",type=int,choices=(32,64),required=True);a=p.parse_args()
 source=a.source.resolve();a.output.mkdir(parents=True,exist_ok=True)
 foam=Path(os.environ.get("WM_PROJECT_DIR","/opt/openfoam10"))
 lib=os.environ.get("FOAM_LIBBIN",str(foam/"platforms/linux64GccDPInt32Opt/lib"))
 binary=a.output/f"loader-fp{a.bits}";log=a.output/f"loader-fp{a.bits}.build.log"
 argv=["g++","-std=c++17","-O2","-DNoRepository","-Dlinux64","-DWM_DP","-DWM_LABEL_SIZE=32",
       "-I"+str(source/"common"),"-I"+str(foam/"src/OpenFOAM/lnInclude"),"-I"+str(foam/"src/OSspecific/POSIX/lnInclude"),
       str(source/"validation/solver_controls/loader_fixture.cpp"),"-L"+lib,"-Wl,-rpath,"+lib,
       "-lOpenFOAM","-ldl","-lm","-pthread","-o",str(binary)]
 with log.open("x") as out:status=subprocess.run(argv,stdout=out,stderr=subprocess.STDOUT)
 record=dict(argv=argv,exit=status.returncode,executable_run=False,source=str(source),bits=a.bits,log=str(log))
 if binary.exists():record["sha256"]=hashlib.sha256(binary.read_bytes()).hexdigest()
 (a.output/f"loader-fp{a.bits}.compile.json").write_text(json.dumps(record,indent=2)+"\n")
 if status.returncode:raise SystemExit(status.returncode)
if __name__=="__main__":main()
