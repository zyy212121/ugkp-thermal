#!/usr/bin/env python3
"""Create an isolated OF10 case from the delivered CPU field and the recorded mesh."""
import argparse, hashlib, json, re, tarfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]

def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument("--archive",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--nodes",type=int,default=24,choices=[24,48,128])
    ap.add_argument("--dt",type=float,default=2e-7)
    ap.add_argument("--duration",type=float,default=5e-5)
    ap.add_argument("--integrator",default="SSPRK2",choices=["Euler","SSPRK2"])
    args=ap.parse_args();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    archive=ROOT/"docs/diagnostics/bvp-coupling-20261020/restart-inputs.tar.gz"
    with tarfile.open(archive) as t:
        for member in t.getmembers():
            prefix="cases/boundaryLayer/"
            if not member.name.startswith(prefix) or not member.isfile():continue
            name=Path(member.name[len(prefix):])
            if name.parts[0] not in {"0","constant","system"}:continue
            assert not name.is_absolute() and ".." not in name.parts
            p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(t.extractfile(member).read())
    inputs=args.archive.resolve()/"complete-coupling/twod/input-exactR"
    arrays={n:np.fromfile(inputs/(f+".bin"),dtype="<f8")for n,f in [("p","p"),("T","Tgas"),("rho","rho"),("rhoE","rhoE"),("k","k"),("omega","omega")]}
    arrays["U"]=np.stack([np.fromfile(inputs/(x+".bin"),dtype="<f8")for x in ["Ux","Uy","Uz"]],axis=1)
    for species in ["Air","CO2"]:arrays["Y_"+species]=np.fromfile(inputs/("speciesRho"+species+".bin"),dtype="<f8")/arrays["rho"]
    for name,a in arrays.items():
        p=out/"0"/name;text=p.read_text();vector=a.ndim==2
        body="\n".join("("+" ".join(format(x,".17g")for x in row)+")"for row in a)if vector else "\n".join(format(x,".17g")for x in a)
        value="internalField nonuniform List<"+("vector"if vector else"scalar")+">\n"+str(len(a))+"\n(\n"+body+"\n);"
        text,n=re.subn(r"internalField\s+uniform\s+[^;]+;",lambda _:value,text);assert n==1,p;p.write_text(text)
    p=out/"system/controlDict";text=p.read_text()
    changes=dict(startFrom="startTime",startTime="0",endTime=format(args.duration,".17g"),deltaT=format(args.dt,".17g"),writeInterval=format(args.duration,".17g"),logInterval="1",adjustTimeStep="false")
    for key,value in changes.items():text=re.sub(r"\b"+key+r"\s+[^;]+;",key+" "+value+";",text)
    p.write_text(text)
    p=out/"constant/fluidProperties";text=p.read_text();text=re.sub(r"nodes\s+\d+;",f"nodes {args.nodes};",text);text=re.sub(r"maxIterations\s+\d+;","maxIterations 100;",text);p.write_text(text)
    p=out/"system/fvSchemes";text=p.read_text().replace("default SSPRK2;",f"default {args.integrator};");p.write_text(text)
    manifest={str(p.relative_to(inputs)):hashlib.sha256(p.read_bytes()).hexdigest()for p in inputs.iterdir()if p.is_file()}
    (out/"input-manifest.json").write_text(json.dumps(dict(template_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),inputs=manifest,nodes=args.nodes,dt=args.dt,duration=args.duration,integrator=args.integrator),indent=2))
    print(out)

if __name__=="__main__":main()
