"""Native node-local equilibriumODE/parcel exchange and closed-wall budgets.
Requires an already-built gasUGKP and the OpenFOAM environment. Output is local.
The shared gas thermo uses two species; both have the same caloric law here.
"""
from pathlib import Path
import argparse,csv,importlib.util,json,struct,subprocess
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("native",ROOT/"validation/transient_particles/check_native.py")
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);parser.add_argument("--height",type=float,default=.0004);parser.add_argument("--heat",action="store_true");parser.add_argument("--binary",type=Path,default=ROOT/"bin/gasUGKP");parser.add_argument("--steps",type=int,default=10);parser.add_argument("--dt",type=float,default=2e-5);args=parser.parse_args()
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);case=out/"closed-wall"
 native.one_cell(case,1e-6,.00012,args.steps,0,2)
 native.set_entry(case/"system/controlDict","deltaT",str(args.dt))
 native.set_entry(case/"system/controlDict","endTime",str(args.dt*args.steps))
 p=case/"system/blockMeshDict";s=p.read_text().replace("(1 1 1) simpleGrading","(1 4 1) simpleGrading")
 s=s.replace("sides { type empty; faces ((0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); }","walls {type wall; faces ((0 1 5 4) (3 7 6 2));} sides {type empty; faces ((0 3 2 1) (4 5 6 7));}")
 p.write_text(s)
 for p in (case/"0").iterdir():
  if p.suffix==".dat":continue
  bc="walls {type fixedValue; value uniform (0 0 0);}" if p.name=="U" else "walls {type fixedValue; value uniform 0;}" if p.name=="k" else "walls {type omegaWallFunction; value uniform 1;}" if p.name=="omega" else "walls {type zeroGradient;}"
  p.write_text(p.read_text().replace("sides { type empty;",bc+" sides { type empty;"))
 p=case/"0/gpuResidentStrictParticles.dat";data=bytearray(p.read_bytes());start=data.index(b"\n")+1+4;struct.pack_into("<i",data,start+80,0);struct.pack_into("<d",data,start+8,args.height);p.write_bytes(data)
 p=case/"constant/fluidProperties";s=p.read_text();p.write_text(s[:s.index("turbulence")]+"turbulence {simulationType RAS; RAS {model kOmegaSST; turbulence on; printCoeffs on; kOmegaSSTCoeffs {wallTreatment boundaryLayer; boundaryLayer {model equilibriumOde; nodes 8; stretch 2.5; maxIterations 120; relativeTolerance 1e-8; absoluteTolerance 1e-10; writeProfiles true;}}}}\n")
 native.set_entry(case/"constant/particleProperties","particleTemperatureTransport","true" if args.heat else "false")
 native.set_entry(case/"constant/particleProperties","particleGasHeatTransferModel","RanzMarshall" if args.heat else "none")
 for command,name in [(["blockMesh","-case",str(case)],"mesh.log"),([str(args.binary.resolve()),"-case",str(case)],"run.log")]:
  with (out/name).open("w") as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,env=native.ENV)
 def field(directory,name):return native.fields.internal(directory/name,4)
 def budget(directory):
  rho=field(directory,"rho");eps=field(directory,"epsilonS");energy=field(directory,"rhoE")
  momentum=field(directory,"rhoU") if (directory/"rhoU").exists() else [[r[0]*u[0],0,0] for r,u in zip(rho,field(directory,"U"))]
  parcels=native.parcel(directory/"gpuResidentStrictParticles.dat")
  return [sum(.175*(1-e[0])*r[0] for e,r in zip(eps,rho))+sum(p[9] for p in parcels),sum(.175*(1-e[0])*r[0] for e,r in zip(eps,momentum))+sum(p[9]*p[3] for p in parcels),sum(.175*(1-e[0])*r[0] for e,r in zip(eps,energy))+sum(p[9]*(800*p[6]+.5*sum(v*v for v in p[3:6])+1.5*p[7]) for p in parcels)]
 end=native.dirs(case)[-1];a=budget(case/"0");b=budget(end);diff=[y-x for x,y in zip(a,b)]
 assert abs(diff[0])<1e-13 and abs(diff[2])<1e-8,diff
 # Actual intrinsic wall traction, integrated on both stationary adiabatic
 # walls. Euler uses one accepted BVP per interval; phase fraction is frozen
 # in that interval's transport operator.
 impulse=0.;nonzero=set();maximum_heat=0.
 previous=case/"0"
 for time in native.dirs(case)[1:]:
  ep=field(previous,"epsilonS")
  for row in csv.DictReader((time/"uniform/gasBoundaryLayer.csv").open()):
   impulse-=.7*args.dt*(1-ep[int(row["owner"])][0])*float(row["tractionX"])
  for row in csv.DictReader((time/"uniform/gasBoundaryLayerParticleReservoirs.csv").open()):
   if abs(float(row["gasMomentumSourceX"]))>0:nonzero.add(int(row["node"]))
   maximum_heat=max(maximum_heat,abs(float(row["gasHeatSource"])))
  previous=time
 assert abs(diff[1]-impulse)<1e-10,(diff,impulse)
 assert len(nonzero)==1,nonzero
 if args.heat:assert maximum_heat>0
 else:assert maximum_heat==0
 # Momentum includes the resolved stationary-wall impulse; here it is tiny.
 rows=list(csv.DictReader((end/"uniform/gasBoundaryLayerProfiles.csv").open()))
 faces={r["face"] for r in rows};assert len(faces)==2 and len(rows)==16
 for face in faces:assert sorted(int(r["node"]) for r in rows if r["face"]==face)==list(range(8))
 parcels=native.parcel(end/"gpuResidentStrictParticles.dat");assert len(parcels)==1
 if args.heat:assert 1000<parcels[0][6]<10000
 else:assert parcels[0][6]==10000
 report={"status":"PASS","initial":a,"final":b,"difference":diff,"profile_nodes":len(rows),"checkpoint":end.name,"active_nodes":sorted(nonzero),"wall_impulse":impulse,"momentum_balance_error":diff[1]-impulse,"maximum_heat_source":maximum_heat,"parcel":parcels[0],"height":args.height,"heat":args.heat}
 (out/"results.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report))
if __name__=="__main__":main()
