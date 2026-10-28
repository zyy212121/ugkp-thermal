#!/usr/bin/env python3
"""Prepare and compare a native CHT cold-wall particle/solid exchange case.

The ordinary CHT models are selected through input dictionaries. This fixture
puts one already deposited particle on each coupled face, with nonuniform
temperature and a complete current-schema cold-wall profile. No GPU execution
occurs with --prepare-only. Run all other modes from sourced OpenFOAM 10.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("interface_layer1",HERE/"layer1.py")
layer=importlib.util.module_from_spec(spec);spec.loader.exec_module(layer)
matrix=layer.matrix
sys.path.insert(0,str(layer.SOURCE/"tool/restart"))
import thermal_particle_restart_current as wire


def list_body(path):
    text=re.sub(r"/\*.*?\*/","",path.read_text(),flags=re.S)
    text=re.sub(r"//[^\n]*","",text)
    text=text[text.index("}")+1:]
    return text[text.index("(")+1:text.rfind(")")]


def prepare(case,grid,end,viscosity="const"):
    meta=json.loads((matrix.HERE/"fixtures.json").read_text())["cht-contact"]
    layer.prepare("cht-contact",case,meta,grid)
    # Keep transport an explicit input choice. The upstream none/none particle
    # model combination does not snapshot couplingTgasOld, which its thermal
    # relaxation reads for Sutherland viscosity before checking model switches.
    # Constant viscosity isolates the interface reduction from that existing bug.
    if viscosity == "const":
        (case/"constant/gasTransportProperties").write_text(
            "FoamFile {version 2.0; format ascii; class dictionary; object gasTransportProperties;}\n"
            "viscosityModel const;\n")
    path=case/"constant/schedulingProperties"
    text=path.read_text().replace("gpuResidentPureGasOnly true;","gpuResidentPureGasOnly false;")
    text=text.replace("gpuResidentParticleCapacity 0;",f"gpuResidentParticleCapacity {2*grid**3};")
    path.write_text(text)
    path=case/"constant/particleProperties"
    path.write_text(path.read_text()+"""
particleTemperatureTransport true;
particleTMin 100;
particleTMax 5000;
gpuResidentStuckWallPatches (left);
gpuResidentStuckModel {
 heatTransfer true;
 wallInteractionModel coldWall1D;
 wallTransientResistance true;
 solidificationIterations 12;
 meltingTemperature 2327; mushyRange 20; latentHeat 1160000;
 solidDensity 3990; solidSpecificHeat 1273; solidThermalConductivity 5.9;
 maximumCoverage 0.63;
 depositionHeatTransferEfficiency 1;
 reflectionHeatTransferEfficiency 1;
}
""")
    (case/"0/Tp").write_text("""FoamFile {version 2.0;format ascii;class volScalarField;object Tp;}
dimensions [0 0 0 1 0 0 0]; internalField uniform 1000;
boundaryField {left{type zeroGradient;}right{type zeroGradient;}sides{type zeroGradient;}}
""")
    matrix.entry(case/"system/controlDict","endTime",end)
    mesh=case/"constant/polyMesh"
    points=np.array([[float(v) for v in row.split()] for row in re.findall(r"\(([^()]*)\)",list_body(mesh/"points"))])
    faces=[[int(v) for v in row.split()] for row in re.findall(r"\d+\(([^()]*)\)",list_body(mesh/"faces"))]
    owner=[int(v) for v in list_body(mesh/"owner").split()]
    patch=re.search(r"\bleft\s*\{([^}]+)\}",(mesh/"boundary").read_text(),re.S)[1]
    count=int(re.search(r"nFaces\s+(\d+)",patch)[1]);start=int(re.search(r"startFace\s+(\d+)",patch)[1])
    assert count==grid**2
    data={name:np.zeros((count,width) if width>1 else count,dtype=dtype)
          for name,dtype,width in wire.field_specs(wire.CHT_MARKER)}
    temperature=1000+(10*np.arange(count))%110
    for i in range(count):
        face=start+i;centre=np.mean(points[faces[face]],axis=0);centre[0]+=1e-7
        for name,value in zip(("px","py","pz"),centre):data[name][i]=value
        data["cell"][i]=owner[face]
    data["status"][:]=1;data["pStuck"][:]=1
    data["pStuckFaceId"][:]=np.arange(start,start+count)
    diameter=.0002;area=np.pi*(diameter/2)**2
    data["pd"][:]=diameter;data["pm"][:]=1e-8;data["pT"][:]=temperature
    data["rng"][:]=np.arange(1,count+1,dtype=np.uint64)*1020301
    data["orig_id"][:]=np.arange(1,count+1,dtype=np.uint64)
    data["pDepositionArea"][:]=area;data["pContactMaximumArea"][:]=area
    data["pContactDuration"][:]=.001;data["pContactPeakFraction"][:]=.5
    data["pColdNodeSpecificEnthalpy"][:]=(1273*temperature)[:,None]
    data["pColdRingSolidMass"][:]=3990*(np.pi/6)*diameter**3/8
    data["pColdFrozenArea"][:]=area;data["pColdContactAge"][:]=.001
    wire.write_restart(case/"0/gpuResidentStrictParticles.dat",[data],count,262144,cht=True)
    (case/"0/gpuResidentStrictEpsGPrev.dat").write_text(str(grid**3)+"\n"+"\n".join(["1"]*grid**3)+"\n")
    (case/"0/gpuResidentStrictSourceResidual.dat").write_text("UGKP_SOURCE_RESIDUAL_SCHEMA1 0\n")
    return dict(grid=grid,particles=count,model="coldWall1D",viscosity_model=viscosity,
                initial_particle_profile_energy_J=float(np.sum(data["pm"]*(1273*temperature))),
                statistical_mass_kg=float(data["pm"].sum()),duration_s=end,
                coupling_interval_s=.00002)


def particles(snapshot):
    chunks=[data for _,data in wire.iter_chunks(snapshot/"gpuResidentStrictParticles.dat")]
    fields={key:np.concatenate([data[key] for data in chunks]) for key in chunks[0]}
    order=np.argsort(fields["orig_id"])
    return {key:value[order] for key,value in fields.items()}


def particle_inventory(snapshot):
    fields=particles(snapshot)
    assert np.all(fields["status"]==1) and np.all(fields["pStuck"]==1)
    assert np.all(fields["pux"]==0) and np.all(fields["puy"]==0) and np.all(fields["puz"]==0)
    # The production cold-wall owner stores a uniform FP64 specific-enthalpy
    # carry in addition to its eight FP32 nodes (GpuColdWall1DPublish.inl).
    h=np.mean(fields["pColdNodeSpecificEnthalpy"].astype(np.float64),axis=1)
    h+=fields["pThermalEnthalpyResidual"]
    assert np.all(np.isfinite(h)) and np.all(np.isfinite(fields["pm"]))
    return math.fsum(fields["pm"]*h),math.fsum(fields["pm"])


def gas_inventory(snapshot,grid):
    # Conserved gas fields are intrinsic densities; the checkpoint porosity
    # converts them to the physical gas inventory in each unit-cube cell.
    values=(snapshot/"gpuResidentStrictEpsGPrev.dat").read_text().split()
    count=grid**3
    assert int(values[0])==count and len(values)==count+1
    eps=[float(x) for x in values[1:]]
    assert all(0 < x <= 1 for x in eps)
    mass=math.fsum(e*r for e,r in zip(eps,layer.field(snapshot/"rho",count)))/count
    energy=math.fsum(e*r for e,r in zip(eps,layer.field(snapshot/"rhoE",count)))/count
    return mass,energy


def patch_flux(snapshot,name,grid):
    text=(snapshot/name).read_text()
    patch=re.search(r"\bleft\s*\{([^}]+)\}",text,re.S)[1]
    match=re.search(r"value\s+uniform\s+("+layer.NUMBER+r")\s*;",patch)
    if match:return float(match[1])
    match=re.search(r"value\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;",patch,re.S)
    assert match,(snapshot,name)
    values=[float(x) for x in re.findall(layer.NUMBER,match[2])]
    assert len(values)==int(match[1])==grid**2
    return math.fsum(values)/grid**2


def audit(case,grid):
    states=[];residuals=dict(total_energy=0.,gas_exchange=0.,particle_exchange=0.,solid_exchange=0.,mass=0.)
    particle_flux=[]
    for snapshot in matrix.times(case):
        ep,mp=particle_inventory(snapshot)
        gm,ge=gas_inventory(snapshot,grid)
        se=1e6*layer.scalar_sum(snapshot/"solid/T",grid**3)
        state=dict(time=float(snapshot.name),gas_energy=ge,solid_energy=se,particle_energy=ep,
                   total_energy=ge+se+ep,mass=gm+mp)
        qg=patch_flux(snapshot,"gasConvectiveWallHeatFlux",grid)
        qp=patch_flux(snapshot,"particleStuckWallHeatFlux",grid)+patch_flux(snapshot,"particleReflectedWallHeatFlux",grid)
        assert all(math.isfinite(v) for v in state.values()),(case,state)
        assert math.isfinite(qg) and math.isfinite(qp),(snapshot,qg,qp)
        particle_flux.append(qp)
        if states:
            previous=states[-1];dt=state["time"]-previous["time"]
            residuals["gas_exchange"]=max(residuals["gas_exchange"],abs(ge-previous["gas_energy"]+dt*qg))
            residuals["particle_exchange"]=max(residuals["particle_exchange"],abs(ep-previous["particle_energy"]+dt*qp))
            residuals["solid_exchange"]=max(residuals["solid_exchange"],abs(se-previous["solid_energy"]-dt*(qg+qp)))
            residuals["total_energy"]=max(residuals["total_energy"],abs(state["total_energy"]-states[0]["total_energy"]))
            residuals["mass"]=max(residuals["mass"],abs(state["mass"]-states[0]["mass"]))
        states.append(state)
    assert len(states)>=2
    assert max(abs(q) for q in particle_flux)>0,"particle wall exchange was not exercised"
    assert residuals["mass"]<1e-10,(case,residuals)
    assert max(residuals[k] for k in residuals if k!="mass")<2e-5,(case,residuals)
    return dict(states=states,max_absolute_residual=residuals,max_particle_wall_power_W=max(abs(q) for q in particle_flux),
                scope="closed gas + solid + deposited-particle inventory between written coupled endpoints")


def compare(reference,result):
    fields=layer.compare_fields(reference,result);maxima={}
    a,b=matrix.times(reference),matrix.times(result)
    assert len(a)==len(b)
    for x,y in zip(a,b):
        left,right=particles(x),particles(y)
        assert left.keys()==right.keys()
        for key,l in left.items():
            r=right[key];assert l.shape==r.shape
            if l.dtype.kind in "iub":assert np.array_equal(l,r),(key,x,y)
            else:
                assert np.all(np.isfinite(l)) and np.all(np.isfinite(r)),key
                error=float(np.max(np.abs(l.astype(float)-r)/np.maximum(1,np.maximum(np.abs(l),np.abs(r))))) if l.size else 0.
                maxima[key]=max(maxima.get(key,0.),error)
                assert error<2e-12,(key,error)
    return dict(fields=fields,maximum_scaled_particle_errors=maxima)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work",type=Path,required=True)
    parser.add_argument("--baseline",type=Path)
    parser.add_argument("--candidate",type=Path)
    parser.add_argument("--grid",type=int,default=16)
    parser.add_argument("--end-time",type=float,default=.0002)
    parser.add_argument("--repeats",type=int,default=5)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--viscosity",choices=("const","sutherland"),default="const",
                        help="explicit transport input; baseline none/none particle models fail with Sutherland")
    args=parser.parse_args()
    args.work=args.work.resolve()
    if args.baseline:args.baseline=args.baseline.resolve()
    if args.candidate:args.candidate=args.candidate.resolve()
    assert args.grid>=2 and args.end_time>=.00004 and args.repeats>=0
    args.work.mkdir(parents=True,exist_ok=False)
    report=dict(runs=[],timing_scope="native process elapsed including startup/output; mesh generation excluded")
    if args.prepare_only:
        report["input"]=prepare(args.work/"input",args.grid,args.end_time,args.viscosity)
    else:
        assert args.baseline and args.candidate
        report["binaries"]={kind:layer.digest(folder/"CHT") for kind,folder in (("baseline",args.baseline),("candidate",args.candidate))}
        reference=None;samples={"baseline":[],"candidate":[]}
        for repeat in range(args.repeats+1):
            order=("baseline","candidate") if repeat%2==0 else ("candidate","baseline")
            for kind in order:
                case=args.work/f"{kind}-{repeat}"
                configuration=prepare(case,args.grid,args.end_time,args.viscosity)
                elapsed=matrix.run([getattr(args,kind)/"CHT","-case",case],case/"run.log")
                if reference is None:reference=case
                row=dict(kind=kind,repeat=repeat,warmup=repeat==0,elapsed_seconds=elapsed,
                         configuration=configuration,audit=audit(case,args.grid),parity=compare(reference,case))
                report["runs"].append(row)
                if repeat:samples[kind].append(elapsed)
                (args.work/"results.json").write_text(json.dumps(report,indent=2)+"\n")
                print(json.dumps({k:row[k] for k in ("kind","repeat","elapsed_seconds")}),flush=True)
        report["samples"]=samples
        if args.repeats:
            report["median_seconds"]={key:statistics.median(value) for key,value in samples.items()}
            report["speedup"]=report["median_seconds"]["baseline"]/report["median_seconds"]["candidate"]
    (args.work/"results.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({key:value for key,value in report.items() if key not in ("runs","samples")}),flush=True)

if __name__=="__main__":main()
