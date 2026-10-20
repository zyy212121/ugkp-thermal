"""CPU-only review: python3 reproduce.py [--output OUTPUT_DIR].
Requires Python 3.9+, NumPy, g++ with C++17. No OpenFOAM/CUDA installation.
All live paths are resolved relative to this file or the supplied output directory.
"""
from pathlib import Path
import argparse, csv, hashlib, json, os, re, shutil, subprocess, sys, time
import numpy as np

B=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument("--output",type=Path,default=B/"results")
args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
source=B/"source/ugkp";commands=[]
def run(command,cwd=None,label="command"):
    t=time.monotonic()
    p=subprocess.run([str(x) for x in command],cwd=cwd or B,capture_output=True,text=True)
    (out/(label+".log")).write_text(p.stdout+p.stderr)
    commands.append({"command":[str(x) for x in command],"cwd":str(cwd or B),
                     "exit":p.returncode,"seconds":time.monotonic()-t,"log":label+".log"})
    if p.returncode:
        print((p.stdout+p.stderr)[-5000:]);raise RuntimeError(label+" failed")
    print(label, "exit=0", flush=True)
    return p.stdout
def rows(p):return list(csv.DictReader(p.open()))
def field(path,n,vec=False):
    b=path.read_bytes();kind=b"vector" if vec else b"scalar"
    m=re.search(rb"internalField\s+nonuniform\s+List<"+kind+rb">\s+(\d+)\s*\(",b)
    if m:
        size=int(m[1]);assert size==n,(path,size,n)
        if re.search(rb"format\s+binary;",b[:1000]):
            v=np.frombuffer(b,dtype="<f8",count=n*(3 if vec else 1),offset=m.end()).copy()
        else:
            end=b.index(b";",m.end());v=np.fromstring(b[m.end():end].replace(b"(",b" ").replace(b")",b" ").decode(),sep=" ")
        return v.reshape((-1,3)) if vec else v
    m=re.search(rb"internalField\s+uniform\s+([^;]+);",b);assert m,path
    v=np.fromstring(m[1].replace(b"(",b" ").replace(b")",b" ").decode(),sep=" ")
    return np.tile(v,(n,1)) if vec else np.full(n,v[0])
def export_latest():
    dest=out/"latest-input";shutil.copytree(B/"data/ugkp-initial",dest,dirs_exist_ok=True)
    case=B/"cases/ugkp-wallFunction"
    tp=next(p for p in case.iterdir() if p.is_dir() and re.fullmatch("[0-9.eE+-]+",p.name) and abs(float(p.name)-.0233)<1e-12)
    n=int((dest/"counts.txt").read_text().split()[0])
    mapping={"T":"Tgas","rho":"rho","p":"p","k":"k","omega":"omega","nut":"nut","rhoE":"rhoE",
             "rhoY_Air":"speciesRhoAir","rhoY_CO2":"speciesRhoCO2"}
    for fname,arrayname in mapping.items():field(tp/fname,n).astype("<f8").tofile(dest/(arrayname+".bin"))
    for name in ["U","rhoU"]:
        v=field(tp/name,n,True)
        for j,d in enumerate("xyz"):v[:,j].astype("<f8").tofile(dest/(name+d+".bin"))
    return dest
def read_array(root,name,integer=False):
    return np.fromfile(root/(name+".bin"),dtype="<i4" if integer else "<f8")
def wall_decomposition(label,path):
    # Independent transcription of matched EEqn algebra, using frozen cell and wall states.
    mu=1.7244421537657099e-5;cp=1004.675;R=287.0500676;Pr=.72;Prt=.9;ratio=Pr/Prt
    of_constants=json.loads((B/"provenance/of10-runtime-constants.json").read_text())
    kappa=.41;E=9.8;cmu=.09
    P=9.24*(ratio**.75-1)*(1+.28*np.exp(-.007*ratio))
    lo=1/(kappa*ratio);hi=2*lo
    eq=lambda y:kappa*ratio*y-np.log(E*y)-kappa*P
    while eq(hi)<0:hi*=2
    for _ in range(80):
        mid=(lo+hi)/2
        if eq(mid)>0:hi=mid
        else:lo=mid
    yt=(lo+hi)/2
    d={name:read_array(path,name) for name in ["Cx","Cy","Cz","Ux","Uy","Uz","rho","p","k","omega","nut","Tgas",
          "riemannBoundaryT","riemannBoundaryUx","riemannBoundaryUy","riemannBoundaryUz","sstWallDistance","deltaCoeffs"]}
    owner=read_array(path,"faceOwner",True);kind=read_array(path,"riemannBoundaryKind",True)
    records=[]
    for f in np.flatnonzero(kind==2):
        c=int(owner[f]);wall=np.array([d["riemannBoundaryU"+a][f] for a in "xyz"])
        vel=np.array([d["U"+a][c] for a in "xyz"]);u=float(np.linalg.norm(vel-wall));uw=float(np.linalg.norm(wall))
        Tw=d["riemannBoundaryT"][f];rhoW=d["p"][c]/(R*Tw);us=cmu**.25*np.sqrt(d["k"][c])
        yp=us*d["sstWallDistance"][c]*rhoW/mu;visc=yp<yt
        tp=Pr*yp if visc else Prt*(np.log(E*yp)/kappa+P)
        uc=0 if visc else us/kappa*np.log(E*yt)-uw
        C=.5*rhoW*us*(Pr*u*u if visc else Prt*u*u+(Pr-Prt)*uc*uc)
        conductivity=max(mu*cp/Pr,cp*rhoW*us*d["sstWallDistance"][c]/tp)
        conduction=conductivity*(Tw-d["Tgas"][c])*d["deltaCoeffs"][f]
        records.append(dict(dataset=label,face=int(f),owner=c,x=d["Cx"][c],y=d["Cy"][c],z=d["Cz"][c],
            Ux=vel[0],Uy=vel[1],Uz=vel[2],T=d["Tgas"][c],p=d["p"][c],rho=d["rho"][c],
            k=d["k"][c],omega=d["omega"][c],nut=d["nut"][c],wallT=Tw,wallUx=wall[0],wallUy=wall[1],wallUz=wall[2],
            rhoWall=rhoW,Cp=cp,Cv=cp-R,R=R,mu=mu,Pr=Pr,Prt=Prt,conductivityLaminar=mu*cp/Pr,
            conductivityEffective=conductivity,yPlus=yp,conduction=conduction,JayatillekeCorrection=-C/tp,netHeat=conduction-C/tp,
            e_UGKP_from_temperature=(cp-R)*d["Tgas"][c],
            e_OF10_from_temperature=of_constants["Cv_J_per_kg_K"]*d["Tgas"][c]+of_constants["energy_offset_J_per_kg"]))
    with (out/(label+"-wall-decomposition.csv")).open("w") as f:
        writer=csv.DictWriter(f,fieldnames=records[0],lineterminator="\n");writer.writeheader();writer.writerows(records)
    return [min(records,key=lambda r:abs(r["x"]-x)) for x in [.35,.38,.58,.75,.85]]

def native_stations(datasets):
    records=[]
    constants=json.loads((B/"provenance/of10-runtime-constants.json").read_text())
    for label,case,time_value in [("of10-ny80","of10-ny80",1282),("of10-ny160","of10-ny160",2887),
                                  ("ugkp-initial","ugkp-wallFunction",.02305),("ugkp-latest250us","ugkp-wallFunction",.0233)]:
        path=datasets[label];n=int((path/"counts.txt").read_text().split()[0])
        directory=B/"cases"/case
        tp=next(p for p in directory.iterdir() if p.is_dir() and re.fullmatch("[0-9.eE+-]+",p.name) and abs(float(p.name)-time_value)<1e-12)
        data={name:field(tp/name,n,name=="U") for name in ["rho","U","T","p","k","omega","nut"]}
        x=read_array(path,"Cx");y=read_array(path,"Cy")
        owner=read_array(path,"faceOwner",True);kind=read_array(path,"riemannBoundaryKind",True)
        for station in [.35,.38,.58,.75,.85]:
            c=int(np.argmin(abs(x-station)+100*y));f=int(np.flatnonzero((owner==c)&(kind==2))[0])
            Rnative=constants["R_J_per_kg_K"] if label.startswith("of10") else 8.31446261815324/.028965206821645145
            offset=constants["energy_offset_J_per_kg"] if label.startswith("of10") else 0
            records.append(dict(dataset=label,time_or_iteration=time_value,owner=c,wall_face=f,x=x[c],y=y[c],
                rho_native=data["rho"][c],rho_budget=read_array(path,"rho")[c],p=data["p"][c],T=data["T"][c],
                Ux=data["U"][c,0],Uy=data["U"][c,1],Uz=data["U"][c,2],k=data["k"][c],omega=data["omega"][c],nut=data["nut"][c],
                Cp=1004.675,R_native=Rnative,Cv_native=1004.675-Rnative,e_native_reconstructed=(1004.675-Rnative)*data["T"][c]+offset,
                wall_temperature=read_array(path,"riemannBoundaryT")[f],wallUx=read_array(path,"riemannBoundaryUx")[f],
                wallUy=read_array(path,"riemannBoundaryUy")[f],wallUz=read_array(path,"riemannBoundaryUz")[f]))
    with (out/"native-case-stations.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=records[0],lineterminator="\n");writer.writeheader();writer.writerows(records)
    return records

try:
    latest=export_latest()
    datasets={"of10-ny80":B/"data/of10-ny80","of10-ny160":B/"data/of10-ny160",
              "ugkp-initial":B/"data/ugkp-initial","ugkp-latest250us":latest}
    law_binary=out/"thermal-laws"
    run(["g++","-std=c++17","-O2","-I"+str(source),"-I"+str(source/"common"),B/"audit/thermal_laws.cpp","-o",law_binary],label="build-thermal-laws")
    for name in ["of10-ny80","of10-ny160"]:
        run([law_binary,B/"cases"/name/"matched-wall-last.csv",out/(name+"-two-laws.csv")],label=name+"-two-laws")
    summaries=[]
    for variant in ["before","after"]:
        binary=out/("budget-"+variant)
        run(["g++","-std=c++17","-O2","-I"+str(B/"audit"/(variant+"-include")),
             "-I"+str(source),"-I"+str(source/"common"),"-I"+str(source/"common/gasNumerics"),
             "-I"+str(source/"validation/tests/gas_transport"),B/"audit/thermal-budget.cpp","-o",binary],
             cwd=source,label="build-budget-"+variant)
        for name,path in datasets.items():
            target=out/(variant+"-"+name);target.mkdir(exist_ok=True)
            run([binary,path,target,".003","normal","4","1","1"],label=variant+"-"+name)
            cells=rows(target/"detailed-cells.csv");faces={int(f["face"]):f for f in rows(target/"detailed-faces.csv")}
            for cell in cells:
                c=int(cell["cell"]);adj=[f for f in faces.values() if int(f["owner"])==c or int(f["neighbour"])==c]
                assert len(adj)==6,(name,c,len(adj))
                for flux,rhs in [("mass","massRhs"),("momentumX","momentumRhsX"),("momentumY","momentumRhsY"),("momentumZ","momentumRhsZ"),("energy","energyRhs")]:
                    measured=sum((-1 if int(f["owner"])==c else 1)*float(f[flux]) for f in adj)/float(cell["volume"])
                    assert np.isclose(measured,float(cell[rhs]),rtol=1e-9,atol=1e-5),(name,c,flux,measured,cell[rhs])
                for f in adj:
                    assert np.isclose(float(f["energy"]),sum(float(f[k]) for k in ["energyAdvection","pressureWork","viscousWork","conductionAndWallCorrection"]),rtol=1e-12,atol=1e-12)
                summaries.append({"variant":variant,"dataset":name,"cell":cell})
    station=min([v for v in summaries if v["variant"]=="after" and v["dataset"]=="of10-ny80"],key=lambda v:abs(float(v["cell"]["x"])-.58))
    historical=json.loads((B/"audit/original/coefficient-frozen-comparison.json").read_text())
    expected=next(r for r in historical if r["dataset"]=="wallfunction-of10-matched-80")["variants"]["after"]["cell"]
    for key in ["Tdot","TdotCellStressWork"]:
        actual=float(station["cell"][key]);ref=float(expected[key])
        assert np.isclose(actual,ref,rtol=1e-8,atol=1e-5),(key,actual,ref)
    decompositions={name:wall_decomposition(name,path) for name,path in datasets.items()}
    native=native_stations(datasets)
    result={"exit_code":0,"reference_frozen_Ny80_station":station,
            "historical_expected_Tdot":expected["Tdot"],"historical_expected_TdotCellStressWork":expected["TdotCellStressWork"],
            "all_station_budgets":summaries,"five_station_wall_decompositions":decompositions,"native_case_stations":native,
            "energy_budget_interpretation":"UGKP production spatial operator evaluated on frozen fields, not OF10's discrete residual. OF-style work is a diagnostic replacement only.",
            "face_flux_units":"Integrated flux, owner-outward; mass kg/s, momentum N, energy W. Cell RHS is negative oriented divergence divided by volume.",
            "wall_heat_sign":"Positive into gas; Jayatilleke correction is negative in these cases.",
            "gas_volume_sources":"Zero mass/momentum/total-energy sources in this inert stationary case. SST source terms are separately available in macro-cell-budget.csv; k is not part of total energy."}
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":"PASS","Tdot":station["cell"]["Tdot"],"TdotCellStressWork":station["cell"]["TdotCellStressWork"],
                      "wall_reference_record":next(r for r in rows(out/"of10-ny80-two-laws.csv") if r["owner"]=="7782")},indent=2),flush=True)
finally:
    (out/"commands.json").write_text(json.dumps(commands,indent=2)+"\n")
