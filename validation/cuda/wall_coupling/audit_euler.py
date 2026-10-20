#!/usr/bin/env python3
"""One Euler step: compare conservative inventory changes with actual device face/source readback."""
import argparse,json
from pathlib import Path
import numpy as np
from compare_case import field
ap=argparse.ArgumentParser();ap.add_argument('--case',type=Path,required=True);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--dt',type=float,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
f=lambda n:np.fromfile(a.input/(n+'.bin'))
i=lambda n:np.fromfile(a.input/(n+'.bin'),dtype=np.int32)
rho=f('rho');nc=len(rho);own=i('faceOwner');nei=i('faceNeighbour');nf=len(own);vol=f('V')
t=max((x for x in a.case.iterdir() if x.is_dir() and x.name[0].isdigit()),key=lambda x:float(x.name))
nrho=field(t/'rho');u=field(t/'U',True)
q=np.array([rho,f('rhoUx'),f('rhoUy'),f('rhoUz'),f('rhoE'),rho*f('k'),rho*f('omega'),f('speciesRhoAir'),f('speciesRhoCO2')])
nq=np.array([nrho,*[nrho*u[:,j] for j in range(3)],field(t/'rhoE'),nrho*field(t/'k'),nrho*field(t/'omega'),nrho*field(t/'Y_Air'),nrho*field(t/'Y_CO2')])
flux=np.fromfile(a.audit/'stage-flux.bin').reshape(9,nf);src=np.fromfile(a.audit/'stage-sources.bin').reshape(2,nc)
rhs=np.zeros_like(q);mask=nei>=0
for j in range(9):
    np.add.at(rhs[j],own,-flux[j]);np.add.at(rhs[j],nei[mask],flux[j,mask])
rhs[5:7]+=src*vol
actual=(nq-q)*vol; predicted=a.dt*rhs;out={}
for j,name in enumerate(['mass','momentumX','momentumY','momentumZ','energy','rhoK','rhoOmega','speciesAir','speciesCO2']):
    valid=np.ones(nc,dtype=bool)
    if j==6:
        import csv
        for row in csv.DictReader((a.audit/'wall-work.csv').open()):valid[int(row['owner'])]=False
    e=(actual-predicted)[j,valid]
    out[name]={'max_cell_inventory_defect':float(abs(e).max()),'sum_inventory_defect':float(e.sum()),'sum_actual_change':float(actual[j,valid].sum()),'sum_flux_and_source_change':float(predicted[j,valid].sum()),'owner_omega_excluded':j==6}
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
