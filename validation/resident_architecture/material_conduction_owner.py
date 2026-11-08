#!/usr/bin/env python3
"""Pair actual material conduction with its previous production owner.

The probe also recomputes local Cartesian Fourier balances from returned fields.
The generated include contains the unmodified production function.
"""
import argparse,json,os,re,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--baseline-source',type=Path,required=True);p.add_argument('--baseline-build',type=Path,required=True);p.add_argument('--candidate-source',type=Path,required=True);p.add_argument('--candidate-build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
foam=Path(os.environ['WM_PROJECT_DIR']);lib=Path(os.environ['FOAM_LIBBIN']);mpi=os.environ['FOAM_MPI']
results=[]
for kind,source,build in [('baseline',a.baseline_source,a.baseline_build),('candidate',a.candidate_source,a.candidate_build)]:
    out=a.output/kind;out.mkdir();app=source/'applications/CHMT'
    code=(app/'materials/CpuMaterialDriver.C').read_text();code=code[:code.index('// The dry interface owns geometry')]
    (out/'native_conduction.inc').write_text(code)
    flags=['g++','-std=c++17','-O2','-ffp-contract=off','-fopenmp','-ffunction-sections','-fdata-sections','-DNoRepository','-DWM_DP','-DWM_LABEL_SIZE=32','-DUGKWP_GAS_SPECIES=2']
    if 'MaterialConductionWorkspace& workspace' in code:flags+=['-DCHMT_RETAINED_CONDUCTION_WORKSPACE']
    elif 'ugkp::gpu::ImplicitDiffusion& residentDiffusion' in code:flags+=['-DCHMT_RETAINED_DIFFUSION']
    flags+=['-I'+str(x) for x in [out,app,app/'materials',source/'common',foam/'src/OpenFOAM/lnInclude',foam/'src/finiteVolume/lnInclude',foam/'src/meshTools/lnInclude',foam/'src/OSspecific/POSIX/lnInclude']]
    objects=[build/(x+'.o') for x in ['MaterialBackend','SparseConjugateGradient','MaterialThermalBackend','ImplicitDiffusion','ThermalFieldOperators']]
    command=flags+[str(ROOT/'validation/resident_architecture/material_conduction_owner.cpp'),str(app/'mesh/Geometry.C'),str(app/'materials/MaterialTransport.C'),*map(str,objects),'-Wl,--gc-sections','-L'+str(lib),'-lfiniteVolume','-lmeshTools','-lOpenFOAM','-L'+str(lib/mpi),'-lPstream','-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcudart','-o',str(out/'probe')]
    with (out/'build.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    case=out/'case';shutil.copytree(ROOT/'validation/tests/fixtures/conduction_energy/case',case)
    with (out/'run.log').open('w') as log:subprocess.run([str(out/'probe'),'-case',str(case)],stdout=log,stderr=subprocess.STDOUT,check=True)
    cases=[]
    for line in (out/'run.log').read_text().splitlines():
        w=line.split()
        if w[:1]==['CASE']:cases.append(dict(id=int(w[1]),cp1=float(w[2]),dt=float(w[3]),tolerance=float(w[4]),cells=[]))
        elif w[:1]==['MATERIAL_CELL']:cases[-1]['cells'].append(list(map(float,w[2:])))
        elif w[:1]==['CELL_BALANCE']:cases[-1]['balance']=list(map(float,w[1:]))
        elif w[:1]==['RESULT']:cases[-1]['success']=w[1]=='1'
    assert len(cases)==27 and all(c['success'] and len(c['cells'])==8 for c in cases),out
    results.append(cases)
comparisons=[]
for b,c in zip(*results):
    assert b['id']==c['id']
    difference=max(abs(x[0]-y[0]) for x,y in zip(b['cells'],c['cells']))
    inventory=sum(abs(x[1]) for x in c['cells'])
    roundoff=128*2.220446049250313e-16*inventory
    assert difference<2e-7,(b['id'],difference)
    assert c['balance'][0]<=2*b['balance'][0]+roundoff,(b['id'],b['balance'],c['balance'],roundoff)
    comparisons.append(dict(id=b['id'],maxTemperatureDifference=difference,baselineCellBalance=b['balance'],candidateCellBalance=c['balance'],inventoryRoundoff=roundoff))
(a.output/'results.json').write_text(json.dumps(dict(status='PASS',cases=comparisons),indent=2)+'\n')
print('PASS material conduction',len(comparisons),'cases; max temperature',max(c['maxTemperatureDifference'] for c in comparisons))
