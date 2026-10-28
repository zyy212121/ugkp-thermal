#!/usr/bin/env python3
"""Materialize the reference CFG as an actual gasUGKPSteady OpenFOAM case.

Run: python3 validation/steady_cpu/of_cases.py cases/reservoir.cfg /tmp/of-case
Then: blockMesh -case /tmp/of-case; gasUGKPSteady -case /tmp/of-case
Generated fields are initial conditions, never expected solution data.
"""
import argparse
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent

def inputs(path):
    source = (HERE/'gas_reference.cpp').read_text().split('struct Input {',1)[1].split('};',1)[0]
    values = {k:float(v) for k,v in re.findall(r'\b(\w+)\s*=\s*([-+\d.eE]+)',source)}
    for line in Path(path).read_text().splitlines():
        words=line.split('#',1)[0].split()
        if words: values[words[0]]=float(words[1])
    return values

def header(name, cls='dictionary'):
    return f'FoamFile {{ version 2.0; format ascii; class {cls}; object {name}; }}\n'

def materialize(cfg, destination, particles=False, foam_universal_gas_constant=8314.46261815324):
    a=inputs(cfg); dest=Path(destination); n=int(a['cells']); length=a['length']; dx=length/n
    for folder in ('0','constant','system'): (dest/folder).mkdir(parents=True,exist_ok=True)
    def write(folder,name,body): (dest/folder/name).write_text(header(name)+body+'\n')
    write('system','blockMeshDict',f'''convertToMeters 1;
vertices ((0 0 0) ({length} 0 0) ({length} 1 0) (0 1 0) (0 0 1) ({length} 0 1) ({length} 1 1) (0 1 1));
blocks (hex (0 1 2 3 4 5 6 7) ({n} 1 1) simpleGrading (1 1 1));
edges ();
boundary (
left {{ type patch; faces ((0 4 7 3)); }}
right {{ type patch; faces ((1 2 6 5)); }}
sides {{ type empty; faces ((0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); }}
);
mergePatchPairs ();''')
    write('system','controlDict',f'''application gasUGKPSteady;
startFrom startTime; startTime 0; stopAt endTime; endTime {int(a['iterations'])};
deltaT 1; writeControl timeStep; writeInterval {int(a['outputEvery'])}; purgeWrite 0;
writeFormat ascii; writePrecision 17; writeCompression off; timeFormat general; timePrecision 17;
runTimeModifiable false; adjustTimeStep false;
steadyGas {{ maxCo {a['courant']}; maxDeltaT {a['maxDt']}; boundaryTime 0;
relativeTolerance 0; absoluteTolerance 0; minimumIterations {int(a['iterations'])+1}; residualInterval {int(a['outputEvery'])};
samplingDeltaT {a['dt']}; windowSamples {int(a.get('windowSamples',a['outputEvery']))};
minimumWindows {int(a['iterations'])+1}; stableWindows 2; meanTolerance 0; inventoryTolerance 0; }}''')
    schemes={1:'Tadmor',2:'Kurganov',3:'HLLE',4:'HLLC',5:'Roe',6:'HLLEM',7:'HLLC-ADC',8:'SLAU2',9:'SLAU2.2'}
    reconstruction=int(a['reconstruction']); divu='MUSCL' if reconstruction==1 else 'upwind'; dive={0:'upwind',1:'MUSCL',2:'limitedLinear 1'}[reconstruction]
    write('system','fvSchemes',f'''fluxScheme {schemes[int(a['scheme'])]}; gasLimiter {['none','barthJespersen','venkatakrishnan'][int(a['limiter'])]};
ddtSchemes {{ default Euler; }} gradSchemes {{ default Gauss linear; }}
divSchemes {{ default none; div(phi,U) Gauss {divu}; div(phi,e) Gauss {dive}; div(phi,K) Gauss {dive}; div(phi,(p|rho)) Gauss {dive}; div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear; }}
laplacianSchemes {{ default Gauss linear corrected; }} interpolationSchemes {{ default linear; }} snGradSchemes {{ default corrected; }}
wallDist {{ method meshWave; }}''')
    write('system','fvSolution','solvers {}\nrelaxationFactors {}\nUGKP { robustFallback true; rhoMin 1e-12; TMin 1e-9; }')
    turb='simulationType RAS; RAS { model kOmegaSST; turbulence on; printCoeffs on; kOmegaSSTCoeffs { wallTreatment lowRe; } }' if a['sst'] else 'simulationType laminar;'
    write('constant','fluidProperties',f'''schemaVersion 1;
thermoType {{ type hePsiThermo; mixture pureMixture; transport const; thermo hConst; equationOfState perfectGas; specie specie; energy sensibleInternalEnergy; }}
mixture {{ specie {{ molWeight {foam_universal_gas_constant/287:.17g}; }} thermodynamics {{ Cp 1004.5; Hf 0; }} transport {{ mu {a['mu']}; Pr 0.7; }} }}
turbulence {{ {turb} }}''')
    if a['mixture']:
        write('constant','gasModelProperties',f'''schemaVersion 1; gasMode mixtureFrozen; species (A B);
speciesThermo {{ A {{ model linearCp; molarMass 0.028; minTemperature 100; maxTemperature 3000; coefficients (1000 0 -1000000); }} B {{ model linearCp; molarMass 0.028; minTemperature 100; maxTemperature 3000; coefficients (1000 0 2000000); }} }}
diffusion {{ model constant; coefficients ({a.get('diffusivity0',0)} {a.get('diffusivity1',0)}); turbulentSchmidt 0.7; }}''')
    write('constant','schedulingProperties',f'''schemaVersion 1;
gpuResidentPureGasOnly {'false' if particles else 'true'}; gpuResidentDynamicInlet false;
gpuResidentParticleCapacity {int(a['particleCapacity']) if particles else 0}; gpuResidentMaxFaceWalkHops 1000;
gpuResidentCourantUpdateInterval 1; gpuResidentMaxDeltaTGrowth 1;
gpuCsrLevel L0; gpuParticleBlockThreads 128; gpuReductionBlockThreads 128;''')
    write('constant','particleProperties',f'''schemaVersion 1; gpuResidentRandomSeed {int(a['seed'])};
rhoS 1000; dS {a['diameter']}; dMin {a['diameter']}; dMax {a['diameter']}; dSigma 0;
parcelMass {a['parcelMass']}; injectionParcelMass {a['parcelMass']};
epsSMin 1e-12; thetaMin 1e-12; TgasMin 1e-9;
dragModel SchillerNaumann; particleGasHeatTransferModel RanzMarshall;
particleTemperatureTransport {'true' if particles else 'false'}; particleRho 1000; particleCp 1000; particleTMin 1; particleTMax 10000;
gpuResidentInjectionTheta {a['particleTheta']}; gpuResidentInjectionTp {a['particleTemperature']};
gpuResidentCollisionalPressure false; gpuResidentJammingPressure false;
{"particleWallCoeffs { default 1; left 0; right 0; }" if particles else ""}''')
    def field(name,dimensions,vals,left,right=None,vector=False,left_type='fixedValue',right_type='fixedValue'):
        def fmt(v): return f'({v:.17g} 0 0)' if vector else f'{v:.17g}'
        body=header(name,'volVectorField' if vector else 'volScalarField')+f'dimensions [{dimensions}];\ninternalField nonuniform List<{"vector" if vector else "scalar"}>\n{n}\n(\n'+'\n'.join(map(fmt,vals))+');\nboundaryField\n{\n'
        for patch,value,kind in [('left',left,left_type),('right',left if right is None else right,right_type)]: body+=f'{patch} {{ type {kind}; value uniform {fmt(value)}; }}\n'
        body+='sides { type empty; }\n}\n'; (dest/'0'/name).write_text(body)
    disturb=[a['perturbation']*math.sin(2*math.pi*(c+.5)/n) for c in range(n)]
    rho=[a['rho']*(1+d) for d in disturb]; velocity=[a['velocity']*(1-d) for d in disturb]
    temperature=[700*(1+d) if a['mixture'] else a['pressure']/(a['rho']*287) for d in disturb]
    gas_r=8314.46261815324/28 if a['mixture'] else 287
    pressure=[r*gas_r*t for r,t in zip(rho,temperature)]; reservoir_t=700 if a['mixture'] else a['pressure']/(a['rho']*287)
    composition=[a['y0']+a.get('yPerturbation',0)*math.sin(2*math.pi*(c+.5)/n) for c in range(n)]
    energy=[r*((1000-gas_r)*t+y*(-1e6)+(1-y)*2e6)+.5*r*u*u if a['mixture'] else p/.4+.5*r*u*u for r,t,u,p,y in zip(rho,temperature,velocity,pressure,composition)]
    field('rho','1 -3 0 0 0 0 0',rho,a['rho']); field('U','0 1 -1 0 0 0 0',velocity,a['velocity'],vector=True)
    field('T','0 0 0 1 0 0 0',temperature,reservoir_t); field('p','1 -1 -2 0 0 0 0',pressure,a['rho']*gas_r*reservoir_t)
    field('rhoE','1 -1 -2 0 0 0 0',energy,energy[0],left_type='zeroGradient',right_type='zeroGradient')
    field('epsilonS','0 0 0 0 0 0 0',[0]*n,a['massRate']/(1000*a['velocity']) if particles else 0,right=0,right_type='zeroGradient')
    field('Us','0 1 -1 0 0 0 0',[0]*n,a['velocity'],vector=True,right_type='zeroGradient')
    field('theta','0 2 -2 0 0 0 0',[0]*n,a['particleTheta'],right_type='zeroGradient')
    field('Tp','0 0 0 1 0 0 0',[a['particleTemperature']]*n,a['particleTemperature'],right_type='zeroGradient')
    field('dMeanCell','0 1 0 0 0 0 0',[a['diameter']]*n,a['diameter'],right_type='zeroGradient')
    field('k','0 2 -2 0 0 0 0',[a['k']]*n,a['k']);field('omega','0 0 -1 0 0 0 0',[a['omega']]*n,a['omega'])
    if a['mixture']:
        field('Y_A','0 0 0 0 0 0 0',composition,a['y0']);field('Y_B','0 0 0 0 0 0 0',[1-y for y in composition],1-a['y0'])
    manifest={'foam_universal_gas_constant':foam_universal_gas_constant,'reference_legacy_Rgas':287,'particle_escape_boundaries':['left','right'] if particles else [],'input':a,'particles':particles,'source_cfg':str(Path(cfg).resolve()),'notes':['Initial fields only; no expected solution generated.','OpenFOAM inlet parcels use gas U, not Us.','Output labels are iteration/sample numbers, not physical time.']}
    (dest/'reference-input.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return dest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cfg',type=Path);p.add_argument('destination',type=Path);p.add_argument('--particles',action='store_true');p.add_argument('--foam-universal-gas-constant',type=float,default=8314.46261815324,help='Target OpenFOAM RR in J/(kmol K); obtain from that build to match effective Rgas=287.');a=p.parse_args();print(materialize(a.cfg,a.destination,a.particles,a.foam_universal_gas_constant))
