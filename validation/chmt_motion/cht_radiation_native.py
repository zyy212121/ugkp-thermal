# Actual CHT executable: nonzero particle radiation, carry, and disk restart.
import argparse,json,sys,subprocess,re,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'validation/molecular_transport'))
import native_matrix as m
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--solver',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
codec=out/'particles';subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT),str(ROOT/'validation/chmt_motion/cht_radiation_particles.cpp'),'-o',str(codec)],check=True)
meta=json.loads((ROOT/'validation/molecular_transport/fixtures.json').read_text())['cht-contact']
results=[]
def particles(path):
 return [list(map(float,line.split())) for line in subprocess.check_output([codec,'read',path],text=True).splitlines()]
for name,t0,t1,wall in [('above',4500,4500,4500),('below',500,500,500),('heterogeneous',1300,2700,4000)]:
 case=out/name;m.prepare(ROOT/'validation/molecular_transport/fixtures/cht-contact',case,meta)
 # Isolate radiation from gas conduction; all particle drag/heat sources are off.
 transport=case/'constant/gasTransportProperties';transport.write_text(re.sub(r'muRef\s+[^;]+;', 'muRef 1e-30;',transport.read_text()))
 eps=case/'0/epsilonS';eps.write_text(eps.read_text().replace('internalField uniform 0;', 'internalField nonuniform List<scalar> 8('+str(1/(2800*.125))+' 0 0 0 0 0 0 0);'))
 schedule=case/'constant/schedulingProperties';m.entry(schedule,'gpuResidentPureGasOnly','false');m.entry(schedule,'gpuResidentParticleCapacity',16);m.entry(schedule,'gpuCsrLevel','L1')
 props=case/'constant/particleProperties';props.write_text(props.read_text()+'\nTpMin 100; TpMax 5000; sommerfeldThreshold false; gpuResidentInjectionTheta 0;\n')
 solid=case/'constant/solidRegionProperties';solid.write_text(solid.read_text().replace('radiationEnabled false','radiationEnabled true').replace('emissivity {type constant;value 0;}','emissivity {type constant;value 1;}').replace('kappa {type constant;value 10;}','kappa {type constant;value 1e8;}'))
 for f in [case/'0/T',case/'0/solid/T']:
  f.write_text(re.sub(r'uniform (?:300|600)\b',f'uniform {wall}',f.read_text()))
 t=(case/'0/T').read_text();t=re.sub(r'internalField uniform [^;]+;',f'internalField uniform {t0};',t).replace('object T;','object Tp;');(case/'0/Tp').write_text(t)
 pf=case/'0/p';pf.write_text(pf.read_text().replace('178167.05610328368',str(wall*8314.46261815324/28)))
 (case/'constant/radiationProperties').write_text('''FoamFile {version 2.0; format ascii; class dictionary; object radiationProperties;}
schemaVersion 1; enabled true; couplingFrequency 1;
model {mieTable "constant/mie.table"; do {nTheta 2; nPhi 4;
IBoundaryCondition diffuseGrey; radiationActiveEpsilonMin 0;
maxSourceIterations 300; sourceRelativeTolerance 1e-13;
linearSolverTolerance 1e-13; linearSolverRelTol 0; linearSolverMaxIterations 200;
maxPhaseBalanceIterations 200; phaseBalanceTolerance 1e-12; conservationRelativeTolerance 1e-8;
maxPhaseScalingCorrection 100; angleParallelThreads 1;
boundaryRadiation {default {type diffuseGrey; emissivity 1; T '''+str(wall)+''';}}}}
''')
 (case/'constant/mie.table').write_text('Tvals 1000 4000\nDvals .001 .002\nMuvals -1 1\nPlanckBandFractions 1 1\nDATA\n'+''.join(f'NODE {d} {t} .6666666666666667 0 .6666666666666667\nPHI 1 1\n' for d in range(2) for t in range(2))+'END_MIE_TABLE\n')
 carry=(.03125,-.0625) if name=='heterogeneous' else (0,0)
 subprocess.run(list(map(str,[codec,'create',case/'0/gpuResidentStrictParticles.dat',t0,t1,*carry])),check=True)
 m.run([a.solver.resolve(),'-case',case],case/'run.log')
 checkpoints=m.times(case);assert len(checkpoints)>=2
 final=particles(checkpoints[-1]/'gpuResidentStrictParticles.dat');assert len(final)==2
 assert re.search(r'particleRadiationApplied\s+true;', (checkpoints[-1]/'thermalExchangeState').read_text())
 if name!='heterogeneous':assert max(abs(row[1]-wall) for row in final)<1e-8,final
 else:assert any(abs(row[1]-t)>1e-8 for row,t in zip(final,[t0,t1])),final
 restart=out/(name+'-restart');m.copy_input(case,restart);saved=checkpoints[0];shutil.copytree(saved,restart/saved.name);m.entry(restart/'system/controlDict','startTime',saved.name)
 m.run([a.solver.resolve(),'-case',restart],restart/'run.log')
 resumed=particles(m.times(restart)[-1]/'gpuResidentStrictParticles.dat')
 error=max(abs(x-y) for left,right in zip(final,resumed) for x,y in zip(left,right));assert error<1e-8,(final,resumed)
 results.append(dict(case=name,final=final,restart=resumed,maximum_particle_difference=error))
 (out/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(name,'PASS',flush=True)
