# Three real moving finite-contact models, gas-only ODE, DO radiation and restart.
import sys,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'validation/pr31'))
import chmt_moving_particles_native as existing
free_particles='--free-particles' in sys.argv
if free_particles:sys.argv.remove('--free-particles')
# Geometry assertions cover attached parcels; the checkpoint comparator still
# covers every free parcel and every thermal state.
existing.PROBE=existing.PROBE.replace('for(size_t i=0;i<state.particles.size();++i){const auto& p=state.particles[i];if(i)std::cout',
 'bool comma=false;for(size_t i=0;i<state.particles.size();++i){const auto& p=state.particles[i];if(!p.state)continue;if(comma)std::cout').replace("<<',';\n  const auto center", "<<',';comma=true;\n  const auto center")
prepare_original=existing.m.prepare
run_original=existing.m.run
def prepare(*args,**kwargs):
 result=prepare_original(*args,**kwargs);case=Path(args[1]);p=case/'constant/chmtProperties'
 p.write_text(re.sub(r'\bemissivity\s+[^;]+;', 'emissivity 0.9;', p.read_text().replace('model reactingSst;','model equilibriumOde;'))+'\ncouplingSchedule solidThermalCoupling;\n')
 header=lambda name:'FoamFile {version 2.0; format ascii; class dictionary; object '+name+';}\n'
 (case/'constant/solidRegionProperties').write_text(header('solidRegionProperties')+'''solidThermalCoupling {couplingInterval 1e-6; solidContactCouplingInterval 1e-6; radiationEnabled true; radiationCouplingFrequency 1;}\n''')
 (case/'constant/particleProperties').write_text(header('particleProperties')+'particleCp 1542; particleRho 2500;\n')
 (case/'constant/radiationProperties').write_text(header('radiationProperties')+'''enabled true;model {mieTable "constant/mie.table";do {
 nTheta 2;nPhi 8;maxSourceIterations 1000;sourceRelativeTolerance 1e-10;phaseBalanceTolerance 1e-10;
 maxPhaseBalanceIterations 100;radiationActiveEpsilonMin 1e-12;conservationRelativeTolerance 1e-8;
 maxPhaseScalingCorrection 100;angleParallelThreads 2;
 boundaryRadiation {default {type diffuseGrey;emissivity 0.9;T 800;}}
 }}\n''')
 if free_particles:
  rad=case/'constant/radiationProperties';rad.write_text(rad.read_text().replace('sourceRelativeTolerance 1e-10','sourceRelativeTolerance 1e-13'))
 table='Tvals 1 10000\nDvals 1e-6 .01\nMuvals -1 1\nPlanckBandFractions 1 1\nDATA\n'
 for d in (1e-6,.01):
  for t in (1,10000):table+=f'NODE {d} {t} .3 .7 1\nPHI 1 1\n'
 (case/'constant/mie.table').write_text(table+'END_MIE_TABLE\n')
 return result
def run(command,log,*args,**kwargs):
 if free_particles and str(command[0]).endswith(('CHMT','CHMT.next')):
  case=Path(command[command.index('-case')+1]);p=case/'constant/chmtProperties';text=p.read_text()
  if 'mobileRadiation' not in text:
   text=text.replace('particles {attached {', 'particles {mobileRadiation {id "43"; rng "7654321"; cell 0; position (0.25 0.005 0.005); velocity (0 0 0); mass 1e-8; diameter .001; temperature 1000;} attached {')
   p.write_text(text)
 elapsed=run_original(command,log,*args,**kwargs)
 if str(command[0]).endswith(('CHMT','CHMT.next')):
  text=Path(log).read_text();rows=re.findall(r'CHMT_RADIATION .*?particleHeat=([^ ]+) materialHeat=([^ ]+) externalHeat=([^ ]+) residual=([^\n]+)',text)
  assert rows,'radiation did not run'
  for row in rows:
   particle,wall,external,residual=map(float,row);assert abs(residual)<1e-12+1e-9*(abs(particle)+abs(wall)+abs(external)),row
  assert any(abs(float(row[1]))>1e-16 for row in rows),'trivial zero wall heat'
  if free_particles:assert any(abs(float(row[0]))>1e-16 for row in rows),'trivial zero free-particle radiation heat'
 return elapsed
existing.m.prepare=prepare
existing.m.run=run
if __name__=='__main__':existing.main()
