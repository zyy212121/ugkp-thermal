# Three real moving finite-contact models, gas-only ODE, DO radiation and restart.
import sys,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'validation/pr31'))
import chmt_moving_particles_native as existing
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
 table='Tvals 1 10000\nDvals 1e-6 .01\nMuvals -1 1\nPlanckBandFractions 1 1\nDATA\n'
 for d in (1e-6,.01):
  for t in (1,10000):table+=f'NODE {d} {t} .3 .7 1\nPHI 1 1\n'
 (case/'constant/mie.table').write_text(table+'END_MIE_TABLE\n')
 return result
def run(command,log,*args,**kwargs):
 elapsed=run_original(command,log,*args,**kwargs)
 if str(command[0]).endswith(('CHMT','CHMT.next')):
  text=Path(log).read_text();rows=re.findall(r'CHMT_RADIATION .*?particleHeat=([^ ]+) materialHeat=([^ ]+) externalHeat=([^ ]+) residual=([^\n]+)',text)
  assert rows,'radiation did not run'
  for row in rows:
   particle,wall,external,residual=map(float,row);assert abs(residual)<1e-12+1e-9*(abs(particle)+abs(wall)+abs(external)),row
  assert any(abs(float(row[1]))>1e-16 for row in rows),'trivial zero wall heat'
 return elapsed
existing.m.prepare=prepare
existing.m.run=run
if __name__=='__main__':existing.main()
