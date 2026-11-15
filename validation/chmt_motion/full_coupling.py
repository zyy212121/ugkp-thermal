# Three real moving finite-contact models, gas-only ODE, DO radiation and restart.
import sys,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'validation/pr31'))
import chmt_moving_particles_native as existing
free_particles='--free-particles' in sys.argv
if free_particles:sys.argv.remove('--free-particles')
csr_auto='--csr-auto' in sys.argv
if csr_auto:sys.argv.remove('--csr-auto')
# Geometry assertions cover attached parcels; the checkpoint comparator still
# covers every free parcel and every thermal state.
existing.PROBE=existing.PROBE.replace('for(size_t i=0;i<state.particles.size();++i){const auto& p=state.particles[i];if(i)std::cout',
 'bool comma=false;for(size_t i=0;i<state.particles.size();++i){const auto& p=state.particles[i];if(!p.state)continue;if(comma)std::cout').replace("<<',';\n  const auto center", "<<',';comma=true;\n  const auto center")
prepare_original=existing.m.prepare
run_original=existing.m.run
def prepare(*args,**kwargs):
 result=prepare_original(*args,**kwargs);case=Path(args[1])
 # Cp=1000 J/kg/K, condensed mass density1000: material500K vs gas700K.
 material_energy=case/'0/solid/solidEnergyDensity'
 material_energy.write_text(re.sub(r'internalField uniform [^;]+;', 'internalField uniform 500000000;', material_energy.read_text()))
 momentum=case/'constant/momentumTransport'
 momentum.write_text(momentum.read_text().replace('model reactingSst;','model equilibriumOde;'))
 # Keep the shared solid-region and interface mapping intact. Particle input is
 # written by the parent native contact fixture after this prepare callback.
 scheme=case/'constant/scheme'
 existing.set_input(scheme,'couplingInterval',1e-6)
 # The case explicitly carried FP64 wall tolerances. Choose native FP32
 # case numerics rather than changing the shared wall-layer method.
 bits=int(sys.argv[sys.argv.index('--precision')+1]) if '--precision' in sys.argv else 64
 if bits==32:
  existing.set_input(scheme,'relativeTolerance',1e-4)
  existing.set_input(scheme,'absoluteTolerance',9.5367431640625e-7)

 scheme.write_text(scheme.read_text()+"""
radiation { do {
 maxPhaseBalanceIterations 100;
 linearSolverMaxIterations 200;
} }
""")
 header=lambda name:'FoamFile {version 2.0; format ascii; class dictionary; object '+name+';}\n'
 (case/'constant/radiationProperties').write_text(header('radiationProperties')+'''enabled true;
model {mieTable "constant/mie.table";do {
 nTheta 2;nPhi 8;
 boundaryRadiation {default {type diffuseGrey;emissivity 0.9;T 800;}}
 }}\n''')
 table='Tvals 1 10000\nDvals 1e-6 .01\nMuvals -1 1\nPlanckBandFractions 1 1\nDATA\n'
 for d in range(2):
  for t in range(2):table+=f'NODE {d} {t} .3 .7 1\nPHI 1 1\n'
 (case/'constant/mie.table').write_text(table+'END_MIE_TABLE\n')
 return result
def run(command,log,*args,**kwargs):
 solver_run=Path(log).name in ('run.log','restart.log') and '-case' in command
 if free_particles and solver_run:
  case=Path(command[command.index('-case')+1]);p=case/'constant/particleProperties';text=p.read_text()
  if 'mobileRadiation' not in text:
   text=text.replace('particles {attached {', 'particles {mobileRadiation {id "43"; rng "7654321"; cell 0; position (0.25 0.005 0.005); velocity (0 0 0); mass 1e-8; diameter .001; temperature 1000;} attached {')
   p.write_text(text)
 if csr_auto and solver_run:
  case=Path(command[command.index('-case')+1]);p=case/'constant/schedulingProperties'
  existing.set_input(p,'gpuCsrLevel','auto')
  existing.set_input(p,'gpuCsrHeavyReductionAutoInterval',1)
 elapsed=run_original(command,log,*args,**kwargs)
 if solver_run:
  if Path(log).name=="restart.log":assert "DOM warm-start cache restored:" in Path(log).read_text(),"DOM restart did not restore warm seed"
  # Check persisted physical DOM state, not the removed balance-print line.
  import struct,math,json
  text=Path(log).read_text()
  case=Path(command[command.index('-case')+1])
  initialized=[]
  bits=int(sys.argv[sys.argv.index('--precision')+1]) if '--precision' in sys.argv else 64
  for cache in sorted((case/'chmtOutput').glob('checkpoint-*/domWarmStart.bin')):
   data=cache.read_bytes()
   assert len(data)>=56,('incomplete DOM checkpoint',str(cache))
   magic,version,width,cells,angles,layout,ready=struct.unpack_from('<7Q',data)
   assert magic==0x55474b50444f4d31 and version==2 and width==bits//8,(str(cache),magic,version,width)
   assert ready in (0,1),(str(cache),ready)
   if not ready:
    assert len(data)==56,('uninitialized DOM payload',str(cache))
    continue
   assert cells>0 and angles>0 and len(data)==56+cells*angles*width,('DOM payload shape',str(cache))
   values=struct.unpack_from('<'+str(cells*angles)+('f' if width==4 else 'd'),data,56)
   assert all(math.isfinite(v) and v>=0 for v in values),('invalid DOM intensity',str(cache))
   assert any(v>0 for v in values),('trivial DOM intensity',str(cache))
   initialized.append(dict(path=str(cache),cells=cells,angles=angles,precision=bits))
  assert initialized,'no initialized DOM checkpoint after successful solver execution'
  # Native device diagnostics report final source/material/transport failure at
  # 34001, phase failure at 34002 and true inner-linear failure at 34003.
  # Do not substitute exit status or a finite intensity field for convergence.
  failure_rows=[line for line in text.splitlines() if re.search(r'\bsite=3400[0-3]\b',line)
       or re.search(r'DO (?:source|phase balance|linear) iteration budget exhausted',line)]
  assert not failure_rows,('native radiation convergence/failure diagnostic',failure_rows)
  Path(str(log)+'.radiation-state-check.json').write_text(json.dumps(dict(
      initializedCheckpoints=initialized,nativeRadiationFailureDiagnostics=failure_rows,
      convergenceEvidence='successful solver completion plus initialized finite DOM state and no native final residual/budget diagnostics; no numeric residual inferred'),indent=2))
 return elapsed
existing.m.prepare=prepare
existing.m.run=run
if __name__=='__main__':existing.main()
