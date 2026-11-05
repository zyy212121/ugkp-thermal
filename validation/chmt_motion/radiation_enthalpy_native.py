# Actual common CUDA owner: extensive radiation heat and nonlinear enthalpy.
import argparse,sys,subprocess,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'validation/pr31'))
from mixture_thermal_native import source
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
s=source()
s=s.replace('CHECK(advanceResidentThermalParticles(s,dt,dt)==0);',r'''
 const ugkwp::ParticleCaloric law;
 const double oldH=law.enthalpy(readOne(s->pT))+readOne(s->pThermalEnthalpyResidual);
 const double mass=readOne(s->pm);
 const double requested=scenario==0?0.:scenario==1?-.2:scenario==2?.2:-1e9;
 double accepted=0;
 CHECK(ugkwpGpuResidentStrictApplyParticleRadiationEnergy(s,1,&requested,&accepted)==0);
 CHECK(ugkwpGpuResidentStrictRefreshParticleEnthalpyPacked(s)==0);
 const double newH=law.enthalpy(readOne(s->pT))+readOne(s->pThermalEnthalpyResidual);
 CHECK(std::abs(mass*(newH-oldH)-accepted)<1e-10);
 if(scenario==1)CHECK(std::abs(accepted-requested)<1e-13);
 if(scenario==0||scenario==2)CHECK(accepted==0);
 if(scenario==3)CHECK(std::abs(newH-law.enthalpy(s->TpMin))<1e-8);
''')
cu=out/'radiation.cu';cu.write_text(s);exe=out/'radiation'
command=['/usr/local/cuda/bin/nvcc','-std=c++17','-O2','--fmad=false','--expt-relaxed-constexpr','--Ofast-compile=min','-arch=sm_89',
 '-I'+str(ROOT/'common/thermal'),'-I'+str(ROOT/'common'),str(cu),str(ROOT/'common/thermal/GpuWallEnergy64.cu'),'-o',str(exe)]
with (out/'build.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
for scenario in range(4):
 for csr in (0,1):
  result=subprocess.run([str(exe),str(scenario),str(csr),'1'],capture_output=True,text=True)
  (out/f'run-{scenario}-{csr}.log').write_text(result.stdout+result.stderr);result.check_returncode()
with (out/'memcheck.log').open('w') as log:
 subprocess.run(['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--leak-check','full','--error-exitcode','88',str(exe),'3','1','1'],stdout=log,stderr=subprocess.STDOUT,check=True)
print('RADIATION_ENTHALPY_PASS runs=8')
