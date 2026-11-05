"""Actual CUDA material root -> resident enthalpy publication, including CSR off/on."""
import argparse,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'validation/pr31'))
from mixture_thermal_native import source
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
s='#include "gpu/ParticleRadiationMaterial.H"\n'+source()
s=s.replace('s->particleCapacity=1;','s->particleCapacity=3;').replace('CHECK(readOne(s->particleCountDevice)==1);','CHECK(readOne(s->particleCountDevice)==3);')
s=s.replace('CHECK(advanceResidentThermalParticles(s,dt,dt)==0);',r'''
 {
 using namespace ugkp::gpu;const ugkwp::ParticleCaloric law;
 s->TpMin=1;s->TpMax=5000;
 const int offsets[2]={0,3},number=3,ids[3]={0,0,0},active[3]={1,1,1};
 const unsigned char walls[3]={0,0,1};const double weights[3]={.3,.7,10.};
 CHECK(copyToDevice(s->particleCountDevice,&number,1,"radiation count")==0);
 CHECK(copyToDevice(s->pCellId,ids,3,"radiation cells")==0);
 CHECK(copyToDevice(s->pStatus,active,3,"radiation activity")==0);
 CHECK(copyToDevice(s->pStuck,walls,3,"radiation attachment")==0);
 CHECK(copyToDevice(s->pm,weights,3,"radiation mass")==0);
 CHECK(copyToDevice(s->preBaseCellOffset,offsets,2,"radiation packed offsets")==0);
 s->preBaseDirectoryReady=csr;CHECK(syncDeviceState(s,"radiation state")==0);
 MieOpticalTable table;table.temperature={1,5000};table.diameter={.001,.002};table.mu={-1,1};table.band={1,1};
 const double absorption=1e-8/(4*5.670374419e-8)*2*.001/(3*.1);
 table.absorption.assign(4,absorption);table.scattering.assign(4,0);table.extinction=table.absorption;table.phase.assign(8,1);
 ParticleRadiationMaterial device(table);
 for(int directory=0;directory<(csr?3:1);++directory){
 // Borrow packed offsets, invalidate the directory, then rebuild and reuse it.
 s->preBaseDirectoryReady=csr&&directory!=1;
 if(directory==2)CHECK(copyToDevice(s->preBaseCellOffset,offsets,2,"rebuilt packed offsets")==0);
 CHECK(syncDeviceState(s,"directory transition")==0);
 for(int heterogeneous=0;heterogeneous<2;++heterogeneous)for(double elapsed:{.01,100.,1000.}){
  const double temps[3]={1000.,heterogeneous?2500.:1000.,3500.},carry[3]={heterogeneous?1e-5:0.,heterogeneous?-3e-5:0.,0.};
  CHECK(copyToDevice(s->pT,temps,3,"radiation initial T")==0);
  CHECK(copyToDevice(s->pThermalEnthalpyResidual,carry,3,"radiation initial carry")==0);
  ParticleRadiationMaterialInput input;input.epsilon={.1};input.mass={1.};input.temperature={.3*temps[0]+.7*temps[1]};
  input.diameter={.001};input.incident={4*5.670374419e-8*std::pow(4000.,4)};input.volume={1.};
  input.boundaryArea={1.};input.boundaryFlux={-1e6};input.coupledBoundary={1};
  input.heatFactor=1542;input.elapsed=elapsed;input.minimumTemperature=1;input.maximumTemperature=5000;input.activeEpsilonMinimum=1e-12;
  CHECK(ugkwpGpuResidentStrictParticleRadiationResponse(s,&input.response)==0);
  CHECK(bool(input.response.cellOffsets)==bool(csr&&directory!=1));
  const auto result=device.solve(input);double accepted=0;
  CHECK(ugkwpGpuResidentStrictApplyParticleRadiationEnergy(s,1,result.energy.data(),&accepted)==0);
  double actualHeat=0,mean=0;
  for(int i=0;i<2;++i){const double h=law.enthalpy(readOne(s->pT+i))+readOne(s->pThermalEnthalpyResidual+i);
   actualHeat+=weights[i]*(h-law.enthalpy(temps[i])-carry[i]);mean+=weights[i]*law.temperature(h);}
  const double power=1e-8*(std::pow(4000.,4)-std::pow(mean,4));
  CHECK(mean<=4000.);CHECK(std::abs(mean-result.temperature[0])<1e-8);
  CHECK(std::abs(accepted-result.energy[0])<1e-7);CHECK(std::abs(actualHeat-accepted)<1e-7);
  CHECK(std::abs(accepted-elapsed*power)<1e-8*std::max(1.,std::abs(accepted)));
  CHECK(std::abs(accepted+result.boundaryEnergy[0])<1e-7);
  CHECK(readOne(s->pT+2)==3500.&&readOne(s->pThermalEnthalpyResidual+2)==0.);
  std::fprintf(stderr,"NATIVE_CALORIC csr=%d directory=%d heterogeneous=%d dt=%.17g meanT=%.17g accepted=%.17g residual=%.17g\n",csr,directory,heterogeneous,elapsed,mean,accepted,accepted-elapsed*power);
 }
 }
 // Real two-kernel publication must retain sub-ULP capacity at TpMax.
 s->TpMax=4000;CHECK(syncDeviceState(s,"carry limit")==0);
 const double edge[3]={4000,4000,3500},remainder[3]={-1e-10,-1e-10,0};
 CHECK(copyToDevice(s->pT,edge,3,"limit T")==0);
 CHECK(copyToDevice(s->pThermalEnthalpyResidual,remainder,3,"limit carry")==0);
 const double request=1e-11;double acceptedTiny=0;
 CHECK(ugkwpGpuResidentStrictApplyParticleRadiationEnergy(s,1,&request,&acceptedTiny)==0);
 CHECK(std::abs(acceptedTiny-request)<1e-24);
 CHECK(std::abs(readOne(s->pThermalEnthalpyResidual)+9e-11)<1e-24);
 std::fprintf(stderr,"NATIVE_LIMIT_CARRY csr=%d accepted=%.17g\n",csr,acceptedTiny);
 }
''')
cu=out/'radiation-response.cu';cu.write_text(s);exe=out/'radiation-response'
command=['/usr/local/cuda/bin/nvcc','-std=c++17','-O2','--fmad=false','--expt-relaxed-constexpr','--Ofast-compile=min','-arch=sm_89',
 '-I'+str(ROOT/'common/thermal'),'-I'+str(ROOT/'common'),str(cu),str(ROOT/'common/thermal/GpuWallEnergy64.cu'),
 str(ROOT/'common/gpu/ParticleRadiationMaterial.cu'),str(ROOT/'common/gpu/MieOpticalProperties.cu'),'-o',str(exe)]
with (out/'build.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
for csr in (0,1):
 result=subprocess.run([str(exe),'1',str(csr),'1'],capture_output=True,text=True)
 (out/f'run-{csr}.log').write_text(result.stdout+result.stderr);result.check_returncode()
for csr in (0,1):
 for tool in ('memcheck','racecheck'):
  with (out/f'{tool}-csr{csr}.log').open('w') as log:
   subprocess.run(['/usr/local/cuda/bin/compute-sanitizer','--tool',tool,'--error-exitcode','88',str(exe),'1',str(csr),'1'],stdout=log,stderr=subprocess.STDOUT,check=True)
print('NATIVE_CALORIC_RADIATION_PASS scenarios=24 limit_carry=2 memcheck_and_racecheck_both_csr_modes')
