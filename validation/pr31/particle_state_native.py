"""Actual common CUDA owner: complete particle restart state, bounded staging.

No claim of a coupled CHMT/contact evolution test. Exercises real device arrays,
shared upload/download APIs, all wall states, both profile representations,
independent theta/age/saved velocity, carry, empty and multi-chunk populations.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"validation/tests/gas_transport"))
sys.path.insert(0,str(ROOT))
from test_full_heat_advance import native_cuda_fixture


def source():
    text=native_cuda_fixture()
    text=text.replace('s->particleCapacity=1;', 's->particleCapacity=std::max(1,argc>1?std::atoi(argv[1]):4);')
    text=text.replace('int main(int argc,char**argv){',r'''
__global__ void surfaceResponse(DeviceState* sp,double* out) {
 const auto& s=*sp;const int f=0,i=0;
 const auto q=particleFiniteContactThermalCandidateDevice(s,i,ugkwp::particleWallTemperature(s,f),1e-8,s.pm[i],1e-6,0);
 const double relative=10.-ugkwp::particleWallVelocityX(s,f);
 const auto impact=Foam::gpuThermal::evaluateSommerfeldImpact(s.pT[i],fabs(relative),s.pd[i],s.sommerfeldThreshold);
 out[0]=q.valid;out[1]=q.wallEnergyJ;out[2]=q.particleTemperatureK;out[3]=q.specificEnthalpyResidual;
 out[4]=impact.valid;out[5]=impact.weber;out[6]=impact.reynolds;out[7]=impact.sommerfeld;
}
int main(int argc,char**argv){
''')
    start=text.index(' CHECK(rebuildResidentParticleMomentsFromParticles(s,1)==0);')
    text=text[:start]+r'''
 const unsigned char mask[6]={1,3,4,0,0,0};
 CHECK(ugkwpGpuResidentStrictConfigureParticleStuckModel(s,6,mask,100.,1,1.,1.,1.,1.,90.,0,8,2327.,20.,1.1e6,3950.,1000.,5.,.01,0.)==0);
 std::vector<ugkp::thermal::ParticleState<>> initial(argc>1?std::atoi(argv[1]):4);
 for(int i=0;i<int(initial.size());++i) {
  auto& q=initial[i];q.id=0xffffffffff000000ULL+i;q.rng=0x8765432100000000ULL+i;q.cell=0;
  q.state=i%4;q.face=q.state?q.state-1:-1;q.mass=.002;q.diameter=.001;
  q.position={.1,.2,.3};q.velocity=q.state?ugkp::thermal::ParticleVector{}:ugkp::thermal::ParticleVector{1.,2.,3.};
  q.contactAge=.00001*(i+1);q.contactDuration=.5;q.contactArea=1e-12;
  auto& h=q.thermal;h.initialized=true;h.status=(i%7==6)?0:1;h.temperature=2500.+i*.001;h.theta=.000123;
  h.savedVelocity[0]=100.;h.savedVelocity[1]=0.;h.savedVelocity[2]=0.;
  h.specificEnthalpyResidual=i%2?0x1p-42:-0x1p-42;h.contactMaximumArea=1e-11;h.contactPeakFraction=.25;
  h.coldFrozenArea=2e-13;h.coldContactAge=.00002*(i+1);h.cold2DFrozenArea=3e-13;
  for(int k=0;k<8;++k){h.coldNodeSpecificEnthalpy[k]=std::nextafter(float(3e6+i+k),float(INFINITY));h.coldRingSolidMass[k]=1e-15*(k+1);h.cold2DRingContactAge[k]=.00003*(k+1);}
  for(int k=0;k<64;++k)h.cold2DNodeSpecificEnthalpy[k]=std::nextafter(float(4e6+i+k),float(INFINITY));
  double specific=ugkwp::ParticleCaloric{}.enthalpy(h.temperature);
  if(q.state==2){specific=0;for(float value:h.coldNodeSpecificEnthalpy)specific+=double(value)/8;}
  if(q.state==3){specific=0;for(float value:h.cold2DNodeSpecificEnthalpy)specific+=double(value)/64;}
  q.energy=q.mass*(specific+h.specificEnthalpyResidual);
  ugkp::thermal::normalizeParticleContactMechanicalEnergy(q);
 }
 CHECK(ugkp::thermal::uploadParticleState(s,int(initial.size()),initial.data())==0);
 std::vector<ugkp::thermal::ParticleState<>> first,second;
 CHECK(ugkp::thermal::downloadParticleState(s,first)==0);CHECK(first.size()==initial.size());
 for(size_t i=0;i<first.size();++i) {
  const auto& a=initial[i];const auto& b=first[i];
#define UGKP_PARTICLE_FIELD(Field,Value) CHECK(a.Value==b.Value);
#define UGKP_PARTICLE_OPTIONAL(Field,Value) CHECK(a.Value==b.Value);
#define UGKP_PARTICLE_ARRAY(Field,Value,N) for(int j=0;j<N;++j)CHECK(a.Value[j]==b.Value[j]);
#include "ParticleStateFields.inc"
#undef UGKP_PARTICLE_FIELD
#undef UGKP_PARTICLE_OPTIONAL
#undef UGKP_PARTICLE_ARRAY
  double expected=ugkwp::ParticleCaloric{}.enthalpy(a.thermal.temperature);
  if(a.state==2){expected=0;for(float h:a.thermal.coldNodeSpecificEnthalpy)expected+=double(h)/8;}
  if(a.state==3){expected=0;for(float h:a.thermal.cold2DNodeSpecificEnthalpy)expected+=double(h)/64;}
  CHECK(b.energy==a.mass*(expected+a.thermal.specificEnthalpyResidual));
  CHECK(b.thermal.initialized);
  CHECK(a.energy==b.energy&&a.contactMechanicalEnergy==b.contactMechanicalEnergy);
  CHECK(b.contactMechanicalEnergy==(a.state>=2?10.:0.));
 }
 CHECK(ugkp::thermal::uploadParticleState(s,0,nullptr)==0);
 CHECK(readOne(s->particleCountDevice)==0);
 s->nBoundarySources=1;CHECK(ugkp::thermal::uploadParticleState(s,0,nullptr)==0);
 CHECK(s->particlesMayBePresent);s->nBoundarySources=0;
 CHECK(ugkp::thermal::uploadParticleState(s,int(first.size()),first.data())==0);
 CHECK(ugkp::thermal::downloadParticleState(s,second)==0);CHECK(second.size()==first.size());
 for(size_t i=0;i<first.size();++i){
  const auto& a=first[i];const auto& b=second[i];
#define UGKP_PARTICLE_FIELD(Field,Value) CHECK(a.Value==b.Value);
#define UGKP_PARTICLE_OPTIONAL(Field,Value) CHECK(a.Value==b.Value);
#define UGKP_PARTICLE_ARRAY(Field,Value,N) for(int j=0;j<N;++j)CHECK(a.Value[j]==b.Value[j]);
#include "ParticleStateFields.inc"
#undef UGKP_PARTICLE_FIELD
#undef UGKP_PARTICLE_OPTIONAL
#undef UGKP_PARTICLE_ARRAY
  CHECK(a.energy==b.energy&&a.contactMechanicalEnergy==b.contactMechanicalEnergy);
 }
 if(initial.size()==4) {
  double* fields[4]{};double host[6];const double material[4]={500.,.5,.25,-.75};
  for(int j=0;j<4;++j){CHECK(allocate(fields[j],6,"surface independent fields")==0);for(double& v:host)v=material[j];CHECK(copyToDevice(fields[j],host,6,"surface init")==0);}
  CHECK(ugkp::thermal::bindParticleSurface(s,6,{fields[0],fields[1],fields[2],fields[3]})==0);
  double* values=nullptr;CHECK(allocate(values,8,"surface physics result")==0);double before[8],after[8],legacy[8];
  surfaceResponse<<<1,1>>>(s->deviceState,values);CHECK(copyToHost(before,values,8,"surface before injection")==0);
  for(double& v:host)v=2900.;CHECK(copyToDevice(s->gasBoundaryT,host,6,"gas thermal boundary")==0);
  for(double& v:host)v=1000.;CHECK(copyToDevice(s->gasBoundaryUx,host,6,"gas ejection velocity")==0);
  surfaceResponse<<<1,1>>>(s->deviceState,values);CHECK(copyToHost(after,values,8,"surface after injection")==0);
  for(int k=0;k<8;++k)CHECK(before[k]==after[k]);CHECK(before[0]==1&&before[4]==1&&before[1]!=0);
  CHECK(ugkp::thermal::bindParticleSurface(s,6,{})==0);
  surfaceResponse<<<1,1>>>(s->deviceState,values);CHECK(copyToHost(legacy,values,8,"legacy gas wall surface")==0);
  CHECK(legacy[1]!=before[1]&&legacy[5]!=before[5]);
  CUDA_OK(cudaFree(values));for(auto field:fields)CUDA_OK(cudaFree(field));
 }
 std::printf("{\"particles\":%zu,\"complete_state_exact\":true,\"independent_theta_age\":true}\n",first.size());
 releaseState(s);
}
'''
    return text


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cu=out/"particle-state.cu";cu.write_text(source());exe=out/"particle-state"
    command=["/usr/local/cuda/bin/nvcc","-std=c++17","-O2","--fmad=false","--expt-relaxed-constexpr","--Ofast-compile=min",
        "-arch=sm_89","-I"+str(ROOT/"common/thermal"),"-I"+str(ROOT/"common"),"-I"+str(ROOT/"common/gasNumerics"),str(cu),str(ROOT/"common/thermal/GpuWallEnergy64.cu"),"-o",str(exe)]
    with (out/"build.log").open("w") as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    results=[]
    for count in (0,4,65537):
        run=subprocess.run([str(exe),str(count)],capture_output=True,text=True)
        (out/f"run-{count}.log").write_text(run.stdout+run.stderr);run.check_returncode();results.append(json.loads(run.stdout))
    with (out/"memcheck.log").open("w") as log:
        subprocess.run(["/usr/local/cuda/bin/compute-sanitizer","--tool","memcheck","--leak-check","full","--error-exitcode","88",str(exe),"4"],stdout=log,stderr=subprocess.STDOUT,check=True)
    (out/"results.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results))
if __name__=="__main__":main()
