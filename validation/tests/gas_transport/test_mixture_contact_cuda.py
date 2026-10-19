"""Real CUDA production mixture dispatcher, including comoving ALE contact."""
import os,re,shutil,subprocess
from pathlib import Path
import pytest
from validation.tools.paths import SOURCE_ROOT,VALIDATION_ROOT
from test_mixture_transport import fixture

def test_real_gpu_stationary_moving_and_ale_mixture_contacts(tmp_path):
    nvcc=os.environ.get('NVCC')or shutil.which('nvcc')
    if not nvcc:pytest.skip('CUDA compiler unavailable')
    gpu=shutil.which('nvidia-smi')
    if not gpu or subprocess.run([gpu,'-L'],capture_output=True).returncode:pytest.skip('CUDA GPU unavailable')
    pre=(Path(VALIDATION_ROOT)/'tests/gas_transport/gas_state_probe.cpp').read_text().split('void hashBytes')[0]
    adapters=('#define __host__','#define __global__','#define __device__','#define __forceinline__','#define asm(', 'struct ThreadIndex ')
    pre='\n'.join(line for line in pre.splitlines()if not line.startswith(adapters))
    setup=fixture().split('void transport')[0]
    setup=setup.replace('blockDim.x=1;blockIdx.x=0;','').replace('for(int c=0;c<2;++c){threadIdx.x=c;recoverGasPrimitivesKernel(&s);}','recoverGasPrimitivesKernel<<<1,2>>>(&s);cudaCheck(cudaDeviceSynchronize());')
    # Every pointer dereferenced by the production device function is genuinely
    # CUDA-accessible, including the complete common borrowed-state fixture.
    setup=re.sub(r'new ([\w:<>]+)\[([^\]]+)\]\{([^}]*)\}',r'managed<\1>(\2,{\3})',setup)
    setup=setup.replace('new ugkwp::SpeciesThermoData<Real>[2]','managed<ugkwp::SpeciesThermoData<Real>>(2)')
    support=r"""
#include <initializer_list>
#include <new>
void cudaCheck(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));std::abort();}}
template<class T>T*managed(std::size_t n,std::initializer_list<T> initial={}){T*p;cudaCheck(cudaMallocManaged(&p,n*sizeof(T)));for(std::size_t i=0;i<n;++i)new(p+i)T{};std::copy(initial.begin(),initial.end(),p);return p;}
"""
    body=r"""
__global__ void evaluate(State*s,Real*f,bool native){if(threadIdx.x==0)f[5]=computeRiemannGasFaceFluxDevice<false>(*s,0,f[0],f[1],f[2],f[3],f[4],native);}
int main(){
for(int mode=0;mode<4;++mode)for(int native=0;native<2;++native){
 State&s=*managed<State>(1);initialise(s);s.gasFluxScheme=4;s.gasSpecies.thermoControls.relativeEnergyTolerance=1e-12;s.gasSpecies.thermoControls.relativeTemperatureTolerance=1e-12;
 auto*m=const_cast<ugkwp::SpeciesThermoData<Real>*>(s.gasSpecies.thermo.species);auto*a=const_cast<Real*>(s.gasSpecies.thermo.coefficients);
 m[0].molarMass=.028;m[1].molarMass=.044;a[0]=1000;a[1]=.1;a[2]=-1e6;a[3]=1200;a[4]=.2;a[5]=2e6;
 Real un=mode==0?0:mode==2?-25:25;
 for(int c=0;c<2;++c){Real y[2]={c?Real(.2):Real(.8),c?Real(.8):Real(.2)};Real T=c?1100:500;Real rho=1e5/(ugkwp::mixtureGasConstant(y,s.gasSpecies.thermo)*T);
 s.rho[c]=rho;s.rhoUx[c]=rho*un;s.rhoUy[c]=rho*(c?-70:120);s.rhoUz[c]=rho*23;
 Real partial[2]={rho*y[0],rho*y[1]};s.gasSpecies.rho[c]=partial[0];s.gasSpecies.rho[2+c]=partial[1];s.rhoE[c]=ugkwp::mixtureEnergy(partial,T,s.gasSpecies.thermo)+.5*rho*(un*un+(c?4900:14400)+23*23);}
 recoverGasPrimitivesKernel<<<1,2>>>(&s);cudaCheck(cudaDeviceSynchronize());
 for(int c=0;c<2;++c)ck(s.gasSpecies.cellStatus[c]==0,"GPU contact recovery");
 if(mode==3){s.gasGeometry.enabled=true;s.gasGeometry.interval=.001;s.gasGeometry.oldVolume=managed<Real>(2,{1,1});s.gasGeometry.newVolume=managed<Real>(2,{1,1});s.gasGeometry.faceSweptVolume=managed<Real>(3,{.025,-.025,.025});}
 Real*out=managed<Real>(6);evaluate<<<1,32>>>(&s,out,native);cudaCheck(cudaDeviceSynchronize());ck(out[5]==1,"GPU mixture dispatcher");
 const int donor=un>=0?0:1;Real wanted[5]={s.rho[donor]*un,s.rhoUx[donor]*un+1e5,s.rhoUy[donor]*un,s.rhoUz[donor]*un,(s.rhoE[donor]+1e5)*un};
 if(mode==3){wanted[0]=wanted[2]=wanted[3]=0;wanted[1]=1e5;wanted[4]=25e5;}

 // Roundoff has physical flux units; a zero energy flux must be compared with
 // the acoustic energy scale, not with an arbitrary dimensionless 1.
 Real sound=std::max(s.gasSpecies.soundSpeed[0],s.gasSpecies.soundSpeed[1]);
 Real density=std::max(s.rho[0],s.rho[1]);Real energy=std::max(std::abs(s.rhoE[0]),std::abs(s.rhoE[1]));
 Real scale[5]={density*sound,1e5+density*sound*sound,density*sound*120,density*sound*23,(energy+1e5)*sound};
 for(int k=0;k<5;++k)if(!(std::abs(out[k]-wanted[k])<=512*std::numeric_limits<Real>::epsilon()*scale[k]+1e-10*std::abs(wanted[k]))){std::fprintf(stderr,"mode=%d native=%d component=%d actual=%.17g expected=%.17g",mode,native,k,out[k],wanted[k]);std::abort();}
 ck(std::abs(s.gasSpecies.flux[0]+s.gasSpecies.flux[3]-out[0])<1e-10,"real GPU species closure");
}
std::puts("8 real GPU contact/ALE/native combinations passed");}
"""
    source=tmp_path/'contact.cu';source.write_text('#include <cuda_runtime.h>\n#include <algorithm>\n'+pre+support+setup+body);exe=tmp_path/'contact'
    cmd=[nvcc,'-std=c++17','-O2','-arch=sm_89','-I'+str(SOURCE_ROOT/'common'),'-I'+str(SOURCE_ROOT/'common/gasNumerics'),'-I'+str(Path(VALIDATION_ROOT)/'tests/gas_transport'),str(source),'-o',str(exe)]
    result=subprocess.run(cmd,capture_output=True,text=True);assert result.returncode==0,result.stdout+result.stderr
    result=subprocess.run([str(exe)],capture_output=True,text=True);assert result.returncode==0,result.stdout+result.stderr
