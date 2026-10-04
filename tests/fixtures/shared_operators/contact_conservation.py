import os
from pathlib import Path
import re,subprocess,sys,os
root=Path(sys.argv[1]);out=Path(sys.argv[2]);app=sys.argv[3];bits=int(sys.argv[4]);out.mkdir(parents=True,exist_ok=True)
src=root/'applications'/app/('gpu' if app=='CHT' else 'private_backend')/'GpuResidentStrict.cu'
t=src.read_text().replace('#include "GpuAutomaticCsrScheduleFields.inl"', (root/"common/GpuAutomaticCsrScheduleFields.inl").read_text() if (root/"common/GpuAutomaticCsrScheduleFields.inl").is_file() else "");a=t.index('struct DeviceState');b=t.index('\n};',a)
fields=re.findall(r'^\s*((?:unsigned\s+)?(?:long long|char)|double|float|int|GpuReal|GpuTime|GpuWallEnergy)\*\s+(\w+)\s*=',t[a:b],re.M)
fields=[(ty,n) for ty,n in fields if n not in ['diagnosticPreTransportParticleCount','sourceInjectedCount'] and (bits==32 or not n.startswith('flatPressure'))]
code='#include "GpuResidentStrict.cu"\n#include <cstdio>\n#include <new>\n#include <cstdlib>\n#include <cmath>\n#include <vector>\n'
if app=='CHT' and bits==32:code+='using namespace ugkwpCudaFp32;\n'
code+=r"""
#define CHECK(x) do{if(!(x)){printf("FAIL %d %s\n",__LINE__,#x);return 1;}}while(0)
template<class T>void mem(T*&p,int n){if(cudaMallocManaged(&p,n*sizeof(T))!=cudaSuccess)std::abort();for(int i=0;i<n;++i)::new(static_cast<void*>(p+i)) T{};}
__global__ void contactStep(DeviceState*sp,double dt){for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<sp->particleCapacity;i+=blockDim.x*gridDim.x)relaxOneParticleToResidentGas(*sp,i,dt,ugkwpGpuDrag::SchillerNaumannDrag{});}
__global__ void enthalpy(DeviceState*sp,double*e){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<sp->particleCapacity)e[i]=double(sp->pm[i])*double(Foam::gpuThermal::aluminaSpecificEnthalpyJkg(sp->pT[i]));}
int main(){setvbuf(stdout,nullptr,_IONBF,0);DeviceState*s;mem(s,1);
"""
code+='\n'.join('mem(s->'+n+',1024);' for _,n in fields)
code+=r"""
const int N=129;s->particleCapacity=N;s->nCells=2;s->nFaces=2;s->rhoMin=1e-12;s->rhoSolid=3000;s->gasMu=1e-5;s->particleDiameterFallback=1e-4;s->TpMin=1;s->TpMax=5000;s->thetaMin=1e-12;s->epsSMin=1e-12;s->dragModelId=0;s->solveParticleTemperature=0;s->particleWallHeatTransferEnabled=1;s->coldWallSolidificationParameters.interfaceResistanceM2KW=1e-5;
double*e;mem(e,N);const double tol=BITS==32?2e-5:1e-11;
for(int block:{32,64,128,256})for(int enabled:{0,1}){
 s->particleWallHeatTransferEnabled=enabled;
 for(int c=0;c<2;++c){s->particleWallReflectedEnergy[c]=0;s->gasBoundaryT[c]=c?3400:800;s->particleStuckCandidateMask[c]=Foam::gpuThermal::particleWallReboundContact;s->particleWallContactAreaScale[c]=1;s->particleWallEffusivityByFace[c]=10000;}
 for(int i=0;i<N;++i){s->pStatus[i]=1;s->pCellId[i]=i%2;s->pm[i]=1e-9*(1+i%3);s->pd[i]=1e-4;s->pT[i]=2800+i%7*10;s->pStuck[i]=Foam::gpuThermal::particleWallTransientRebound;s->pStuckFaceId[i]=i%2;s->pContactDuration[i]=.001;s->pContactMaximumArea[i]=1e-8;s->pContactPeakFraction[i]=.5;s->pTheta[i]=0;AGE=.0002;s->pDepositionArea[i]=0;}
 enthalpy<<<2,128>>>(s,e);CHECK(cudaDeviceSynchronize()==cudaSuccess);double before[2]={};for(int i=0;i<N;++i)before[i%2]+=e[i];
 for(int step=0;step<3;++step){contactStep<<<(N+block-1)/block,block>>>(s,.00005);CHECK(cudaDeviceSynchronize()==cudaSuccess);}
 enthalpy<<<2,128>>>(s,e);CHECK(cudaDeviceSynchronize()==cudaSuccess);double after[2]={};for(int i=0;i<N;++i){after[i%2]+=e[i];CHECK(s->pStatus[i]==1);CHECK(s->pStuck[i]==Foam::gpuThermal::particleWallTransientRebound);CHECK(fabs(double(AGE)-.00035)<1e-10);}
 for(int c=0;c<2;++c){double delta=after[c]-before[c],q=s->particleWallReflectedEnergy[c];double residual=delta+q;double scaled=fabs(residual)/fmax(1e-30,fabs(delta)+fabs(q));printf("HEAT block=%d enabled=%d face=%d deltaParticle=%.17g wallGain=%.17g residual=%.17g relative=%.17g\n",block,enabled,c,delta,q,residual,scaled);CHECK(scaled<tol);if(enabled)CHECK(c?q<0:q>0);else CHECK(delta==0&&q==0);}
}
puts("PASS finite-contact per-face particle enthalpy + wall ledger conservation");}
"""
code=code.replace('BITS',str(bits)).replace('AGE','s->pContactAge[i]' if app=='CHT' else 's->pTheta[i]')
f=out/'contact.cu';f.write_text(code);exe=out/'contact'
cmd=[str(Path(os.environ.get('CUDA_HOME','/usr/local/cuda'))/'bin/nvcc'),'-std=c++17','-O3','-arch='+os.environ.get('UGKWP_CUDA_ARCH','sm_89'),'--fmad='+('false' if app=='CHT' else 'true'),'-DUGKWP_GPU_REAL_BITS='+str(bits),'-I'+str(src.parent),'-I'+str(root/'common'),'-I'+str(root/'applications'/app/'gpu'),str(f),'-o',str(exe)]
if app=='CHT':cmd.append(str(root/'applications/CHT/gpu/GpuWallEnergy64.cu'))
with (out/'build.log').open('w') as o:q=subprocess.run(cmd,stdout=o,stderr=subprocess.STDOUT)
if q.returncode:print((out/'build.log').read_text()[-5000:]);raise SystemExit(q.returncode)
q=subprocess.run([str(exe)],capture_output=True,text=True);(out/'run.log').write_text(q.stdout+q.stderr);print(q.stdout+q.stderr);raise SystemExit(q.returncode)
