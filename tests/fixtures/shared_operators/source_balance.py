import os
from pathlib import Path
import subprocess,sys,json,time
root=Path(sys.argv[1]).resolve(); out=Path(sys.argv[2]).resolve(); app=sys.argv[3];bits=int(sys.argv[4]);out.mkdir(parents=True,exist_ok=True)
leaf='gpu' if app=='CHT' else 'private_backend'; src=root/'applications'/app/leaf
code='#include "GpuResidentStrict.cu"\n#include <cstdio>\n#include <new>\n#include <cstdlib>\n#include <cmath>\n'
if app=='CHT' and bits==32:code+='using namespace ugkwpCudaFp32;\n'
code+=r"""
template<class T> void alloc(T*&p,int n){if(cudaMallocManaged(&p,n*sizeof(T))!=cudaSuccess)std::abort();for(int i=0;i<n;i++)::new(static_cast<void*>(p+i)) T{};}
int main(int argc,char**argv){DeviceState*s;alloc(s,1);s->nCells=6;s->nFaces=0;s->rhoSolid=1000;s->rhoMin=1e-12;s->gammaGas=1.4;s->Rgas=1;s->TgasMin=.001;
ALLOC
const double before[6]={.8,.9,.5,.99,.2,.7},after[6]={.9,.8,.5,.98,.24,.6};
for(int c=0;c<6;c++){s->epsGPrev[c]=before[c];s->momRhoP[c]=(1-after[c])*1000;s->rho[c]=1;s->rhoUx[c]=2;s->rhoUy[c]=-1;s->rhoUz[c]=.5;s->rhoE[c]=100;s->p[c]=(100-2.625)*.4;s->Ux[c]=2;s->Uy[c]=-1;s->Uz[c]=.5;s->V[c]=1;}
if(argc>1){s->epsGPrev[0]=.2;s->momRhoP[0]=200;}
applyGasVolumeFractionSourceKernel<<<1,32>>>(s,.001);
auto err=cudaDeviceSynchronize();if(argc>1){if((err==cudaErrorIllegalInstruction || err==cudaErrorLaunchFailure)){puts("PASS rejected nonpositive internal energy");return 0;}printf("FAIL expected trap, got %s\n",cudaGetErrorString(err));return 3;}if(err!=cudaSuccess){printf("CUDA FAIL %s\n",cudaGetErrorString(err));return 2;}
double worst=0;
for(int c=0;c<6;c++){
 double dm=after[c]*s->rho[c]-before[c];
 double dpx=after[c]*s->rhoUx[c]-2*before[c];
 double dpy=after[c]*s->rhoUy[c]+before[c];
 double dpz=after[c]*s->rhoUz[c]-.5*before[c];
 double de=after[c]*s->rhoE[c]-100*before[c]+(after[c]-before[c])*(100-2.625)*.4;
 if(!std::isfinite(dm)||!std::isfinite(dpx)||!std::isfinite(dpy)||!std::isfinite(dpz)||!std::isfinite(de)){puts("FAIL nonfinite balance");return 4;}
 double err=fmax(fabs(dm),fmax(fabs(dpx),fmax(fabs(dpy),fmax(fabs(dpz),fabs(de)/100))));worst=fmax(worst,err);
 printf("CELL %d mass=%.17g px=%.17g py=%.17g pz=%.17g energy_balance=%.17g\n",c,dm,dpx,dpy,dpz,de);
}
printf("MAX_NORMALIZED_RESIDUAL %.17g\n",worst);
if(worst>TOL){puts("FAIL discrete phase-weighted source balance");return 1;}
for(int c=0;c<6;c++)if(fabs(s->epsGPrev[c]-after[c])>TOL){puts("FAIL previous fraction not committed");return 5;}
double saved[6];for(int c=0;c<6;c++)saved[c]=s->rhoE[c];
applyGasVolumeFractionSourceKernel<<<1,32>>>(s,.001);
if(cudaDeviceSynchronize()!=cudaSuccess)return 6;
for(int c=0;c<6;c++)if(s->rhoE[c]!=saved[c]){puts("FAIL repeated zero source changed energy");return 7;}
puts("PASS discrete phase-weighted source balance and repeated zero source");return 0;}
"""
fields='epsGPrev momRhoP rho rhoUx rhoUy rhoUz rhoE p Ux Uy Uz V cellPlaneStart cellPlaneCount'.split()
code=code.replace('ALLOC','\n'.join('alloc(s->'+f+',6);' for f in fields)).replace('TOL','2e-6' if bits==32 else '2e-12')
f=out/'source_balance.cu';f.write_text(code);exe=out/'source_balance'
cmd=[str(Path(os.environ.get('CUDA_HOME','/usr/local/cuda'))/'bin/nvcc'),'-std=c++17','-O3','-arch='+os.environ.get('UGKWP_CUDA_ARCH','sm_89'),'--fmad='+('false' if app=='CHT' else 'true'),'-DUGKWP_GPU_REAL_BITS='+str(bits),'-I'+str(src),'-I'+str(root/'common'),'-I'+str(root/'applications'/app/'gpu'),str(f),'-o',str(exe)]
if app=='CHT':cmd.append(str(src/'GpuWallEnergy64.cu'))
with (out/'build.log').open('w') as log:q=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
if q.returncode: print((out/'build.log').read_text()[-8000:]);sys.exit(q.returncode)
q=subprocess.run([str(exe)],capture_output=True,text=True);(out/'run.log').write_text(q.stdout+q.stderr);print(app,bits,q.stdout,q.stderr,flush=True)
if q.returncode==0:
 q=subprocess.run([str(exe),'reject'],capture_output=True,text=True);(out/'reject.log').write_text(q.stdout+q.stderr);print(q.stdout,q.stderr,flush=True)
sys.exit(q.returncode)
