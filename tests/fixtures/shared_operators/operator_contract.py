import os
from pathlib import Path
import sys,subprocess
root=Path(sys.argv[1]);out=Path(sys.argv[2]);app=sys.argv[3];bits=int(sys.argv[4]);out.mkdir(parents=True,exist_ok=True)
src=root/f'applications/{app}/{"gpu" if app=="CHT" else "private_backend"}'
code='#include "GpuResidentStrict.cu"\n#include <cstdio>\n#include <new>\n#include <cstdlib>\n#include <cmath>\n'
if app=='CHT' and bits==32:code+='using namespace ugkwpCudaFp32;\n'
code+=r'''
template<class T>void alloc(T*&p,int n){if(cudaMallocManaged(&p,n*sizeof(T))!=cudaSuccess)exit(10);for(int i=0;i<n;i++)::new(static_cast<void*>(p+i)) T{};}
void deviceSync(){auto e=cudaDeviceSynchronize();if(e!=cudaSuccess){printf("CUDA %s\n",cudaGetErrorString(e));exit(11);}}
void check(double a,double b,const char*what){if(!std::isfinite(a)||fabs(a-b)>TOL*fmax(1.,fabs(b))){printf("FAIL %s %.17g != %.17g\n",what,a,b);exit(12);}}
int main(){DeviceState*s;alloc(s,1);s->nCells=2;s->nFaces=3;s->nInternalFaces=1;s->rhoMin=1e-12;s->Rgas=1;s->TgasMin=.001;s->sstConfigured=1;
ALLOC
s->cellPlaneStart[0]=0;s->cellPlaneStart[1]=2;s->cellPlaneCount[0]=s->cellPlaneCount[1]=2;
s->cellFaceId[0]=0;s->cellFaceId[1]=1;s->cellFaceId[2]=0;s->cellFaceId[3]=2;
s->faceOwner[0]=0;s->faceOwner[1]=0;s->faceOwner[2]=1;s->faceNeighbour[0]=1;s->faceNeighbour[1]=1;s->faceNeighbour[2]=0;
s->facePeriodicPair[0]=-1;s->facePeriodicPair[1]=2;s->facePeriodicPair[2]=1;
s->faceWeight[0]=.25;s->faceWeight[1]=.75;s->faceWeight[2]=.25;
s->Sfx[0]=1;s->Sfx[1]=-1;s->Sfx[2]=1;
s->k[0]=1;s->k[1]=3;s->omega[0]=2;s->omega[1]=6;s->p[0]=2;s->p[1]=8;s->V[0]=s->V[1]=1;
computeSstGradientsKernel<<<1,32>>>(s);computeGasHllcAdcSensorKernel<<<1,32>>>(s);deviceSync();
for(int c=0;c<2;c++){check(s->gradKX[c],c?-1:1,"periodic gradK");check(s->gradOmegaX[c],c?-2:2,"periodic gradOmega");check(s->gradKY[c],0,"gradKY");check(s->gradKZ[c],0,"gradKZ");check(s->gasHllcAdcSensor[c],.015625,"sensor");}
puts("PASS nonuniform periodic neighbour gradient and sensor");
s->rho[0]=s->rho[1]=1;s->gasPhiRho[0]=2;s->gasPhiRho[1]=0;s->gasPhiRho[2]=0;
s->gasPhiRhoUx[0]=6;s->gasPhiRhoUy[0]=-2;s->gasPhiRhoUz[0]=4;s->gasPhiRhoE[0]=10;
computeGasFluxPositivityScaleKernel<<<1,32>>>(s,1.);applyGasFluxPositivityScaleKernel<<<1,32>>>(s);deviceSync();
double expected=.999*(1.-s->rhoMin)/2.;check(s->gasFluxPositivityScale[0],expected,"outflow limiter");check(s->gasPhiRho[0],2*expected,"mass flux");check(s->gasPhiRhoUx[0],6*expected,"momentum flux");check(s->gasPhiRhoE[0],10*expected,"energy flux");
WALL_TEST
puts("PASS limited mass momentum energy flux and optional wall ledger");
for(int c=0;c<2;c++){s->rho[c]=2;s->rhoUx[c]=3;s->rhoUy[c]=-1;s->rhoUz[c]=.5;s->rhoE[c]=100;}
s->gravityX=.2;s->gravityY=-9.8;s->gravityZ=1;
GRAVITY_KERNEL<<<1,32>>>(s,.125);deviceSync();
for(int c=0;c<2;c++){
 check(s->rhoUx[c],3+2*.2*.125,"gravity momentum X");check(s->rhoUy[c],-1+2*(-9.8)*.125,"gravity momentum Y");check(s->rhoUz[c],.5+2*.125,"gravity momentum Z");
 double kinetic=(double(s->rhoUx[c])*s->rhoUx[c]+double(s->rhoUy[c])*s->rhoUy[c]+double(s->rhoUz[c])*s->rhoUz[c])/4.;
 check(s->rhoE[c]-kinetic,100-10.25/4.,"gravity internal energy invariant");
}
puts("PASS gravity impulse and work energy budget");
DeviceState*m;alloc(m,1);m->nCells=2;m->particleCapacity=37;m->TpMin=1;m->TpMax=10000;m->particleDiameterFallback=1e-5;
MALLOC
*m->particleCountDevice=37;
double ref[2][7]={};
for(int i=0;i<37;i++){int c=i%2;m->pCellId[i]=c;m->pStatus[i]=(i%7!=0);m->pm[i]=.5+.125*(i%3);m->pux[i]=i*.125;m->puy[i]=-.5;m->puz[i]=.25;m->pTheta[i]=.125;m->pd[i]=.0625;m->pT[i]=300;STUCK_INIT
if(!m->pStatus[i])continue;double mass=m->pm[i],theta=THETA_REF;ref[c][0]+=mass;ref[c][1]+=mass*m->pux[i];ref[c][2]+=mass*m->puy[i];ref[c][3]+=mass*m->puz[i];ref[c][4]+=mass*(.5*(m->pux[i]*m->pux[i]+.3125)+1.5*theta);ref[c][5]+=mass*m->pd[i];ref[c][6]++;}
clearParticleMomentsAndCountsAtomicKernel<<<1,32>>>(m);accumulateParticleMomentsAtomicKernel<<<2,32>>>(m);deviceSync();
for(int c=0;c<2;c++){double got[]={m->momRhoP[c],m->momRhoUPx[c],m->momRhoUPy[c],m->momRhoUPz[c],m->momRhoEP[c],m->momRhoPD[c],double(m->cellParticleCount[c])};for(int j=0;j<7;j++)check(got[j],ref[c][j],"atomic particle moment");}
check(m->cellParticleCount[2],0,"count sentinel");puts("PASS atomic mass momentum mechanical energy diameter and survivor count");return 0;}
'''
fields='cellPlaneStart cellPlaneCount cellFaceId faceOwner faceNeighbour facePeriodicPair faceWeight Sfx Sfy Sfz k omega p V gradKX gradKY gradKZ gradOmegaX gradOmegaY gradOmegaZ gasHllcAdcSensor rho gasPhiRho gasPhiRhoUx gasPhiRhoUy gasPhiRhoUz gasPhiRhoE gasFluxPositivityScale rhoUx rhoUy rhoUz rhoE'.split()
code=code.replace('ALLOC','\n'.join('alloc(s->'+f+',4);' for f in fields),1)
fields='particleCountDevice pCellId pStatus pm pux puy puz pTheta pd pT momRhoP momRhoUPx momRhoUPy momRhoUPz momRhoEP momRhoPD momRhoHpP cellParticleCount'.split()
if app!='gasUGKP':fields+=['pStuck']
code=code.replace('MALLOC','\n'.join('alloc(m->'+f+',40);' for f in fields))
code=code.replace('STUCK_INIT','m->pStuck[i]=(i%3==0);' if app!='gasUGKP' else '').replace('THETA_REF','(i%3==0?0.:.125)' if app!='gasUGKP' else '.125')
wall='alloc(s->wallEnergy.gasWallEnergy,4);alloc(s->wallEnergy.gasWallEnergyMask,4);alloc(s->wallEnergy.gasWallFlux,4);alloc(s->gasBoundaryKind,4);s->wallEnergy.gasWallEnergyMask[1]=1;s->gasBoundaryKind[1]=2;s->gasPhiRho[1]=1;s->gasPhiRhoE[1]=20;s->gasFluxPositivityScale[0]=.25;applyGasFluxPositivityScaleKernel<<<1,32>>>(s);deviceSync();check(s->gasPhiRhoE[1],5,"wall limited energy");check(s->wallEnergy.gasWallFlux[1],5,"published wall limited energy");'
code=code.replace('GRAVITY_KERNEL','applyGasGravitySourceKernel' if app=='gasUGKP' else 'applyGasGravityKernel')
code=code.replace('WALL_TEST',wall if app=='CHT' else '').replace('TOL','2e-6' if bits==32 else '2e-12')
f=out/'operator_contract.cu';f.write_text(code);exe=out/'operator_contract';cmd=[str(Path(os.environ.get('CUDA_HOME','/usr/local/cuda'))/'bin/nvcc'),'-std=c++17','-O3','-arch='+os.environ.get('UGKWP_CUDA_ARCH','sm_89'),'--fmad='+('false' if app=='CHT' else 'true'),'-DUGKWP_GPU_REAL_BITS='+str(bits),'-I'+str(src),'-I'+str(root/'common'),'-I'+str(root/'applications'/app/'gpu'),str(f),'-o',str(exe)]
if app=='CHT':cmd.append(str(src/'GpuWallEnergy64.cu'))
with (out/'build.log').open('w') as log:q=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
if q.returncode:print((out/'build.log').read_text()[-6000:]);sys.exit(q.returncode)
q=subprocess.run([str(exe)],capture_output=True,text=True);(out/'run.log').write_text(q.stdout+q.stderr);print(app,bits,q.stdout,q.stderr);sys.exit(q.returncode)
