
from pathlib import Path
import re,subprocess,sys,os
root=Path(sys.argv[1]).resolve();p=Path(sys.argv[2]).resolve();branch=sys.argv[3];bits=int(sys.argv[4]);mode=sys.argv[5] if len(sys.argv)>5 else "split"
for name in ['tests','bin','logs']:(p/name).mkdir(parents=True,exist_ok=True)
folder='private_backend' if branch in ['gasUGKP','FSH'] else 'gpu'
src=root/'applications'/branch/folder/'GpuResidentStrict.cu'
raw=src.read_text().replace('#include "GpuAutomaticCsrScheduleFields.inl"', (root/"common/GpuAutomaticCsrScheduleFields.inl").read_text() if (root/"common/GpuAutomaticCsrScheduleFields.inl").is_file() else "");a=raw.index('struct DeviceState');b=raw.index('\n};',a);d=raw[a:b]
ptr=re.findall(r'^\s*((?:unsigned\s+)?(?:long long|char)|double|float|int|unsigned int|GpuReal|GpuTime|GpuWallEnergy)\*\s+(\w+)\s*=',d,re.M)
ptr=[(t,n) for t,n in ptr if n not in ['diagnosticPreTransportParticleCount','sourceInjectedCount']]
if branch == 'gasUGKP' and mode == 'unsorted':ptr.append(('PressureProjectionCell','pressureProjectionCache'))
if bits!=32:ptr=[(t,n) for t,n in ptr if not n.startswith('flatPressure')]
txt='#include "GpuResidentStrict.cu"\n#include <cstdio>\n#include <new>\n#include <cstdlib>\n'
if branch=='CHT' and bits==32:txt+='using namespace ugkwpCudaFp32;\n'
txt+=r'''
template<class T> void mem(T*&p,int n){if(cudaMallocManaged(&p,sizeof(T)*n)!=cudaSuccess)std::abort();for(int i=0;i<n;++i)::new(static_cast<void*>(p+i)) T{};}
#define CHECK(x) do {if(!(x)){printf("FAIL line%d %s\n",__LINE__,#x);return 1;}}while(0)
int main(){setvbuf(stdout,nullptr,_IONBF,0);
DeviceState*s=new DeviceState{};
'''
txt+='\n'.join(f'mem(s->{name},129);' for typ,name in ptr)
txt+=r'''
s->deviceState=s;s->nCells=2;s->nFaces=1;s->particleCapacity=129;s->rhoMin=1e-20;
s->rhoSolid=3000;s->epsSMin=1e-12;s->thetaMin=1e-12;s->pressureKickFraction=.01;
s->TpMin=1;s->TpMax=5000;s->particleDiameterFallback=1e-5;s->csrCellLocalPathEnabled=1;s->splitPreDirectoryActive=1;
s->particleWorkGrid=4;s->particleBlockThreads=64;s->nBoundarySources=1;*s->particleCountDevice=6;
s->faceOwner[0]=0;s->faceNeighbour[0]=1;
s->solidPressurePhiMomX[0]=1;s->solidPressurePhiMomY[0]=0;s->solidPressurePhiMomZ[0]=0;s->solidPressurePhiEnergy[0]=0;
for(int c=0;c<2;++c){
 s->V[c]=1;s->cellLength[c]=1;s->cellPlaneStart[c]=c;s->cellPlaneCount[c]=1;s->cellFaceId[c]=0;
 s->momRhoP[c]=1;s->momRhoUPx[c]=s->momRhoUPy[c]=s->momRhoUPz[c]=0;s->momRhoEP[c]=1.75;
 s->preBaseCellOffset[c]=2*c;s->cellParticleOffset[c]=c;s->sortedParticleIndex[c]=4+c;
 for(int j=0;j<3;++j){int i=j==2?4+c:2*c+j;s->pStatus[i]=1;s->pCellId[i]=c;s->pm[i]=j==2?.5:.25;
 s->pux[i]=j==2?0:j==1?1:-1;s->puy[i]=s->puz[i]=0;s->pTheta[i]=1;s->pT[i]=1000;s->pd[i]=1e-5;
 THERMAL_INIT
 }
}
s->preBaseCellOffset[2]=4;s->cellParticleOffset[2]=2;
DeviceState* device;mem(device,1);*device=*s;s->deviceState=device;
CHECK(PHYSICS::limit(s,1.,1,64)==cudaSuccess);
scaleCollisionalPressureFaceFluxKernel<<<1,64>>>(s->deviceState);
CHECK(launchPressurePreflight(s,1.,1,64,true,false)==cudaSuccess);
CHECK(PHYSICS::project(s,1.,1,64,64,true,false)==cudaSuccess);
publishPressureCanonicalMomentsKernel<<<1,64>>>(s->deviceState);
CHECK(cudaGetLastError()==cudaSuccess);
CHECK(cudaDeviceSynchronize()==cudaSuccess);
const double phi=s->solidPressurePhiMomX[0];
CHECK(fabs(phi-.01)<TOL);
double totalMass=0,totalPx=0,totalEnergy=0;
for(int c=0;c<2;++c){
 double m=0,px=0,energy=0;
 for(int i=0;i<6;++i)if(s->pCellId[i]==c){m+=s->pm[i];px+=s->pm[i]*s->pux[i];energy+=s->pm[i]*(.5*(s->pux[i]*s->pux[i]+s->puy[i]*s->puy[i]+s->puz[i]*s->puz[i])+1.5*s->pTheta[i]);}
 printf("cell%d limitedPhi=%.17g momentum=%.17g mass=%.17g energy=%.17g\n",c,phi,px,m,energy);
 CHECK(fabs(px-(c?phi:-phi))<TOL);
 CHECK(fabs(s->momRhoUPx[c]-px)<TOL);
 CHECK(fabs(s->momRhoEP[c]-energy)<TOL);
 CHECK(fabs(m-1)<TOL);totalMass+=m;totalPx+=px;totalEnergy+=energy;
}
CHECK(fabs(totalMass-2)<TOL);CHECK(fabs(totalPx)<TOL);CHECK(fabs(totalEnergy-3.5)<TOL);

// Preserve the original balance assertions above. These read-only transaction
// regressions reproduce the extreme normal inputs from the independent review.
// Deliberately inspect the failure flag rather than launching the fatal trap.
for(int example=0;example<2;++example)for(int route=0;route<4;++route){
 s->nCells=1;s->nFaces=1;*s->particleCountDevice=2;
 s->faceOwner[0]=0;s->faceNeighbour[0]=-1;
 s->cellPlaneStart[0]=0;s->cellPlaneCount[0]=1;s->cellFaceId[0]=0;
 s->V[0]=example?(sizeof(PressureReal)==4?PressureReal(1e20):PressureReal(1e160)):PressureReal(1);
 s->momRhoP[0]=1;s->momRhoEP[0]=example?PressureReal(1):PressureReal(1.5);
 s->momRhoUPx[0]=s->momRhoUPy[0]=s->momRhoUPz[0]=0;
 s->pressureKickFraction=0;
 s->solidPressurePhiMomX[0]=s->solidPressurePhiMomY[0]=s->solidPressurePhiMomZ[0]=0;
 s->solidPressurePhiEnergy[0]=example?PressureReal(0):PressureReal(-.2);
 s->preBaseCellOffset[0]=0;s->preBaseCellOffset[1]=route==3?1:0;
 s->cellParticleOffset[0]=route==3?1:0;s->cellParticleOffset[1]=2;
 s->sortedParticleIndex[0]=0;s->sortedParticleIndex[1]=1;
 const PressureReal speed=sizeof(PressureReal)==4?PressureReal(1e-20):PressureReal(1e-170);
 const PressureReal eps=pressure_convex::epsilon<PressureReal>();
 for(int i=0;i<2;++i){
  s->pStatus[i]=s->compactPStatus[i]=1;s->pCellId[i]=s->compactPCellId[i]=0;
  s->pm[i]=s->compactPm[i]=s->V[0]/PressureReal(2);
  s->pux[i]=s->compactPux[i]=example?PressureReal(0):i?-speed:speed*(PressureReal(1)+PressureReal(32)*eps);
  s->puy[i]=s->compactPuy[i]=s->puz[i]=s->compactPuz[i]=0;
  s->pTheta[i]=s->compactPTheta[i]=example?PressureReal(2)/PressureReal(3):PressureReal(1);
  THERMAL_RESET
 }
 for(int i=0;i<3;++i)s->pressureFailure[i]=0;
 *device=*s;
 preparePressurePreflightKernel<<<1,64>>>(device,1.);
 if(route==0)previewPressureSortedKernel<false,false><<<1,128>>>(device);
 else if(route==1)previewPressureUnsortedKernel<<<1,128>>>(device);
 else if(route==2)previewPressureSortedKernel<false,true><<<1,128>>>(device);
 else previewPressureSortedKernel<true,false><<<1,128>>>(device);
 auditPressurePreviewKernel<<<1,64>>>(device,1.);
 CHECK(cudaGetLastError()==cudaSuccess);CHECK(cudaDeviceSynchronize()==cudaSuccess);
 if(example){
  CHECK(s->pressureFailure[0]==0);
  CHECK(s->pux[0]==0&&s->pux[1]==0);
 }else{
  const PressureReal change=s->pressurePreviewMoments[0]-s->pressurePreviewMoments[6];
  CHECK(change!=0&&change*change==0);
  CHECK((s->pressureFailure[0]&pressureUnrealizableParticles)!=0);
  CHECK(s->pressureFailure[1]==1&&s->pressureFailure[2]!=0);
  CHECK(s->pux[0]==speed*(PressureReal(1)+PressureReal(32)*eps)&&s->pux[1]==-speed);
 }
 CHECK(s->momRhoUPx[0]==0);
 CHECK(s->momRhoEP[0]==(example?PressureReal(1):PressureReal(1.5)));
}
puts("PASS extreme pressure: normal dU underflow rejected and finite closure scale retained in all four preview routes");
puts("PASS limited pressure: local face balance, particle/Eulerian moments, mass, momentum and energy");
}
'''
if branch=='gasUGKP':txt=txt.replace('s->splitPreDirectoryActive=1;','')
if mode == 'unsorted':
    txt=txt.replace('s->csrCellLocalPathEnabled=1;', 's->csrCellLocalPathEnabled=0;')
    txt=txt.replace('PHYSICS::project(s,1.,1,64,64,true,false)', 'PHYSICS::projectUnsorted(s,1.,1,64)')

txt=txt.replace('THERMAL_INIT','s->pStuck[i]=0;' if branch!='gasUGKP' else '')
txt=txt.replace('THERMAL_RESET','s->pStuck[i]=s->compactPStuck[i]=0;' if branch!='gasUGKP' else '')
txt=txt.replace('PHYSICS','AnalyticPressureLaunch' if branch=='gasUGKP' else 'ConstrainedPressureLaunch<true>')
txt=txt.replace('TOL','2e-6' if bits==32 else '2e-12')
f=p/'tests'/f'pressure_balance_{branch}{bits}.cu';f.write_text(txt)
exe=p/'bin'/f'pressure_balance_{branch}{bits}'
cmd=[str(Path(os.environ.get('CUDA_HOME','/usr/local/cuda'))/'bin/nvcc'),'-std=c++17','-O3','-arch='+os.environ.get('UGKWP_CUDA_ARCH','sm_89'),'--fmad='+('false' if branch=='CHT' else 'true'),'-DUGKWP_GPU_REAL_BITS='+str(bits),'-I'+str(src.parent),'-I'+str(root/'common'),'-I'+str(root/('applications/gasUGKP/gpu' if branch=='gasUGKP' else 'applications/FSH/gpu')),str(f),'-o',str(exe)]
if branch=='CHT' and bits==32:cmd.append('-DUGKP_DEVELOPMENT_PROBES=1')
if branch=='CHT':cmd.append(str(root/'applications/CHT/gpu/GpuWallEnergy64.cu'))
log=p/'logs'/f'pressure_balance_{branch}{bits}_build.log'
with log.open('w') as out:q=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT)
print('BUILD',branch,bits,q.returncode,flush=True)
if q.returncode:print(log.read_text()[-3500:]);raise SystemExit(q.returncode)
if os.environ.get("UGKP_PRESSURE_FIXTURE_COMPILE_ONLY") == "1":
    print("COMPILE_ONLY: GPU execution intentionally omitted", flush=True)
    raise SystemExit(0)
q=subprocess.run([str(exe)],capture_output=True,text=True)
(p/'logs'/f'pressure_balance_{branch}{bits}_run.log').write_text(q.stdout+q.stderr)
print('RUN_RETURN',q.returncode,q.stdout+q.stderr);raise SystemExit(q.returncode)
