"""Compile actual backend kernels: occupancy independence and grid-invariant transport.
No timing result is obtained from this fixture; performance uses full native solvers.
"""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);p.add_argument('app',choices=['gasUGKP','FSH','CHT']);p.add_argument('bits',type=int);p.add_argument('--require-independent',action='store_true');p.add_argument('--require-particle-independent',action='store_true');a=p.parse_args()
root=a.root;out=a.out;out.mkdir(parents=True,exist_ok=True)
src=root/'applications'/a.app/('gpu' if a.app=='CHT' else 'private_backend');t=(src/'GpuResidentStrict.cu').read_text()
i=t.index('struct DeviceState');j=t.index('\n};',i)
fields=re.findall(r'^\s*((?:unsigned\s+)?(?:long long|char)|double|float|int|GpuReal|GpuTime|GpuWallEnergy)\*\s+(\w+)\s*=',t[i:j],re.M)
fields=[(ty,n) for ty,n in fields if not n.startswith('flatPressure') and n not in ['diagnosticPreTransportParticleCount','sourceInjectedCount']]
code='#define UGKP_DEVELOPMENT_PROBES 1\n#include "GpuResidentStrict.cu"\n#include <cstdio>\n#include <new>\n#include <cstdlib>\n#include <cmath>\n#include <vector>\n'
if a.app=='CHT' and a.bits==32:code+='using namespace ugkwpCudaFp32;\n'
code+=r'''
#define CK(x) do {if(!(x)){printf("FAIL %d %s\n",__LINE__,#x);return 1;}}while(0)
constexpr int N=20003;
template<class T>void mem(T*&p,int n=N+32){if(cudaMallocManaged(&p,n*sizeof(T))!=cudaSuccess)std::abort();for(int k=0;k<n;++k)::new(static_cast<void*>(p+k)) T{};}
std::vector<double> snapshot(DeviceState*s){std::vector<double>v;for(int i=0;i<N;++i){
 SNAPSHOT
}return v;}
void initialize(DeviceState*s){
 s->nCells=5;s->nFaces=10;s->nInternalFaces=0;s->particleCapacity=N;s->maxFaceWalkHops=512;*s->particleCountDevice=N;
 for(int c=0;c<5;++c){s->cellPlaneStart[c]=2*c;s->cellPlaneCount[c]=2;s->cellLength[c]=2;
  for(int side=0;side<2;++side){int f=2*c+side;s->cellFaceId[f]=f;s->planeNx[f]=side?1:-1;s->planeNy[f]=s->planeNz[f]=0;s->planeD[f]=1;s->faceCx[f]=side?1:-1;s->faceCy[f]=s->faceCz[f]=0;s->cellFaceRestitution[f]=1;s->cellFaceKind[f]=1;s->cellFaceNeighbor[f]=-1;s->facePeriodicPair[f]=-1;
   if(c==2){s->cellFaceKind[f]=GPU_PERIODIC_FACE_KIND;s->cellFaceNeighbor[f]=c;s->facePeriodicPair[f]=2*c+1-side;s->facePeriodicDx[f]=side?2:-2;}
   if(c==3)s->cellFaceKind[f]=3;
   CONTACT_FACE
  }
 }
 for(int i=0;i<N;++i){int c=i%5;s->pStatus[i]=(i%29!=0);s->pCellId[i]=c;s->px[i]=c==0?0:.9;s->py[i]=s->pz[i]=0;s->pux[i]=s->puxOld[i]=1;s->puy[i]=s->puyOld[i]=0;s->puz[i]=s->puzOld[i]=0;s->pm[i]=1e-9;s->pd[i]=1e-4;s->pT[i]=2800;s->pTheta[i]=.125;s->pRng[i]=101u+i;
  CONTACT_PARTICLE
 }
 s->pCellId[N-1]=-1;
 CONTACT_STATE
}
int main(){setvbuf(stdout,nullptr,_IONBF,0);DeviceState*s;mem(s,1);mem(s->deviceState,1);
 ALLOC
 cudaDeviceProp prop{};CK(cudaGetDeviceProperties(&prop,0)==cudaSuccess);s->multiprocessorCount=prop.multiProcessorCount;s->fixedCellBlockThreads=64;s->fixedFaceBlockThreads=64;
 cudaFuncAttributes attr{};CK(cudaFuncGetAttributes(&attr,trackParticlesLocalFaceWalkKernel)==cudaSuccess);
 printf("RESOURCES regs=%d staticShared=%zu localBytes=%zu maxThreads=%d SM=%d\n",attr.numRegs,attr.sharedSizeBytes,attr.localSizeBytes,attr.maxThreadsPerBlock,prop.multiProcessorCount);
 bool independent=true;bool particleIndependent=true;
 for(int b:{32,64,128,256}){s->particleBlockThreads=b;s->reductionBlockThreads=b;s->particleCapacity=N;int occupancy=0;CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,trackParticlesLocalFaceWalkKernel,b,0)==cudaSuccess);int grids[2]={};int particleGrids[2]={};
  for(int heavy:{0,1}){s->csrHeavyReductionEnabled=heavy;CK(CONFIGURE(s)==0);grids[heavy]=SELECTED_GRID;particleGrids[heavy]=s->particleWorkGrid;printf("GEOMETRY B2=%d state=%d particleGrid=%d trackingGrid=%d trackingBlocksPerSM=%d\n",b,heavy,s->particleWorkGrid,grids[heavy],occupancy);}
  independent&=grids[0]==grids[1];particleIndependent&=particleGrids[0]==particleGrids[1];
  std::vector<double>reference;
  for(int grid:{1,3,prop.multiProcessorCount*occupancy,prop.multiProcessorCount*(occupancy+1)}){
   initialize(s);CK(syncDeviceState(s,"tracking fixture")==0);trackParticlesLocalFaceWalkKernel<<<grid,b>>>(s->deviceState,.2);CK(cudaDeviceSynchronize()==cudaSuccess);
   const auto actual=snapshot(s);if(reference.empty())reference=actual;else CK(reference==actual);
   for(int q=1;q<N-1;++q){if(q%29==0)continue;int c=q%5;if(c==0){CK(s->pStatus[q]==1);CK(fabs(double(s->px[q])-.2)<2e-6);}if(c==1){CK(s->pStatus[q]==1);CK(fabs(double(s->px[q])-.9)<2e-6);CK(s->pux[q]==-1);}if(c==2){CK(s->pStatus[q]==1);CK(fabs(double(s->px[q])+.9)<2e-6);}if(c==3)CK(s->pStatus[q]==0);CONTACT_CHECK}
   CK(s->pStatus[N-1]==0);printf("TRANSPORT B2=%d grid=%d N=%d PASS free/reflection/periodic/outflow/tail/contact\n",b,grid,N);
  }
 }
 printf("INDEPENDENT_TRACKING_GRID %s\n",independent?"PASS":"FAIL");
 if(REQUIRE_INDEPENDENT&&!independent)return 42;
 printf("INDEPENDENT_PARTICLE_GRID %s\n",particleIndependent?"PASS":"FAIL");
 if(REQUIRE_PARTICLE_INDEPENDENT&&!particleIndependent)return 43;
 return 0;
}
'''
code=code.replace('ALLOC','\n'.join('mem(s->'+n+');' for _,n in fields))
numeric=['px','py','pz','pux','puy','puz','puxOld','puyOld','puzOld','pm','pd','pT','pTheta','pRng','pCellId','pStatus']
thermal=a.app!='gasUGKP'
if thermal:numeric+=['pStuck','pStuckFaceId','pDepositionArea','pContactDuration','pContactMaximumArea','pContactPeakFraction']+(['pContactAge'] if a.app=='CHT' else [])
code=code.replace('SNAPSHOT','\n'.join('v.push_back(double(s->'+f+'[i]));' for f in numeric))
code=code.replace('CONFIGURE','configureParticleLaunchGeometry' if a.app=='gasUGKP' else 'configureLaunchOccupancy')
code=code.replace('SELECTED_GRID','s->trackingWorkGrid' if 'int trackingWorkGrid' in t else 's->particleWorkGrid')
code=code.replace('REQUIRE_INDEPENDENT','1' if a.require_independent else '0')
code=code.replace('REQUIRE_PARTICLE_INDEPENDENT','1' if a.require_particle_independent else '0')
repl={
 'CONTACT_FACE':'if(c==4){s->cellFaceKind[f]=5;s->particleStuckCandidateMask[f]=Foam::gpuThermal::particleWallReboundContact;s->gasBoundaryUx[f]=s->gasBoundaryUy[f]=s->gasBoundaryUz[f]=0;}',
 'CONTACT_PARTICLE':'s->pStuck[i]=0;s->pStuckFaceId[i]=-1;s->pContactDuration[i]=s->pContactMaximumArea[i]=s->pContactPeakFraction[i]=s->pDepositionArea[i]=0;'+('s->pContactAge[i]=0;' if a.app=='CHT' else ''),
 'CONTACT_STATE':'s->particleStuckModelConfigured=1;s->sommerfeldThreshold=1;s->particleWallContactAngleCosine=.5;',
 'CONTACT_CHECK':'if(c==4){CK(s->pStatus[q]==1);CK(s->pStuck[q]==Foam::gpuThermal::particleWallTransientRebound);CK(s->pux[q]==0);CK(s->puxOld[q]==-1);CK(s->pContactDuration[q]>0);CK(s->pTheta[q]==0);}'
}
for k,v in repl.items():code=code.replace(k,v if thermal else '')
f=out/'tracking.cu';f.write_text(code);exe=out/'tracking'
cmd=[str(Path(os.environ.get('CUDA_HOME','/usr/local/cuda'))/'bin/nvcc'),'-std=c++17','-O3','-arch='+os.environ.get('UGKWP_CUDA_ARCH','sm_89'),'--fmad='+('false' if a.app=='CHT' else 'true'),'-DUGKWP_GPU_REAL_BITS='+str(a.bits),'-I'+str(src),'-I'+str(root/'common'),'-I'+str(root/'applications'/a.app/'gpu'),str(f),'-o',str(exe)]
if a.app=='CHT':cmd.append(str(src/'GpuWallEnergy64.cu'))
with (out/'build.log').open('w') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
record={'app':a.app,'bits':a.bits,'command':cmd,'fixture_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'build_exit':result.returncode}
if not result.returncode:
 record['binary_sha256']=hashlib.sha256(exe.read_bytes()).hexdigest();q=subprocess.run([str(exe)],capture_output=True,text=True);(out/'run.log').write_text(q.stdout+q.stderr);record['run_exit']=q.returncode;print(q.stdout+q.stderr)
else:print((out/'build.log').read_text()[-6000:])
(out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
raise SystemExit(record.get('run_exit',result.returncode))
