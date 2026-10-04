import os
from pathlib import Path
import re,subprocess,sys,json,os
W=Path(sys.argv[2]).resolve();W.mkdir(parents=True,exist_ok=True);app,bits=sys.argv[3],int(sys.argv[4]);kind='current';r=Path(sys.argv[1]).resolve();leaf='gpu' if app=='CHT' else 'private_backend';src=r/'applications'/app/leaf/'GpuResidentStrict.cu'

for sub in ['02-tests','03-build','logs']:(W/sub).mkdir(exist_ok=True)
t=src.read_text().replace('#include "GpuAutomaticCsrScheduleFields.inl"', (r/"common/GpuAutomaticCsrScheduleFields.inl").read_text() if (r/"common/GpuAutomaticCsrScheduleFields.inl").is_file() else "");a=t.index('struct DeviceState');b=t.index('\n};',a);fields=re.findall(r'^\s*((?:unsigned\s+)?(?:long long|char)|double|float|int|GpuReal|GpuTime|GpuWallEnergy)\*\s+(\w+)\s*=',t[a:b],re.M)
needed=set('pStatus pCellId pRng pm pux puy puz pTheta pd pT pStuck pContactAge pStuckFaceId pDepositionArea pContactDuration pContactMaximumArea pContactPeakFraction compactPStatus compactCountDevice cellParticleOffset sortedParticleIndex preBaseCellOffset csrCellTaskOffset csrCellTaskCount csrHeavyPartials csrMultiTaskCellList csrHeavyTaskCount csrHeavyCellCount csrHeavyTaskCursor poissonCellCollisionProbability particleCountDevice thetaDragAlpha momRhoP momRhoUPx momRhoUPy momRhoUPz momRhoEP momRhoPD'.split())
needed.update(n for _,n in fields if n.startswith('poissonPool') or n.startswith('poolThermal'))
text='#include "GpuResidentStrict.cu"\n#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n#include <cmath>\n#include <vector>\n'
if app=='CHT' and bits==32:text+='using namespace ugkwpCudaFp32;\n'
text+='using Real='+('GpuReal' if app=='CHT' else 'double')+';\n'
text+=r'''
#define CHECK(x) do{if(!(x)){printf("FAIL %d %s\n",__LINE__,#x);return 1;}}while(0)
template<class T>void mem(T*&p,int n){if(cudaMallocManaged(&p,sizeof(T)*n)!=cudaSuccess)std::abort();for(int i=0;i<n;++i)p[i]=T{};}
__global__ void initParticles(DeviceState*sp,int contact){auto&s=*sp;for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<s.particleCapacity;i+=blockDim.x*gridDim.x){
 s.pStatus[i]=(i%11==0?0:1);s.pCellId[i]=(i<s.particleCapacity*3/4?0:1);s.pRng[i]=123456789ULL+i*333ULL;
 s.pm[i]=1+(i%3)*.25;s.pux[i]=(i%17-8)*.125;s.puy[i]=(i%7-3)*.25;s.puz[i]=(i%5-2)*.25;s.pTheta[i]=.75;s.pd[i]=.001;s.pT[i]=1000;
 THERMAL_INIT
 }}
__global__ void cacheProbability(DeviceState*sp,double dt){auto&s=*sp;int c=threadIdx.x;if(c<s.nCells){ CACHE_PROBABILITY }}
__global__ void sampleOneLaunch(DeviceState*sp){for(int i=threadIdx.x;i<sp->particleCapacity;i+=blockDim.x)sampleOnePoissonPoolParticle(*sp,i,false);}
__global__ void rejectedThetaProbe(DeviceState*sp,int level){
 auto&s=*sp;Real mass=0,mx=0,my=0,mz=0,energy=0,d=0,d2=0,count=0;
 const Real p=s.thetaDragAlpha[0];
 if(level==1)accumulateOnePoolParticle<true>(s,0,1,p,mass,mx,my,mz,energy,d,d2,count);
 else accumulateOnePoolParticle<true>(s,0,1,p,mass,mx,my,mz,energy,d,d2,count);
}
int main(int argc,char**argv){setvbuf(stdout,nullptr,_IONBF,0);const int N=argc>1&&strcmp(argv[1],"late-probe")?atoi(argv[1]):129;bool bench=argc>2;
DeviceState*s;mem(s,1);s->deviceState=s;s->particleCapacity=N;s->nCells=2;s->rhoSolid=1000;s->epsSMin=1e-12;s->thetaMin=1e-12;s->particleDiameterFallback=.001;s->particleDiameterMin=1e-6;s->particleDiameterMax=1.;s->injectionParcelMass=1.;s->csrHeavyWorkerGrid=24;s->multiprocessorCount=24;
'''
text+='\n'.join('mem(s->'+n+', '+('8*(N+4)' if n=='csrHeavyPartials' else 'N+32')+');' for typ,n in fields if n in needed)
text+=r'''
mem(s->csrReductionTasks,N+4);*s->particleCountDevice=N;s->cellParticleOffset[0]=0;s->cellParticleOffset[1]=N*3/4;s->cellParticleOffset[2]=N;
for(int i=0;i<N;++i)s->sortedParticleIndex[i]=i;
for(int c=0;c<2;++c){s->momRhoP[c]=20;s->momRhoEP[c]=30;s->momRhoPD[c]=.02;s->thetaDragAlpha[c]=1;}
if(argc>1&&!strcmp(argv[1],"late-probe")){
 initParticles<<<1,32>>>(s,0);CHECK(cudaDeviceSynchronize()==cudaSuccess);
 s->thetaDragAlpha[0]=1e-15;
 auto*theta=s->pTheta;s->pTheta=nullptr;
 for(int level=1;level<=2;++level){
  rejectedThetaProbe<<<1,1>>>(s,level);
  CHECK(cudaDeviceSynchronize()==cudaSuccess);
  CHECK(s->pStatus[1]==1);
 }
 accumulateParticlePoolAtomicKernel<true><<<1,32>>>(s,0.0);
 CHECK(cudaDeviceSynchronize()==cudaSuccess);
 CHECK(s->pStatus[1]==1);
 s->pTheta=theta;puts("PASS rejected particles do not read theta in L1, L2 or atomic fallback");return 0;
}
const double dt=.005;const double tol=sizeof(Real)==4?2e-5:2e-12;
for(int poisson=0;poisson<2;++poisson)for(int contact=0;contact<(bench?1:CONTACT_MODES);++contact)for(int block:{32,64,128,256})for(int heavy=0;heavy<2;++heavy){
 if(bench && (block!=64 || !poisson))continue;
 s->csrHeavyReductionEnabled=heavy;int nt=0,nh=0;int tile=17;
 if(bench){DeviceState config=*s;config.reductionBlockThreads=block;config.particleBlockThreads=64;CHECK(CONFIGURE(&config)==0);CHECK(cudaDeviceSynchronize()==cudaSuccess);long long concurrency=1LL*block*config.multiprocessorCount*config.RESIDENT_BLOCKS;CHECK(concurrency>0);tile=block*std::max(1LL,(N+concurrency-1)/concurrency);printf("GEOMETRY N=%d heavy=%d block=%d tile=%d resident=%d workerGrid=%d\n",N,heavy,block,tile,config.RESIDENT_BLOCKS,config.csrHeavyWorkerGrid);}

 for(int c=0;c<2;++c){s->csrCellTaskOffset[c]=nt;s->csrCellTaskCount[c]=0;
  for(int start=s->cellParticleOffset[c];start<s->cellParticleOffset[c+1];start+=tile){s->csrReductionTasks[nt++]={c,start,std::min(start+tile,s->cellParticleOffset[c+1]),0};++s->csrCellTaskCount[c];}
  if(s->csrCellTaskCount[c]>1)s->csrMultiTaskCellList[nh++]=c;
 }
 s->csrCellTaskOffset[2]=nt;*s->csrHeavyTaskCount=nt;*s->csrHeavyCellCount=nh;DIR_KIND
 DeviceState host=*s;
 auto launch=[&](){clearPoissonThermalPoolKernel<<<1,32>>>(s);
  if(heavy){if(launchCsrSegmentedPoolReduction(&host,dt,poisson,block))std::abort();}
  else if(poisson) { S1_LAUNCH }
  else accumulateParticlePoolAtomicKernel<false><<<24,block>>>(s,dt);
 };
 initParticles<<<24,128>>>(s,contact);launch();CHECK(cudaDeviceSynchronize()==cudaSuccess);
 unsigned long long hash=1469598103934665603ULL;int selected=0;
 for(int c=0;c<2;++c){double sum[5]={};int count=0;
  for(int i=s->cellParticleOffset[c];i<s->cellParticleOffset[c+1];++i){hash=(hash^s->pRng[i])*1099511628211ULL;hash=(hash^s->pStatus[i])*1099511628211ULL;
   if(s->pStatus[i]!=2)continue;++count;++selected;double m=s->pm[i],ux=s->pux[i],uy=s->puy[i],uz=s->puz[i];double theta=THETA_VALUE;
   sum[0]+=m;sum[1]+=m*ux;sum[2]+=m*uy;sum[3]+=m*uz;sum[4]+=m*(.5*(ux*ux+uy*uy+uz*uz)+1.5*theta);
  }
  CHECK(count==s->poolThermalCount[c]);double got[5]={s->poissonPoolMass[c],s->poissonPoolMomX[c],s->poissonPoolMomY[c],s->poissonPoolMomZ[c],s->poissonPoolEnergy[c]};for(int j=0;j<5;++j)CHECK(fabs(got[j]-sum[j])<=tol*(1+fabs(sum[j])));
 }
 if(!bench){
  *s->compactCountDevice=0;preparePoissonPoolSamplingKernel<<<1,32>>>(s);sampleOneLaunch<<<1,64>>>(s);CHECK(cudaDeviceSynchronize()==cudaSuccess);
  for(int i=0;i<N;++i){CHECK(std::isfinite(double(s->pux[i])));CONTACT_CHECK;hash=(hash^s->pRng[i])*1099511628211ULL;}
  printf("STATE poisson=%d contact=%d block=%d heavy=%d selected=%d hash=%llu\n",poisson,contact,block,heavy,selected,hash);
 }else{
  for(int rep=0;rep<5;++rep){initParticles<<<24,128>>>(s,0);for(int j=0;j<30;++j)launch();CHECK(cudaDeviceSynchronize()==cudaSuccess);cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);for(int j=0;j<200;++j)launch();cudaEventRecord(b);cudaEventSynchronize(b);float ms=0;cudaEventElapsedTime(&ms,a,b);printf("TIME N=%d heavy=%d rep=%d ms=%.9g\n",N,heavy,rep,ms/200);cudaEventDestroy(a);cudaEventDestroy(b);}
 }
}
puts("PASS collision pool CPU moments, selected IDs/RNG and contact metadata");}
'''
thermal=app!='gasUGKP'
text=text.replace('CONFIGURE','configureLaunchOccupancy' if thermal else 'configureParticleLaunchGeometry').replace('RESIDENT_BLOCKS','lightResidentBlocksPerSm' if thermal else 'lightBlocksPerSm')
text=text.replace('CONTACT_MODES','2' if thermal else '1')
text=text.replace('THERMAL_INIT', ('s.pStuck[i]=(contact&&i%3==0?Foam::gpuThermal::particleWallTransientDeposit:0);s.pStuckFaceId[i]=7;s.pContactDuration[i]=.5;' + ('s.pContactAge[i]=.125;' if app=='CHT' else 'if(s.pStuck[i])s.pTheta[i]=.125;')) if thermal else '')
text=text.replace('THETA_VALUE','(s->pStuck[i]?0.:double(s->pTheta[i]))' if thermal else 'double(s->pTheta[i])')
text=text.replace('CONTACT_CHECK', ('if(s->pStatus[i]==2&&s->pStuck[i])CHECK('+('s->pContactAge[i]' if app=='CHT' else 's->pTheta[i]')+'==.125)') if thermal else '')
text=text.replace('DIR_KIND','s->csrReductionDirectoryKind=0;' if not thermal else '')
text=text.replace('CACHE_PROBABILITY','s.poissonCellCollisionProbability[c]=poissonCollisionProbabilityForCell(s,c,dt);' if not thermal else '')
text=text.replace('S1_LAUNCH','accumulatePoissonPoolParticlesByCellKernel'+('<false>' if not thermal else '')+'<<<2,block,8*block*sizeof(Real)>>>(s,dt);')
# gas lacks the selected-stuck counter; allocating it in the source would not compile.
if not thermal:text=text.replace('*s->compactCountDevice=0;','').replace('clearPoissonThermalPoolKernel<<<1,32>>>(s);','clearPoissonThermalPoolKernel<<<1,32>>>(s,dt);')
f=W/'02-tests'/f'collision-{kind}-{app}-{bits}.cu';f.write_text(text);exe=W/'03-build'/f'collision-{kind}-{app}-{bits}'
cmd=[str(Path(os.environ.get('CUDA_HOME','/usr/local/cuda'))/'bin/nvcc'),'-std=c++17','-O3','-arch='+os.environ.get('UGKWP_CUDA_ARCH','sm_89'),'--fmad='+('false' if app=='CHT' else 'true'),'-DUGKWP_GPU_REAL_BITS='+str(bits),'-I'+str(src.parent),'-I'+str(r/'common'),'-I'+str(r/'applications'/app/'gpu'),str(f),'-o',str(exe)]
if app=='CHT':cmd.append(str(r/'applications/CHT/gpu/GpuWallEnergy64.cu'))
log=W/'logs'/f'collision-{kind}-{app}-{bits}-build.log'
with log.open('w') as o:q=subprocess.run(cmd,stdout=o,stderr=subprocess.STDOUT)
print('BUILD',kind,app,bits,q.returncode,flush=True)
if q.returncode:print(log.read_text()[-5000:]);raise SystemExit(q.returncode)
q=subprocess.run([str(exe)],capture_output=True,text=True);(W/'logs'/f'collision-{kind}-{app}-{bits}-run.log').write_text(q.stdout+q.stderr);print('RUN_RETURN',q.returncode,q.stdout+q.stderr,flush=True)
probe=subprocess.run([str(exe),'late-probe'],capture_output=True,text=True)
print('LATE_PROBE_RETURN',probe.returncode,probe.stdout+probe.stderr,flush=True)
if probe.returncode:raise SystemExit(probe.returncode)
if q.returncode==0:
 groups={}
 for line in q.stdout.splitlines():
  if not line.startswith('STATE '):continue
  values=dict(item.split('=',1) for item in line.split()[1:]);key=(values['poisson'],values['contact']);groups.setdefault(key,set()).add((values['selected'],values['hash']))
 assert groups and all(len(v)==1 for v in groups.values()),groups
raise SystemExit(q.returncode)
