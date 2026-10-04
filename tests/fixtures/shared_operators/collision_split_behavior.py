import os
from pathlib import Path
import re,subprocess,sys,json,os
W=Path(sys.argv[2]).resolve();W.mkdir(parents=True,exist_ok=True);app,bits=sys.argv[3],int(sys.argv[4]);kind='current';r=Path(sys.argv[1]).resolve();leaf='gpu' if app=='CHT' else 'private_backend';src=r/'applications'/app/leaf/'GpuResidentStrict.cu'

for sub in ['02-tests','03-build','logs']:(W/sub).mkdir(exist_ok=True)
t=src.read_text().replace('#include "GpuAutomaticCsrScheduleFields.inl"', (r/"common/GpuAutomaticCsrScheduleFields.inl").read_text() if (r/"common/GpuAutomaticCsrScheduleFields.inl").is_file() else "");a=t.index('struct DeviceState');b=t.index('\n};',a);fields=re.findall(r'^\s*((?:unsigned\s+)?(?:long long|char)|double|float|int|GpuReal|GpuTime|GpuWallEnergy)\*\s+(\w+)\s*=',t[a:b],re.M)
needed=set('pStatus pCellId pRng pm pux puy puz pTheta pd pT pStuck pContactAge pStuckFaceId pDepositionArea pContactDuration pContactMaximumArea pContactPeakFraction compactPStatus compactCountDevice cellParticleOffset sortedParticleIndex preBaseCellOffset csrCellTaskOffset csrCellTaskCount csrHeavyPartials csrMultiTaskCellList csrHeavyTaskCount csrHeavyCellCount csrHeavyTaskCursor poissonCellCollisionProbability particleCountDevice thetaDragAlpha momRhoP momRhoUPx momRhoUPy momRhoUPz momRhoEP momRhoPD'.split())
needed.update(n for _,n in fields if n.startswith('poissonPool') or n.startswith('poolThermal'))
text='#include "GpuResidentStrict.cu"\n#include <cstdio>\n#include <new>\n#include <cstdlib>\n#include <cmath>\n#include <vector>\n#include <string>\n'
if app=='CHT' and bits==32:text+='using namespace ugkwpCudaFp32;\n'
text+='using Real='+('GpuReal' if app=='CHT' else 'double')+';\n'
text+=r'''
#define CHECK(x) do{if(!(x)){printf("FAIL %d %s\n",__LINE__,#x);return 1;}}while(0)
template<class T>void mem(T*&p,int n){if(cudaMallocManaged(&p,sizeof(T)*n)!=cudaSuccess)std::abort();for(int i=0;i<n;++i)::new(static_cast<void*>(p+i)) T{};}
__global__ void initParticles(DeviceState*sp,int contact){auto&s=*sp;for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<s.particleCapacity;i+=blockDim.x*gridDim.x){
 s.pStatus[i]=(i%11==0?0:1);s.pCellId[i]=(i<s.particleCapacity*3/4?0:1);s.pRng[i]=123456789ULL+i*333ULL;
 s.pm[i]=1+(i%3)*.25;s.pux[i]=(i%17-8)*.125;s.puy[i]=(i%7-3)*.25;s.puz[i]=(i%5-2)*.25;s.pTheta[i]=.75;s.pd[i]=.001;s.pT[i]=1000;
 THERMAL_INIT
 }}
__global__ void cacheProbability(DeviceState*sp,double dt){auto&s=*sp;int c=threadIdx.x;if(c<s.nCells){ CACHE_PROBABILITY }}
__global__ void sampleOneLaunch(DeviceState*sp){for(int i=threadIdx.x;i<sp->particleCapacity;i+=blockDim.x)sampleOnePoissonPoolParticle(*sp,i,false);}
int main(int argc,char**argv){setvbuf(stdout,nullptr,_IONBF,0);const int N=argc>1?atoi(argv[1]):129;bool bench=argc>2&&std::string(argv[2])!="zero";
DeviceState*s;mem(s,1);s->deviceState=s;s->particleCapacity=N;s->nCells=2;s->rhoSolid=1000;s->epsSMin=1e-12;s->thetaMin=1e-12;s->particleDiameterFallback=.001;s->particleDiameterMin=1e-6;s->particleDiameterMax=1.;s->injectionParcelMass=1.;s->csrHeavyWorkerGrid=24;s->multiprocessorCount=24;
'''
text+='\n'.join('mem(s->'+n+', '+('8*(N+4)' if n=='csrHeavyPartials' else 'N+32')+');' for typ,n in fields if n in needed)
text+=r'''
mem(s->csrReductionTasks,N+4);*s->particleCountDevice=N;s->cellParticleOffset[0]=0;s->cellParticleOffset[1]=N*3/4;s->cellParticleOffset[2]=N;
for(int i=0;i<N;++i)s->sortedParticleIndex[i]=i;
for(int c=0;c<2;++c){s->momRhoP[c]=20;s->momRhoEP[c]=30;s->momRhoPD[c]=.02;s->thetaDragAlpha[c]=1;}
const double dt=(argc>2&&std::string(argv[2])=="zero")?0.:.005;const double tol=sizeof(Real)==4?2e-5:2e-12;
for(int layout=0;layout<5;++layout)for(int poisson=0;poisson<2;++poisson)for(int contact=0;contact<(bench?1:CONTACT_MODES);++contact)for(int block:{32,64,128,256})for(int heavy=0;heavy<2;++heavy){
 if(layout && !poisson)continue;
 if(layout==4 && heavy)continue;
 if(bench && (block!=64 || !poisson))continue;
 s->cellParticleOffset[0]=0;s->cellParticleOffset[1]=N*3/4;s->cellParticleOffset[2]=N;for(int i=0;i<N;i++)s->sortedParticleIndex[i]=i;
 for(int c=0;c<3;c++)s->preBaseCellOffset[c]=0;
 if(layout==1){s->preBaseCellOffset[1]=N*3/4;s->preBaseCellOffset[2]=N;for(int c=0;c<3;c++)s->cellParticleOffset[c]=0;}
 if(layout==3){s->preBaseCellOffset[1]=N*3/4;s->preBaseCellOffset[2]=N*3/4;s->cellParticleOffset[1]=0;s->cellParticleOffset[2]=N-N*3/4;for(int j=0;j<N-N*3/4;j++)s->sortedParticleIndex[j]=N*3/4+j;}
 s->csrHeavyReductionEnabled=heavy;int nt=0,nh=0;int tile=17;
 if(bench){DeviceState config=*s;config.reductionBlockThreads=block;config.particleBlockThreads=64;CHECK(CONFIGURE(&config)==0);CHECK(cudaDeviceSynchronize()==cudaSuccess);long long concurrency=1LL*block*config.multiprocessorCount*config.RESIDENT_BLOCKS;CHECK(concurrency>0);tile=block*std::max(1LL,(N+concurrency-1)/concurrency);printf("GEOMETRY N=%d heavy=%d block=%d tile=%d resident=%d workerGrid=%d\n",N,heavy,block,tile,config.RESIDENT_BLOCKS,config.csrHeavyWorkerGrid);}

 for(int c=0;c<2;++c){s->csrCellTaskOffset[c]=nt;s->csrCellTaskCount[c]=0;
  const int begin=layout?0:s->cellParticleOffset[c];
  const int end=layout?(s->preBaseCellOffset[c+1]-s->preBaseCellOffset[c]+s->cellParticleOffset[c+1]-s->cellParticleOffset[c]):s->cellParticleOffset[c+1];
  for(int start=begin;start<end;start+=tile){s->csrReductionTasks[nt++]={c,start,std::min(start+tile,end),static_cast<int>(layout?CsrReductionTaskSource::splitLogical:CsrReductionTaskSource::fullIndexed)};++s->csrCellTaskCount[c];}
  if(s->csrCellTaskCount[c]>1)s->csrMultiTaskCellList[nh++]=c;
 }
 s->csrCellTaskOffset[2]=nt;*s->csrHeavyTaskCount=nt;*s->csrHeavyCellCount=nh;DIR_KIND
 DeviceState host=*s;
 auto launch=[&](){clearPoissonThermalPoolKernel<<<1,32>>>(s);
  if(heavy){if(launchCsrSegmentedPoolReduction(&host,dt,poisson,block))std::abort();}
  else if(poisson) { if(layout==4){accumulateParticlePoolAtomicKernel<true><<<24,block>>>(s,dt);}else if(!layout){S1_LAUNCH}else{SPLIT_LAUNCH} }
  else accumulateParticlePoolAtomicKernel<false><<<24,block>>>(s,dt);
 };
 initParticles<<<24,128>>>(s,contact);launch();CHECK(cudaDeviceSynchronize()==cudaSuccess);
 unsigned long long hash=1469598103934665603ULL;int selected=0;
 for(int c=0;c<2;++c){double sum[7]={};int count=0;
  for(int i=0;i<N;++i){if(s->pCellId[i]!=c)continue;hash=(hash^s->pRng[i])*1099511628211ULL;hash=(hash^s->pStatus[i])*1099511628211ULL;
   if(s->pStatus[i]!=2)continue;++count;++selected;double m=s->pm[i],ux=s->pux[i],uy=s->puy[i],uz=s->puz[i];double theta=THETA_VALUE;
   sum[0]+=m;sum[1]+=m*ux;sum[2]+=m*uy;sum[3]+=m*uz;sum[4]+=m*(.5*(ux*ux+uy*uy+uz*uz)+1.5*theta);sum[5]+=m*s->pd[i];sum[6]+=m*s->pd[i]*s->pd[i];
  }
  CHECK(count==s->poolThermalCount[c]);double got[7]={s->poissonPoolMass[c],s->poissonPoolMomX[c],s->poissonPoolMomY[c],s->poissonPoolMomZ[c],s->poissonPoolEnergy[c],s->poissonPoolDiameter[c],s->poissonPoolDiameter2[c]};for(int j=0;j<7;++j)CHECK(fabs(got[j]-sum[j])<=tol*(1+fabs(sum[j])));
 }
 if(!bench){
  *s->compactCountDevice=0;preparePoissonPoolSamplingKernel<<<1,32>>>(s);sampleOneLaunch<<<1,64>>>(s);CHECK(cudaDeviceSynchronize()==cudaSuccess);
  for(int i=0;i<N;++i){CHECK(std::isfinite(double(s->pux[i])));CONTACT_CHECK;hash=(hash^s->pRng[i])*1099511628211ULL;}
  printf("STATE layout=%d poisson=%d contact=%d block=%d heavy=%d selected=%d hash=%llu\n",layout,poisson,contact,block,heavy,selected,hash);
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
text=text.replace('DIR_KIND','s->csrReductionDirectoryKind=layout?1:0;s->useSplitPreDirectory=layout!=0;' if not thermal else 's->splitPreDirectoryActive=layout!=0;')
text=text.replace('CACHE_PROBABILITY','s.poissonCellCollisionProbability[c]=poissonCollisionProbabilityForCell(s,c,dt);' if not thermal else '')
text=text.replace('S1_LAUNCH','accumulatePoissonPoolParticlesByCellKernel'+('<false>' if not thermal else '')+'<<<2,block,8*block*sizeof(Real)>>>(s,dt);')
split_name='accumulatePoissonPoolSplitSegmentByCellKernel' if not thermal else 'accumulatePoissonPoolSplitSegmentKernel'
flags=', false' if not thermal else ''
text=text.replace('SPLIT_LAUNCH',split_name+'<true, false'+flags+'><<<2,block,8*block*sizeof(Real)>>>(s,dt);'+split_name+'<false, true'+flags+'><<<2,block,8*block*sizeof(Real)>>>(s,dt);')
# gas lacks the selected-stuck counter; allocating it in the source would not compile.
if not thermal:text=text.replace('*s->compactCountDevice=0;','').replace('clearPoissonThermalPoolKernel<<<1,32>>>(s);','clearPoissonThermalPoolKernel<<<1,32>>>(s,dt);')
f=W/'02-tests'/f'collision-{kind}-{app}-{bits}.cu';f.write_text(text);exe=W/'03-build'/f'collision-{kind}-{app}-{bits}'
cmd=[str(Path(os.environ.get('CUDA_HOME','/usr/local/cuda'))/'bin/nvcc'),'-std=c++17','-O3','-arch='+os.environ.get('UGKWP_CUDA_ARCH','sm_89'),'--fmad='+('false' if app=='CHT' else 'true'),'-DUGKWP_GPU_REAL_BITS='+str(bits),'-I'+str(src.parent),'-I'+str(r/'common'),'-I'+str(r/'applications'/app/'gpu'),str(f),'-o',str(exe)]
if app=='CHT':cmd.append(str(r/'applications/CHT/gpu/GpuWallEnergy64.cu'))
log=W/'logs'/f'collision-{kind}-{app}-{bits}-build.log'
with log.open('w') as o:q=subprocess.run(cmd,stdout=o,stderr=subprocess.STDOUT)
print('BUILD',kind,app,bits,q.returncode,flush=True)
if q.returncode:print(log.read_text()[-5000:]);raise SystemExit(q.returncode)
for arguments, suffix in [([], 'normal'), (['129', 'zero'], 'zero')]:
 q=subprocess.run([str(exe), *arguments],capture_output=True,text=True)
 (W/'logs'/f'collision-{kind}-{app}-{bits}-{suffix}-run.log').write_text(q.stdout+q.stderr)
 print('RUN_RETURN',suffix,q.returncode,q.stdout+q.stderr,flush=True)
 if q.returncode:raise SystemExit(q.returncode)
 groups={}
 for line in q.stdout.splitlines():
  if not line.startswith('STATE '):continue
  values=dict(item.split('=',1) for item in line.split()[1:]);key=(values['poisson'],values['contact']);groups.setdefault(key,set()).add((values['selected'],values['hash']))
 assert groups and all(len(v)==1 for v in groups.values()),groups
