#include "GpuResidentStrict.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
using Real=double;

#define CHECK(x) do{if(!(x)){printf("FAIL %d %s\n",__LINE__,#x);return 1;}}while(0)
template<class T>void mem(T*&p,int n){if(cudaMallocManaged(&p,sizeof(T)*n)!=cudaSuccess)std::abort();for(int i=0;i<n;++i)p[i]=T{};}
__global__ void initParticles(DeviceState*sp,int contact){auto&s=*sp;for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<s.particleCapacity;i+=blockDim.x*gridDim.x){
 s.pStatus[i]=(i%11==0?0:1);s.pCellId[i]=(i<s.particleCapacity*3/4?0:1);s.pRng[i]=123456789ULL+i*333ULL;
 s.pm[i]=1+(i%3)*.25;s.pux[i]=(i%17-8)*.125;s.puy[i]=(i%7-3)*.25;s.puz[i]=(i%5-2)*.25;s.pTheta[i]=.75;s.pd[i]=.001;s.pT[i]=1000;
 
 }}
__global__ void cacheProbability(DeviceState*sp,double dt){auto&s=*sp;int c=threadIdx.x;if(c<s.nCells){ s.poissonCellCollisionProbability[c]=poissonCollisionProbabilityForCell(s,c,dt); }}
__global__ void sampleOneLaunch(DeviceState*sp){for(int i=threadIdx.x;i<sp->particleCapacity;i+=blockDim.x)sampleOnePoissonPoolParticle(*sp,i,false);}
__global__ void rejectedThetaProbe(DeviceState*sp,int level){
 auto&s=*sp;Real mass=0,mx=0,my=0,mz=0,energy=0,d=0,d2=0,count=0;
 const Real p=s.thetaDragAlpha[0];
 if(level==1)accumulateOnePoolParticle<true>(s,0,1,p,mass,mx,my,mz,energy,d,d2,count);
 else accumulateOnePoolParticle<true>(s,0,1,p,mass,mx,my,mz,energy,d,d2,count);
}
int main(int argc,char**argv){setvbuf(stdout,nullptr,_IONBF,0);const int N=argc>1&&strcmp(argv[1],"late-probe")?atoi(argv[1]):129;bool bench=argc>2;
DeviceState*s;mem(s,1);s->deviceState=s;s->particleCapacity=N;s->nCells=2;s->rhoSolid=1000;s->epsSMin=1e-12;s->thetaMin=1e-12;s->particleDiameterFallback=.001;s->particleDiameterMin=1e-6;s->particleDiameterMax=1.;s->injectionParcelMass=1.;s->csrHeavyWorkerGrid=24;s->multiprocessorCount=24;
mem(s->csrHeavyTaskCount, N+32);
mem(s->csrHeavyCellCount, N+32);
mem(s->csrHeavyTaskCursor, N+32);
mem(s->particleCountDevice, N+32);
mem(s->thetaDragAlpha, N+32);
mem(s->momRhoP, N+32);
mem(s->momRhoUPx, N+32);
mem(s->momRhoUPy, N+32);
mem(s->momRhoUPz, N+32);
mem(s->momRhoEP, N+32);
mem(s->momRhoPD, N+32);
mem(s->poolThermalCount, N+32);
mem(s->poolThermalSumUx, N+32);
mem(s->poolThermalSumUy, N+32);
mem(s->poolThermalSumUz, N+32);
mem(s->poolThermalSumU2, N+32);
mem(s->poissonPoolSampleTargetCount, N+32);
mem(s->poissonPoolMass, N+32);
mem(s->poissonPoolMomX, N+32);
mem(s->poissonPoolMomY, N+32);
mem(s->poissonPoolMomZ, N+32);
mem(s->poissonPoolEnergy, N+32);
mem(s->poissonPoolDiameter, N+32);
mem(s->poissonPoolDiameter2, N+32);
mem(s->pux, N+32);
mem(s->puy, N+32);
mem(s->puz, N+32);
mem(s->pT, N+32);
mem(s->pTheta, N+32);
mem(s->pd, N+32);
mem(s->pm, N+32);
mem(s->pCellId, N+32);
mem(s->pStatus, N+32);
mem(s->pRng, N+32);
mem(s->sortedParticleIndex, N+32);
mem(s->cellParticleOffset, N+32);
mem(s->preBaseCellOffset, N+32);
mem(s->compactCountDevice, N+32);
mem(s->csrCellTaskCount, N+32);
mem(s->csrCellTaskOffset, N+32);
mem(s->csrMultiTaskCellList, N+32);
mem(s->csrHeavyPartials, 8*(N+4));
mem(s->compactPStatus, N+32);
mem(s->poissonCellCollisionProbability, N+32);
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
const double dt=.005;const double tol=2e-12;
for(int poisson=0;poisson<2;++poisson)for(int contact=0;contact<(bench?1:1);++contact)for(int block:{32,64,128,256})for(int heavy=0;heavy<2;++heavy){
 if(bench && (block!=64 || !poisson))continue;
 s->csrHeavyReductionEnabled=heavy;int nt=0,nh=0;int tile=17;
 if(bench){DeviceState config=*s;config.reductionBlockThreads=block;config.particleBlockThreads=64;CHECK(configureParticleLaunchGeometry(&config)==0);CHECK(cudaDeviceSynchronize()==cudaSuccess);long long concurrency=1LL*block*config.multiprocessorCount*config.lightBlocksPerSm;CHECK(concurrency>0);tile=block*std::max(1LL,(N+concurrency-1)/concurrency);printf("GEOMETRY N=%d heavy=%d block=%d tile=%d resident=%d workerGrid=%d\n",N,heavy,block,tile,config.lightBlocksPerSm,config.csrHeavyWorkerGrid);}

 for(int c=0;c<2;++c){s->csrCellTaskOffset[c]=nt;s->csrCellTaskCount[c]=0;
  for(int start=s->cellParticleOffset[c];start<s->cellParticleOffset[c+1];start+=tile){s->csrReductionTasks[nt++]={c,start,std::min(start+tile,s->cellParticleOffset[c+1]),0};++s->csrCellTaskCount[c];}
  if(s->csrCellTaskCount[c]>1)s->csrMultiTaskCellList[nh++]=c;
 }
 s->csrCellTaskOffset[2]=nt;*s->csrHeavyTaskCount=nt;*s->csrHeavyCellCount=nh;s->csrReductionDirectoryKind=0;
 DeviceState host=*s;
 auto launch=[&](){clearPoissonThermalPoolKernel<<<1,32>>>(s,dt);
  if(heavy){if(launchCsrSegmentedPoolReduction(&host,dt,poisson,block))std::abort();}
  else if(poisson) { accumulatePoissonPoolParticlesByCellKernel<false><<<2,block,8*block*sizeof(Real)>>>(s,dt); }
  else accumulateParticlePoolAtomicKernel<false><<<24,block>>>(s,dt);
 };
 initParticles<<<24,128>>>(s,contact);launch();CHECK(cudaDeviceSynchronize()==cudaSuccess);
 unsigned long long hash=1469598103934665603ULL;int selected=0;
 for(int c=0;c<2;++c){double sum[5]={};int count=0;
  for(int i=s->cellParticleOffset[c];i<s->cellParticleOffset[c+1];++i){hash=(hash^s->pRng[i])*1099511628211ULL;hash=(hash^s->pStatus[i])*1099511628211ULL;
   if(s->pStatus[i]!=2)continue;++count;++selected;double m=s->pm[i],ux=s->pux[i],uy=s->puy[i],uz=s->puz[i];double theta=double(s->pTheta[i]);
   sum[0]+=m;sum[1]+=m*ux;sum[2]+=m*uy;sum[3]+=m*uz;sum[4]+=m*(.5*(ux*ux+uy*uy+uz*uz)+1.5*theta);
  }
  CHECK(count==s->poolThermalCount[c]);double got[5]={s->poissonPoolMass[c],s->poissonPoolMomX[c],s->poissonPoolMomY[c],s->poissonPoolMomZ[c],s->poissonPoolEnergy[c]};for(int j=0;j<5;++j)CHECK(fabs(got[j]-sum[j])<=tol*(1+fabs(sum[j])));
 }
 if(!bench){
  preparePoissonPoolSamplingKernel<<<1,32>>>(s);sampleOneLaunch<<<1,64>>>(s);CHECK(cudaDeviceSynchronize()==cudaSuccess);
  for(int i=0;i<N;++i){CHECK(std::isfinite(double(s->pux[i])));;hash=(hash^s->pRng[i])*1099511628211ULL;}
  printf("STATE poisson=%d contact=%d block=%d heavy=%d selected=%d hash=%llu\n",poisson,contact,block,heavy,selected,hash);
 }else{
  for(int rep=0;rep<5;++rep){initParticles<<<24,128>>>(s,0);for(int j=0;j<30;++j)launch();CHECK(cudaDeviceSynchronize()==cudaSuccess);cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);for(int j=0;j<200;++j)launch();cudaEventRecord(b);cudaEventSynchronize(b);float ms=0;cudaEventElapsedTime(&ms,a,b);printf("TIME N=%d heavy=%d rep=%d ms=%.9g\n",N,heavy,rep,ms/200);cudaEventDestroy(a);cudaEventDestroy(b);}
 }
}
puts("PASS collision pool CPU moments, selected IDs/RNG and contact metadata");}
