#include "/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261014T072053Z-fast/review/pr14/reviewer-deliveries/20261014T072053Z-fast/inputs/actual-CHT-state-prefix.cuh"
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <vector>
#include <cstring>
extern "C" int ugkwpGpuResidentStrictConfigureSst
(
    void* handle,
    double alphaK1Wire,
    double alphaK2Wire,
    double alphaOmega1Wire,
    double alphaOmega2Wire,
    double beta1Wire,
    double beta2Wire,
    double betaStarWire,
    double gamma1Wire,
    double gamma2Wire,
    double a1Wire,
    double b1Wire,
    double c1Wire,
    double kMinWire,
    double omegaMinWire,
    double maxSourceNumberWire,
    int wallTreatment,
    double wallKappaWire,
    double wallEWire,
    double wallCmuWire,
    const double* k,
    const double* omega,
    const double* wallDistance,
    const int* boundaryKMode,
    const int* boundaryOmegaMode,
    const double* boundaryK,
    const double* boundaryOmega
)
;
extern "C" void ugkwpGpuResidentStrictRelease
(
    void* handle
)
;
extern "C" int ugkwpGpuResidentStrictConfigureParticleStuckModel
(
    void* handle,
    int nFaces,
    const unsigned char* candidateFaceMask,
    double sommerfeldThresholdWire,
    int heatTransferEnabled,
    double maximumCoverageWire,
    double depositionHeatTransferEfficiencyWire,
    double reflectionHeatTransferEfficiencyWire,
    double adhesionEnergyScaleWire,
    double contactAngleDegreeWire,
    int wallTransientResistance,
    int nonlinearIterations,
    double meltingTemperatureKWire,
    double mushyRangeKWire,
    double latentHeatJkgWire,
    double solidDensityKgM3Wire,
    double solidSpecificHeatJkgKWire,
    double solidThermalConductivityWmKWire,
    double pinningThicknessFractionWire,
    double interfaceResistanceM2KWWire
)
;
extern "C" int ugkwpGpuResidentStrictUploadParticleWallEffusivityPatch
(
    void* handle,
    int patchStartFace,
    int patchFaceCount,
    const double* wallEffusivity
)
;
void check(bool x,const char* s){if(!x){std::fprintf(stderr,"FAIL %s\n",s);std::exit(11);}}
void cudaCheck(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));std::exit(12);}}
DeviceState* makeOwner(int cells=0){auto*s=new DeviceState;s->nFaces=4;s->nCells=cells;s->nInternalFaces=0;s->particleCapacity=4;
 cudaCheck(cudaMalloc(&s->deviceState,sizeof(DeviceState)));
 if(cells){cudaCheck(cudaMalloc(&s->rho,cells*sizeof(double)));cudaCheck(cudaMalloc(&s->nut,cells*sizeof(double)));double rho[2]={1.25,.75},nut[2]={97,98};cudaCheck(cudaMemcpy(s->rho,rho,cells*sizeof(double),cudaMemcpyHostToDevice));cudaCheck(cudaMemcpy(s->nut,nut,cells*sizeof(double),cudaMemcpyHostToDevice));}
 cudaCheck(cudaMemcpy(s->deviceState,s,sizeof(DeviceState),cudaMemcpyHostToDevice));return s;}
using SetMalloc=void(*)(int);using SetCopy=void(*)(std::size_t);using Count=int(*)();
SetMalloc setMalloc;SetCopy setCopy;Count errorCode,mallocCount,freeCount,copyCount;
void faultAPI(){setMalloc=reinterpret_cast<SetMalloc>(dlsym(RTLD_DEFAULT,"pr14_fault_malloc"));setCopy=reinterpret_cast<SetCopy>(dlsym(RTLD_DEFAULT,"pr14_fault_copy"));errorCode=reinterpret_cast<Count>(dlsym(RTLD_DEFAULT,"pr14_observed_error"));mallocCount=reinterpret_cast<Count>(dlsym(RTLD_DEFAULT,"pr14_malloc_count"));freeCount=reinterpret_cast<Count>(dlsym(RTLD_DEFAULT,"pr14_free_count"));copyCount=reinterpret_cast<Count>(dlsym(RTLD_DEFAULT,"pr14_copy_count"));check(setMalloc&&setCopy&&errorCode&&mallocCount&&freeCount&&copyCount,"actualCUDA faultobserver available");}
int configureSstOwner(DeviceState*s,double selected=.3){double k[2]={selected,.7},omega[2]={2,3},distance[2]={.5,.6},bk[4]={.1,.2,.3,.4},bw[4]={1,2,3,4};int km[4]={0,1,2,0},wm[4]={2,1,0,2};
 return ugkwpGpuResidentStrictConfigureSst(s,.85,1,.5,.856,.075,.0828,.09,.55555,.44,.31,1,10,1e-12,1e-9,.25,0,.41,9.8,.09,k,omega,distance,km,wm,bk,bw);}
std::vector<void*> owners(DeviceState*s){return {s->rhoK,s->rhoOmega,s->rhoKInitial,s->rhoOmegaInitial,s->k,s->omega,s->sstWallDistance,s->sstF1,s->sstF2,s->gradKX,s->gradKY,s->gradKZ,s->gradOmegaX,s->gradOmegaY,s->gradOmegaZ,s->sstPhiRhoK,s->sstPhiRhoOmega,s->sstBoundaryKMode,s->sstBoundaryOmegaMode,s->sstBoundaryK,s->sstBoundaryOmega,s->sstSourceNumber};}
void checkSst(DeviceState*s,double selected=.3){DeviceState d;cudaCheck(cudaMemcpy(&d,s->deviceState,sizeof(d),cudaMemcpyDeviceToHost));check(owners(s)==owners(&d),"all22canonical owners published exactly");
 double vals[2];for(auto item:{s->rhoK,s->rhoKInitial}){cudaCheck(cudaMemcpy(vals,item,sizeof(vals),cudaMemcpyDeviceToHost));check(vals[0]==1.25*selected&&vals[1]==.75*.7,"nativeSST rhoK/initial values");}
 for(auto item:{s->rhoOmega,s->rhoOmegaInitial}){cudaCheck(cudaMemcpy(vals,item,sizeof(vals),cudaMemcpyDeviceToHost));check(vals[0]==1.25*2&&vals[1]==.75*3,"nativeSST rhoOmega/initial values");}
 cudaCheck(cudaMemcpy(vals,s->nut,sizeof(vals),cudaMemcpyDeviceToHost));check(vals[0]==0&&vals[1]==0,"actualSST initializer resetsnut");
 std::printf("SST_DATA k0=%a rhoK0=%a rhoK1=%a rhoOmega0=%a rhoOmega1=%a pending=%u owners=%zu\n",selected,1.25*selected,.75*.7,1.25*2,.75*3,unsigned(s->particleStuckInitialisationPending),owners(s).size());}
void sstTest(){
 auto*s=makeOwner(2);check(configureSstOwner(s)==0,"lateSST configure");checkSst(s);auto all=owners(s);check(configureSstOwner(s,2.)==0,"repeatSST configure");check(owners(s)==all,"repeatowners unchanged");checkSst(s,2.);ugkwpGpuResidentStrictRelease(s);
 for(int n=1;n<=22;++n){s=makeOwner(2);setMalloc(n);check(configureSstOwner(s)!=0,"actualSST allocation error mustreturn");check(errorCode()==int(cudaErrorMemoryAllocation),"realCUDA OOM observed");auto before=owners(s);check(mallocCount()==22,"all22existing acquisition slots attempted");check(s->particleStuckInitialisationPending&0x1000,"canonical partialpublication stillpending");(void)cudaGetLastError();setMalloc(0);check(configureSstOwner(s)==0,"actualSST allocationretry");auto after=owners(s);for(std::size_t i=0;i<before.size();++i)if(before[i])check(before[i]==after[i],"partialSST ownersretained");check(mallocCount()==1,"retryallocates onlymissingowner");checkSst(s);setMalloc(0);ugkwpGpuResidentStrictRelease(s);std::printf("PASS SST_ACTUAL_ALLOC_RETRY n=%d frees=%d\n",n,freeCount());}
 s=makeOwner(2);setCopy(sizeof(double)*2);check(configureSstOwner(s)!=0,"actualSST copy error mustreturn");check(errorCode()!=0,"realCUDA copyerror observed");all=owners(s);(void)cudaGetLastError();setMalloc(0);check(configureSstOwner(s)==0,"actualSST copyretry");check(owners(s)==all&&mallocCount()==0,"copyretry preserves22owners");checkSst(s);ugkwpGpuResidentStrictRelease(s);std::puts("PASS actualGPU SST22 configure/reconfigure/realresource retry and destroy");}
#if CHT_TEST
int configureColdOwner(DeviceState*s){unsigned char mask[4]={0,3,4,0};return ugkwpGpuResidentStrictConfigureParticleStuckModel(s,4,mask,20,1,.63,1,1,1,145,1,12,2327,20,1160000,2500,1273,20,.5,0);}
void copyRetry(){auto*s=makeOwner();setCopy(4*sizeof(double));check(configureColdOwner(s)!=0,"coldinitialization actualcopyerror mustreturn");check(errorCode()!=0,"realCUDA copyerror observed");check(s->particleStuckInitialisationPending&4,"failedinitializer mustremainpending");auto*owner=s->particleWallEffusivityByFace;check(owner,"realallocatedeffusivity retained");(void)cudaGetLastError();setMalloc(0);check(configureColdOwner(s)==0,"actualcold initretry");check(s->particleWallEffusivityByFace==owner,"retryretains owner");double actual[4],fallback=std::sqrt(1800.*710.*100.);cudaCheck(cudaMemcpy(actual,owner,sizeof(actual),cudaMemcpyDeviceToHost));for(double x:actual)check(x==fallback,"allactual fallbackfaces initialized afterretry");double patch[2]={24001.25,17003.5};check(ugkwpGpuResidentStrictUploadParticleWallEffusivityPatch(s,1,2,patch)==0,"actualcoupled patchupload");check(configureColdOwner(s)==0,"reconfigureafter patch");cudaCheck(cudaMemcpy(actual,owner,sizeof(actual),cudaMemcpyDeviceToHost));check(actual[0]==fallback&&actual[1]==patch[0]&&actual[2]==patch[1]&&actual[3]==fallback,"mixedcoupled faces preserve overrides");for(int i=0;i<4;++i)std::printf("FACE_AFTER_RETRY i=%d value=%a\n",i,actual[i]);ugkwpGpuResidentStrictRelease(s);std::puts("PASS actualCUDA cold initialization copyerror/retry and mixedcoupled override");}
#endif
int main(int argc,char**argv){faultAPI();if(argc>1&&std::strcmp(argv[1],"sst")==0)sstTest();
#if CHT_TEST
 else copyRetry();
#endif
}
