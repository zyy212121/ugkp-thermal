#include <cuda_runtime.h>
#include "common/operators/advanceGasChemistryKernel.cuh"
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
using namespace ugkwp;
#define CU(call) do { auto chemistry_cuda_error=(call);if(chemistry_cuda_error!=cudaSuccess){std::cerr<<"CUDA_ERROR "<<__LINE__<<" "<<cudaGetErrorString(chemistry_cuda_error)<<"\n";std::exit(86);} } while(0)
std::vector<void*> allocations;
template<class T>T* upload(const T* a,int n){T* p=nullptr;CU(cudaMalloc(&p,n*sizeof(T)));CU(cudaMemcpy(p,a,n*sizeof(T),cudaMemcpyHostToDevice));allocations.push_back(p);return p;}
template<class T>T read(T* p){T v;CU(cudaMemcpy(&v,p,sizeof(T),cudaMemcpyDeviceToHost));return v;}
struct State:GasStateView<double,double,int>{GasSpeciesState<double,2> gasSpecies;GasBudgetAuditState gasBudgetAudit;};
struct Model {
 SpeciesThermoData<double> sp[2];double coef[6]={1000,0,0,1000,0,0},elements[2]={1,1},basis[2]={-1,1};
 GasReactionData<double> rx[1];GasStoichTerm<double> re[1],pr[1];SpeciesThermoView<double,2> t;GasMechanismView<double,2> m;
 Model(){for(int s=0;s<2;++s){sp[s].coefficientOffset=3*s;sp[s].molarMass=.028;sp[s].minTemperature=200;sp[s].maxTemperature=4000;sp[s].referencePressure=101325;sp[s].referenceTemperature=300;sp[s].referenceEntropy=1500;sp[s].hasEntropyReference=true;}
 t.species=sp;t.coefficients=coef;t.coefficientCount=6;t.elementCount=1;t.elementComposition=elements;t.speciesOrderHash=1;
 rx[0].reactantOffset=0;rx[0].reactantCount=1;rx[0].productOffset=0;rx[0].productCount=1;rx[0].highRate.preExponential=2;re[0].species=0;re[0].coefficient=1;pr[0].species=1;pr[0].coefficient=1;
 m.reactions=rx;m.reactionCount=1;m.reactants=re;m.reactantTermCount=1;m.products=pr;m.productTermCount=1;m.referencePressure=101325;m.speciesOrderHash=1;m.stoichiometricBasis=basis;m.independentRank=1; }
};
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdint>
using Clock=std::chrono::steady_clock;
double us(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
uint64_t hashBits(uint64_t h,const void* ptr,size_t n){auto p=(const unsigned char*)ptr;while(n--){h^=*p++;h*=1099511628211ULL;}return h;}
template<class T>void addHash(uint64_t& h,const T& x){h=hashBits(h,&x,sizeof x);}
int main(int argc,char**argv){
 const int n=argc>1?std::atoi(argv[1]):32768;const double dt=1e-6;Model model;State s{};s.nCells=n;s.gasSpecies.mode=GasMode::MixtureChemistry;
 const double mass[2]={28,0};const double energy=mixtureEnergy(mass,700.,model.t);
 size_t freeBefore,totalMemory;CU(cudaMemGetInfo(&freeBefore,&totalMemory));
 std::vector<double> rho(n,28),zero(n,0),en(n,energy),species(2*n,0),volume(n,.001),budget(2*n,0);std::fill(species.begin(),species.begin()+n,28);
 std::vector<int> cell(n,0);std::vector<ChemistryStatus> status(n);std::vector<ChemistryAudit<double,2>> audit(n);
 s.rho=upload(rho.data(),n);s.rhoUx=upload(zero.data(),n);s.rhoUy=upload(zero.data(),n);s.rhoUz=upload(zero.data(),n);s.rhoE=upload(en.data(),n);
 s.gasSpecies.rho=upload(species.data(),2*n);s.gasSpecies.cellStatus=upload(cell.data(),n);s.gasSpecies.chemistryStatus=upload(status.data(),n);s.gasSpecies.chemistryAudit=upload(audit.data(),n);
 s.gasSpecies.thermo=model.t;s.gasSpecies.mechanism=model.m;
 auto&t=s.gasSpecies.thermo;auto&m=s.gasSpecies.mechanism;t.species=upload(model.sp,2);t.coefficients=upload(model.coef,6);t.elementComposition=upload(model.elements,2);
 m.reactions=upload(model.rx,1);m.reactants=upload(model.re,1);m.products=upload(model.pr,1);m.stoichiometricBasis=upload(model.basis,2);
 s.gasSpecies.chemistryControls.maximumSteps=100;s.gasSpecies.chemistryControls.initialStep=dt;s.gasSpecies.chemistryControls.maximumStep=dt;
 s.gasBudgetAudit.enabled=true;s.gasBudgetAudit.speciesCount=2;s.gasBudgetAudit.chemistrySpecies=upload(budget.data(),2*n);
 State* dev=upload(&s,1);double* vols=upload(volume.data(),n);size_t freeAfter,unused;CU(cudaMemGetInfo(&freeAfter,&unused));
 std::vector<double> published(2*n),bud(2*n),qr(n),qx(n),qy(n),qz(n),qe(n);std::vector<int> cs(n);std::vector<ChemistryStatus> st(n);std::vector<ChemistryAudit<double,2>> au(n);
 cudaEvent_t start,stop;CU(cudaEventCreate(&start));CU(cudaEventCreate(&stop));
 std::cout<<std::setprecision(17)<<"{\"cells\":"<<n<<",\"dt\":"<<dt<<",\"warmup\":3,\"repeats\":7,\"malloc_count\":"<<allocations.size()<<",\"device_allocated_delta_bytes\":"<<freeBefore-freeAfter<<",\"samples\":[\n";
 bool allGood=true;uint64_t referenceHash=0;int kernels=0;
 for(int k=0;k<10;++k){auto begin=Clock::now();
  CU(cudaMemcpy(s.gasSpecies.rho,species.data(),2*n*sizeof(double),cudaMemcpyHostToDevice));auto resetEnd=Clock::now();
  CU(cudaMemset(s.gasSpecies.cellStatus,0,n*sizeof(int)));CU(cudaMemset(s.gasSpecies.chemistryStatus,0,n*sizeof(ChemistryStatus)));CU(cudaMemset(s.gasSpecies.chemistryAudit,0,n*sizeof(ChemistryAudit<double,2>)));CU(cudaMemset(s.gasBudgetAudit.chemistrySpecies,0,2*n*sizeof(double)));auto prepareEnd=Clock::now();
  CU(cudaEventRecord(start));auto launchBegin=Clock::now();
 #if defined(BASELINE)
  advanceGasChemistryKernel<<<(n+127)/128,128>>>(dev,dt,vols);
 #else
  advanceGasChemistryKernel<<<(n+127)/128,128>>>(dev,dt,vols,true);
 #endif
  ++kernels;CU(cudaGetLastError());auto launchEnd=Clock::now();CU(cudaEventRecord(stop));auto syncBegin=Clock::now();CU(cudaEventSynchronize(stop));auto syncEnd=Clock::now();float kernelMs;CU(cudaEventElapsedTime(&kernelMs,start,stop));
  auto copyBegin=Clock::now();
  CU(cudaMemcpy(published.data(),s.gasSpecies.rho,2*n*sizeof(double),cudaMemcpyDeviceToHost));CU(cudaMemcpy(bud.data(),s.gasBudgetAudit.chemistrySpecies,2*n*sizeof(double),cudaMemcpyDeviceToHost));
  CU(cudaMemcpy(cs.data(),s.gasSpecies.cellStatus,n*sizeof(int),cudaMemcpyDeviceToHost));CU(cudaMemcpy(st.data(),s.gasSpecies.chemistryStatus,n*sizeof(ChemistryStatus),cudaMemcpyDeviceToHost));CU(cudaMemcpy(au.data(),s.gasSpecies.chemistryAudit,n*sizeof(ChemistryAudit<double,2>),cudaMemcpyDeviceToHost));
  CU(cudaMemcpy(qr.data(),s.rho,n*sizeof(double),cudaMemcpyDeviceToHost));CU(cudaMemcpy(qx.data(),s.rhoUx,n*sizeof(double),cudaMemcpyDeviceToHost));CU(cudaMemcpy(qy.data(),s.rhoUy,n*sizeof(double),cudaMemcpyDeviceToHost));CU(cudaMemcpy(qz.data(),s.rhoUz,n*sizeof(double),cudaMemcpyDeviceToHost));CU(cudaMemcpy(qe.data(),s.rhoE,n*sizeof(double),cudaMemcpyDeviceToHost));
  auto copyEnd=Clock::now();uint64_t hash=14695981039346656037ULL;double maxAnalytic=0,maxClosure=0;bool good=true;
  for(int i=0;i<n;++i){good &= qr[i]==28&&qx[i]==0&&qy[i]==0&&qz[i]==0&&qe[i]==energy&&cs[i]==0&&int(st[i].code)==0&&au[i].acceptedSteps==1&&au[i].integratedTime==dt;
   maxAnalytic=std::max(maxAnalytic,std::abs(published[n+i]-28*(-std::expm1(-2*dt))));maxClosure=std::max(maxClosure,std::abs(published[i]+published[n+i]-28));
   for(double x:{qr[i],qx[i],qy[i],qz[i],qe[i],published[i],published[n+i],bud[i],bud[n+i],au[i].speciesMassChange[0],au[i].speciesMassChange[1],au[i].massResidual,au[i].maximumElementResidual,au[i].energyResidual,au[i].maximumLocalError,au[i].integratedTime}){good &= std::isfinite(x);addHash(hash,x);}
   for(int x:{cs[i],int(st[i].code),int(st[i].thermoCode),st[i].reaction,st[i].attemptedSteps,st[i].rejectedSteps,st[i].nonlinearIterations,au[i].acceptedSteps,au[i].rejectedSteps,au[i].nonlinearIterations})addHash(hash,x);
  }
  good &= maxAnalytic<1e-9&&maxClosure<1e-10;if(k==0)referenceHash=hash;good &= hash==referenceHash;allGood &= good;
  std::cout<<(k?",\n":"")<<"{\"sample\":"<<k<<",\"warmup\":"<<(k<3?"true":"false")<<",\"kernel_ms\":"<<kernelMs<<",\"reset_h2d_us\":"<<us(begin,resetEnd)<<",\"prepare_memsets_us\":"<<us(resetEnd,prepareEnd)<<",\"host_launch_us\":"<<us(launchBegin,launchEnd)<<",\"host_wait_us\":"<<us(syncBegin,syncEnd)<<",\"full_publication_d2h_us\":"<<us(copyBegin,copyEnd)<<",\"whole_pipeline_us\":"<<us(begin,copyEnd)<<",\"publication_hash_fnv1a64\":\""<<std::hex<<hash<<std::dec<<"\",\"max_analytic_species_abs\":"<<maxAnalytic<<",\"max_density_closure_abs\":"<<maxClosure<<",\"all_fields_pass\":"<<(good?"true":"false")<<"}";
 }
 std::cout<<"\n],\"all_fields_pass\":"<<(allGood?"true":"false")<<",\"actual_chemistry_kernels\":"<<kernels<<",\"per_sample_h2d_bytes\":"<<2*n*sizeof(double)<<",\"per_sample_d2h_bytes\":"<<n*(9*sizeof(double)+sizeof(int)+sizeof(ChemistryStatus)+sizeof(ChemistryAudit<double,2>))<<"}\n";
 CU(cudaEventDestroy(start));CU(cudaEventDestroy(stop));for(auto p:allocations)CU(cudaFree(p));return allGood?0:2;
}
