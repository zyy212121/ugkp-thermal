#include "gpu/MaterialSourceKernels.cuh"
#include "tests/TestSupport.H"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
using namespace chmt;
using namespace chmt::materialSourceDetail;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>T* managed(int count){T* p;check(cudaMallocManaged(&p,sizeof(T)*count));for(int i=0;i<count;++i)p[i]=T{};return p;}
int main(){
 auto p=managed<PhysicsConfig>(1);*p=chmt_test::physics();p->enableReactions=true;p->nReactions=1;
 for(int j=0;j<Nc;++j){p->condensed[j]=p->condensed[0];p->condensed[j].cp0=1000;p->condensed[j].cp1=0;p->condensed[j].e0=0;}
 p->reactions[0].A=100;p->reactions[0].condensedNu[0]=-1;p->reactions[0].condensedNu[1]=1;p->reactions[0].order[0]=1;
 auto current=managed<SolidQ>(2),rates=managed<SolidQ>(2),initial=managed<SolidQ>(2),change=managed<SolidQ>(2),candidateChange=managed<SolidQ>(2),next=managed<SolidQ>(2),donor=managed<SolidQ>(2);
 auto status=managed<MaterialStepResult>(2),result=managed<MaterialStepResult>(1);auto volumes=managed<Real>(2);auto offsets=managed<int>(3);
 for(int c=0;c<2;++c){current[c].condensed[0]=1;current[c].energy=500000;current[c].porosity=.5;initial[c]=current[c];}
 rates[0].condensed[0]=-2;rates[1].condensed[0]=2;rates[0].energy=-1000000;rates[1].energy=1000000;
 // Interior ALE face transfers 0.2 kg and its sensible energy conservatively.
 volumes[0]=.0016;volumes[1]=.0024;
 SolidQ oldReaction;auto old=advanceMaterialReactionsMath(current[0],.002,*p,.1,.001,.4,10000,oldReaction);
 chmt_test::check(old.code==MaterialReactionCode::Success,"old independent reaction");
 chmt_test::check(oldReaction.condensed[0]+.1*rates[0].condensed[0]<0,"original joint-negative counterexample");
 for(bool retain:{false,true}){
  *result={};
  applyMaterialSourcesKernel<<<1,128>>>(2,current,rates,initial,change,candidateChange,nullptr,offsets,nullptr,offsets,nullptr,nullptr,nullptr,volumes,p,.1,false,next,retain?donor:nullptr,status,.001,.4,10000);
  selectSourceFailure<<<1,256>>>(status,2,result);check(cudaDeviceSynchronize());
  chmt_test::check(!result->phase,"consistent source split accepted");
  double mass=0,energy=0;
  for(int c=0;c<2;++c){for(double m:next[c].condensed){chmt_test::check(m>=0,"positive condensed inventory");mass+=m;}energy+=next[c].energy;
   chmt_test::near(next[c].condensed[0],(c?1.2:.8)*std::pow(.9,100),"unchanged explicit reaction integrator",2e-11,1e-15);
   MaterialPrimitive primitive;chmt_test::check(recoverMaterial(next[c],volumes[c],*p,primitive),"ALE endpoint EOS");chmt_test::near(primitive.temperature,500,"isothermal reaction and transport");
   if(retain)chmt_test::near(donor[c].condensed[0],next[c].condensed[0]-initial[c].condensed[0],"donor cumulative inventory");
  }
  chmt_test::near(mass,2,"two-cell mass closure");chmt_test::near(energy,1000000,"two-cell energy closure");
 }
 // Failure precedence is phase first, then lowest global cell; empty input
 // and off-block errors exercise the same production reduction.
 const int n=1000003;auto failures=managed<MaterialStepResult>(n);
 for(int trial=0;trial<4;++trial){
  if(trial==1){failures[n-1]={3,9,n-1,false};failures[700]={2,8,700,true};failures[800000]={1,7,800000,true};}
  if(trial==2)failures[900]={1,6,900,false};
  *result={};const int count=trial==3?0:n;
  selectSourceFailure<<<1,256>>>(failures,count,result);check(cudaDeviceSynchronize());
  const int expected=trial==1?800000:trial==2?900:-1;
  chmt_test::check(result->entity==expected,"deterministic phase/cell failure selection");
 }
 for(void* v:{(void*)p,(void*)current,(void*)rates,(void*)initial,(void*)change,(void*)candidateChange,(void*)next,(void*)donor,(void*)status,(void*)result,(void*)volumes,(void*)offsets,(void*)failures})check(cudaFree(v));
 std::cout<<"PASS strong reaction plus internal ALE transport, conservation, donor optional, deterministic failure priority\n";
}
