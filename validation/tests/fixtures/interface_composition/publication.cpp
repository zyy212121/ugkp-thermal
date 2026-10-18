// A selected, certified state must not repeat the same physical evaluation.
// Instrument real function entry, not a mocked residual. External verification
// below is excluded from counting and checks the actual original equations.
#include "ablation/InterfaceMath.H"
#include <cstdio>
#include <cstdlib>
using namespace chmt;
static bool counting=false;
static unsigned publishedEvaluations=0,fullDiffusionPreparations=0,divisionCalls=0;
extern "C" void __attribute__((no_instrument_function)) __cyg_profile_func_enter(void* fn,void*) {
 if(counting && fn==reinterpret_cast<void*>(&compositionResidualPolicy<true>))++publishedEvaluations;
 if(counting && fn==reinterpret_cast<void*>(&composition_detail::diffusionScale))++fullDiffusionPreparations;
 if(counting && fn==reinterpret_cast<void*>(&composition_detail::quotientFinite))++divisionCalls;
}
extern "C" void __attribute__((no_instrument_function)) __cyg_profile_func_exit(void*,void*) {}
void require(bool ok,const char* message){if(!ok){std::fprintf(stderr,"%s\n",message);std::exit(2);}}
int main(){
 for(int mode=0;mode<3;++mode){
  PhysicsConfig p;ugkp::diagnostics::DeviceContext diagnostics{};p.diagnostics=&diagnostics;
  Real b[Ns],target[Ns],Js[Ns]{},out[Ns]{},R=0,S=0,sum=0;
  for(int i=0;i<Ns;++i){p.species[i].R=250+10*i;p.gasDiffusivity[i]=mode==1?0:1e-5*(i+1);b[i]=1./Ns;target[i]=i+1.;sum+=target[i];}
  for(int i=0;i<Ns;++i){target[i]/=sum;R+=target[i]*p.species[i].R;}
  for(int i=0;i<Ns;++i)S+=101325/(R*300)*p.gasDiffusivity[i]/.01*(target[i]-b[i]);
  for(int i=0;i<Ns;++i)Js[i]=mode==2?0:.01*target[i]+101325/(R*300)*p.gasDiffusivity[i]/.01*(target[i]-b[i])-target[i]*S;
  if(mode==1&&Ns>1){Js[0]=1;Js[1]=-1;for(int i=2;i<Ns;++i)Js[i]=0;}
  publishedEvaluations=fullDiffusionPreparations=divisionCalls=0;counting=true;composition_detail::Info info;
  bool ok=mode==1?composition_detail::linearComposition(Js,b,Ns>1?0:.01,101325,300,.01,p,1e-12,out,&info):composition_detail::scalarSolve(Js,b,101325,300,.01,p,1e-12,out,&info);
  counting=false;require(ok&&!diagnostics.fatalClaim,"valid selected state failed");
  Real f[Ns]{},actual=0;require(compositionResidual(out,b,Js,101325,300,.01,p,f,nullptr),"original equations failed");
  for(int i=0;i<Ns;++i){actual=maxValue(actual,absValue(f[i]));require(out[i]>=0&&out[i]<=1,"published outside simplex");}
  require(info.actualResidual==actual,"published residual does not match selected state");
  require((info.state==composition_detail::SolveState::Converged)==(actual<=1e-12),"published typed state changed");
  std::printf("Ns=%d mode=%d duplicate_final_evaluations=%u actual=%.17g\n",Ns,mode,publishedEvaluations,actual);
  require(publishedEvaluations==0,"already certified solution evaluated again at publication");
  std::printf("full diffusion preparations=%u\n",fullDiffusionPreparations);
  require(fullDiffusionPreparations<=Ns,"normal call repeated full diffusion preparation across candidates");
  // Zero-flux initial-state certification needs one EOS division and Ns
  // coefficient/distance divisions. It performs no scalar/Newton root work.
  if(mode==2)require(divisionCalls==Ns+1,"repeated EOS divisions inside species loop");
 }
}
