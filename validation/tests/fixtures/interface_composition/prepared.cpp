#include "ablation/InterfaceMath.H"
#include <fenv.h>
#include <random>
#include <cstdio>
#include <cstdlib>
using namespace chmt;
void require(bool ok,const char* msg){if(!ok){std::fprintf(stderr,"%s\n",msg);std::exit(2);}}
int main(){feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);std::mt19937_64 rng(82139);std::uniform_real_distribution<Real> u(0,1);
 for(int trial=0;trial<20000;++trial){PhysicsConfig p;Real b[Ns],j[Ns],y[Ns],sum=0;
  for(int i=0;i<Ns;++i){p.species[i].R=std::pow(10.,-10+20*u(rng));p.gasDiffusivity[i]=trial%7?std::pow(10.,-15+15*u(rng)):0;b[i]=u(rng);sum+=b[i];j[i]=std::pow(10.,-5+5*u(rng))*(u(rng)-.5);y[i]=u(rng);}
  for(int i=0;i<Ns;++i)b[i]/=sum;
  CompositionResidualInputs fixed;require(prepareCompositionResidual<false>(b,j,p,fixed),"fixed preparation failed");
  for(int candidate=0;candidate<3;++candidate){Real oldF[Ns]{},newF[Ns]{},oldJ[Ns*Ns]{},newJ[Ns*Ns]{};
   bool old=compositionResidualPolicy<false>(y,b,j,101325,300,.01,p,oldF,oldJ);
   bool now=compositionResidualPolicy<false>(y,b,j,101325,300,.01,p,newF,newJ,&fixed);require(old==now,"prepared evaluation changed success");
   if(old){for(int i=0;i<Ns;++i)require(oldF[i]==newF[i],"prepared original residual differs");for(int i=0;i<Ns*Ns;++i)require(oldJ[i]==newJ[i],"prepared Jacobian differs");}
   // The same invocation uses genuinely different compositions and mixture R.
   for(int i=0;i<Ns;++i)y[i]=u(rng);
  }
 }
 // The common EOS work must preserve the guarded extreme/subnormal path too.
 const Real numbers[]={0,4.9406564584124654e-324,2.2250738585072014e-308,1e-200,1e-20,1,1e20,1e200,1.7976931348623157e308};
 for(Real pressure:numbers)for(Real T:numbers)for(Real R:numbers)if(pressure>0&&T>0&&R>0){composition_detail::DiffusionState state(pressure,T,.01,R);
  for(Real D:numbers){Real before=0,after=0;bool ok=composition_detail::diffusionScale(pressure,D,T,.01,R,before);bool fresh=state.scale(D,after);require(ok==fresh&&(!ok||before==after),"shared EOS changed extreme diffusion coefficient");}}
 require(!std::fetestexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW),"unexpected floating exception");std::printf("PASS prepared Ns=%d: 60000 changing candidates plus diffusion extremes\n",Ns);
}
