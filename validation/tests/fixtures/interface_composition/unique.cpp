#include "ablation/InterfaceMath.H"
#include <cassert>
#include <cstdio>
using namespace chmt;
int main(){PhysicsConfig p;Real bulk[Ns],target[Ns],Js[Ns],out[Ns],sum=0,R=0;for(int i=0;i<Ns;++i){p.species[i].R=250+10*i;p.gasDiffusivity[i]=1e-5*(i+1);bulk[i]=1./Ns;target[i]=(i+1.);sum+=target[i];}for(int i=0;i<Ns;++i){target[i]/=sum;R+=target[i]*p.species[i].R;}Real S=0;for(int i=0;i<Ns;++i)S+=101325/(R*300)*p.gasDiffusivity[i]/.01*(target[i]-bulk[i]);for(int i=0;i<Ns;++i)Js[i]=.01*target[i]+101325/(R*300)*p.gasDiffusivity[i]/.01*(target[i]-bulk[i])-target[i]*S;
 composition_detail::Info info;bool ok=composition_detail::scalarSolve(Js,bulk,101325,300,.01,p,1e-12,out,&info);assert(ok&&info.state==composition_detail::SolveState::Converged);assert(info.polynomialCount==0);for(int i=0;i<Ns;++i)assert(absValue(out[i]-target[i])<1e-8);printf("PASS unique Ns=%d polynomial=%d residual=%.17g\n",Ns,info.polynomialCount,info.actualResidual);}
