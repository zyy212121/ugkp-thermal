#include <fenv.h>
#include "ablation/InterfaceMath.H"
#include <random>
#include <iostream>
#include <iomanip>
using namespace chmt;
int main(){feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);std::cout<<std::setprecision(17);std::mt19937_64 g(582341);std::uniform_real_distribution<double> u(0,1);int bad=0,total=0;for(int mode=0;mode<12;++mode){int mb=0;for(int k=0;k<20;++k){PhysicsConfig p;double b[Ns]{},y[Ns]{},Js[Ns]{},out[Ns]{},F[Ns]{},sb=0,sy=0,R=0,Db=0,Dy=0;for(int i=0;i<Ns;++i){b[i]=.01+u(g);y[i]=.01+u(g);sb+=b[i];sy+=y[i];p.species[i].R=(mode==4||mode==5)?300:10+1000*u(g);p.gasDiffusivity[i]=1e-6+.01*u(g);if(mode==1||mode==5)p.gasDiffusivity[i]=.01;if(mode==2||mode==10)p.gasDiffusivity[i]=i%2?.01:0;if(mode==3)p.gasDiffusivity[i]=i%2?.01:.02;if(mode==11)p.gasDiffusivity[i]=0;}
for(int i=0;i<Ns;++i){b[i]/=sb;y[i]/=sy;R+=p.species[i].R*y[i];Db+=p.gasDiffusivity[i]*b[i];Dy+=p.gasDiffusivity[i]*y[i];}double rho=101325/(300*R),J=.01+.1*u(g),S=rho/.01*(Dy-Db);if(mode==6)J=0;if(mode==7)J=-J;if(mode==9)J=1e-20;if(mode==10)J=S;for(int i=0;i<Ns;++i)Js[i]=J*y[i]+rho*p.gasDiffusivity[i]/.01*(y[i]-b[i])-y[i]*S;if(mode==8)for(int i=0;i<Ns;++i)Js[i]=0;if(mode==10)for(int i=0;i<Ns;++i)if(p.gasDiffusivity[i]==0)Js[i]=0;
bool ok=interfaceComposition(Js,b,101325,300,.01,p,1e-12,out);bool valid=compositionResidual(out,b,Js,101325,300,.01,p,F,nullptr);double norm=0,sum=0,mn=1,mx=0;for(int i=0;i<Ns;++i){norm=std::max(norm,std::abs(F[i]));sum+=out[i];mn=std::min(mn,out[i]);mx=std::max(mx,out[i]);}double tolerance=1e-12;for(int i=0;i<Ns;++i)tolerance+=p.tolerances.relativeMass*std::abs(Js[i]);bool good=ok&&valid&&norm<=tolerance&&std::abs(sum-1)<1e-10&&mn>=0&&mx<=1;++total;if(!good){++bad;++mb;std::cout<<"BAD "<<mode<<' '<<k<<' '<<ok<<' '<<valid<<' '<<norm<<' '<<sum<<' '<<mn<<' '<<mx<<'\n';}}
std::cout<<"MODE "<<mode<<" bad="<<mb<<'\n';}std::cout<<"TOTAL "<<total<<" bad="<<bad<<'\n';return bad?1:0;}
