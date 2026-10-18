#include <fenv.h>
#include "ablation/InterfaceMath.H"
#include <random>
#include <iostream>
#include <iomanip>
using namespace chmt;
int main(){feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);std::cout<<std::setprecision(17);std::mt19937_64 gen(74921);std::uniform_real_distribution<double> u(0,1);
int bad=0;for(int k=0;k<1000;k++){ugkp::diagnostics::DeviceContext context{};PhysicsConfig q;q.diagnostics=&context;Real b[Ns]{},y[Ns]{},flux[Ns]{},out[Ns]{},res[Ns]{};double sb=0,sy=0,R=0;
for(int i=0;i<Ns;i++){b[i]=std::pow(10.,-5*u(gen));y[i]=std::pow(10.,-5*u(gen));sb+=b[i];sy+=y[i];q.species[i].R=std::pow(10.,1+3*u(gen));q.gasDiffusivity[i]=std::pow(10.,-6+5*u(gen));}
for(int i=0;i<Ns;i++){b[i]/=sb;y[i]/=sy;R+=y[i]*q.species[i].R;}
double rho=101325/(R*300),J=std::pow(10.,-5+5*u(gen)),S=0;for(int i=0;i<Ns;i++)S+=rho*q.gasDiffusivity[i]/.01*(y[i]-b[i]);for(int i=0;i<Ns;i++)flux[i]=J*y[i]+rho*q.gasDiffusivity[i]/.01*(y[i]-b[i])-y[i]*S;
bool yes=interfaceComposition(flux,b,101325,300,.01,q,1e-12,out);bool rr=compositionResidual(out,b,flux,101325,300,.01,q,res,nullptr);double norm=0,err=0;for(int i=0;i<Ns;i++){norm=std::max(norm,std::abs(res[i]));err=std::max(err,std::abs(out[i]-y[i]));}double sum=0,minimum=1,maximum=0;for(int i=0;i<Ns;i++){sum+=out[i];minimum=std::min(minimum,out[i]);maximum=std::max(maximum,out[i]);}double tolerance=1e-12;for(int i=0;i<Ns;++i)tolerance+=q.tolerances.relativeMass*std::abs(flux[i]);if(context.fatalClaim||!yes||!rr||norm>tolerance||std::abs(sum-1)>1e-10||minimum<0||maximum>1)++bad;
}
std::cout<<"manufactured bad="<<bad<<"\n";return bad?1:0;
}
