#include "gasNumerics/RiemannGasFlux.cuh"
#include <iostream>
#include <iomanip>
#include <cmath>
int main(){
 using namespace ugkpriemann;std::cout<<std::setprecision(17)<<"gamma,rho,p,leftUt,rightUt,exact_Fmx,Kurganov_Fmx,expected_Kurganov,HLLC_Fmx,SLAU2_Fmx\n";int count=0;
 for(double gamma:{1.2,1.4,5./3})for(double rho:{.1,1.,10.})for(double Mach:{.1,.55,2.})for(double jump:{.01,.2,1.}){
  double p=1e5,a=std::sqrt(gamma*p/rho),u=Mach*a;Primitive l{rho,u,0,0,p},r{rho,u+jump,0,0,p};
  auto k=fluxUnitArea(l,r,0,1,0,gamma,Scheme::HllKurganov),h=fluxUnitArea(l,r,0,1,0,gamma,Scheme::HLLC),s=fluxUnitArea(l,r,0,1,0,gamma,Scheme::SLAU2);double expected=-.5*a*rho*jump;
  if(!k.valid||!h.valid||!s.valid||std::abs(k.flux[1]-expected)>1e-9*std::abs(expected)||h.flux[1]!=0||s.flux[1]!=0||k.flux[0]!=0||h.flux[0]!=0)return 2;
  std::cout<<gamma<<','<<rho<<','<<p<<','<<u<<','<<u+jump<<",0,"<<k.flux[1]<<','<<expected<<','<<h.flux[1]<<','<<s.flux[1]<<'\n';++count;
 }
 std::cerr<<"stationary contact probes="<<count<<"; exact Euler shear flux=0; all mathematical checks passed\n";
}
