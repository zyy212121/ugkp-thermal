#include "gasNumerics/OpenFoamWallFunctions.cuh"
#define ugkpwall legacywall
#include "old-OpenFoamWallFunctions.cuh"
#undef ugkpwall
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <algorithm>
int main(){
 std::string line;std::getline(std::cin,line);
 std::cout<<std::setprecision(17)<<"face,cell,x,y,U,T,p,rhoWall,k,omega,nut,uTau_Spalding,yPlus_Spalding,nutFace,omegaExpected,productionExpected,yPlus_k,qIntoGas,conductivity,qFourierOut,qKineticOut,thermalYPlus,valid,legacy_uTau,legacy_omega,legacy_production,legacy_qIntoGas,spalding_root_residual\n";
 while(std::getline(std::cin,line)){
  std::replace(line.begin(),line.end(),',',' ');std::istringstream in(line);
  int face,cell;double x,y,U,T,p,rho,k,omega,nut;in>>face>>cell>>x>>y>>U>>T>>p>>rho>>k>>omega>>nut;
  constexpr double mu=1.7244421537657099e-5,cp=1004.675,Pr=.72,Prt=.9,Cmu=.09,kappa=.41,E=9.8,Tw=290.53216418474562;
  const double nu=mu/rho,grad=(Tw-T)/y;
  const auto mechanical=ugkpwall::spaldingWallStateFromNormalGradient(U,y,nu,U/y,kappa,E);
  const auto turbulent=ugkpwall::omegaWallFunctionState(k,U/y,y,nu,.075,Cmu,kappa,E);
  const double P=ugkpwall::jayatillekeSmoothP(Pr/Prt),yt=ugkpwall::jayatillekeThermalYPlus(Pr/Prt,kappa,E);
  const auto thermal=ugkpwall::sstJayatillekeThermalTransport(rho,cp,mu,Pr,Prt,Cmu,kappa,E,P,yt,k,y,U,0,grad);
  const auto oldMechanical=legacywall::spaldingWallStateFromNormalGradient(U,y,nu,U/y,kappa,E);
  const auto oldTurbulent=legacywall::omegaWallFunctionState(k,U/y,y,nu,.075,Cmu,kappa,E);
  const auto oldThermal=legacywall::sstJayatillekeThermalTransport(rho,cp,mu,Pr,Prt,Cmu,kappa,E,P,yt,k,y,U,0,grad);
  const auto root=ugkpwall::spaldingReynoldsRatio(U/mechanical.uTau,U*y/nu,std::log(U*y/nu),kappa,E)-1;
  std::cout<<face<<','<<cell<<','<<x<<','<<y<<','<<U<<','<<T<<','<<p<<','<<rho<<','<<k<<','<<omega<<','<<nut<<','<<mechanical.uTau<<','<<mechanical.yPlus<<','<<mechanical.nut<<','<<turbulent.omega<<','<<turbulent.production<<','<<turbulent.yPlus<<','<<-thermal.heatFlux<<','<<thermal.conductivity<<','<<-thermal.conductivity*grad<<','<<thermal.heatFlux+thermal.conductivity*grad<<','<<yt<<','<<thermal.valid<<','<<oldMechanical.uTau<<','<<oldTurbulent.omega<<','<<oldTurbulent.production<<','<<-oldThermal.heatFlux<<','<<root<<'\n';
 }
}
