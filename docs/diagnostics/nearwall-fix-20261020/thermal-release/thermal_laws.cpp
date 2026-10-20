#include "common/gasNumerics/OpenFoamWallFunctions.cuh"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
// Independent scalar transcription of the bundled matched-of10/EEqn.H.
// Inputs are the exact values written by the original OF10 EEqn audit.
int main(int argc,char**argv){
 if(argc!=3)return 1;
 std::ifstream in(argv[1]);std::ofstream out(argv[2]);std::string line;
 std::getline(in,line);out<<std::setprecision(17)<<"face,owner,conduction,JayatillekeCorrection,matchedRecomputed,matchedRecorded,ugkpRecomputed,scaledDifference\n";
 const double Pr=.72,Prt=.9,kappa=.41,E=9.8,Cmu=.09,ratio=Pr/Prt;
 const double P=9.24*(std::pow(ratio,.75)-1)*(1+.28*std::exp(-.007*ratio));
 auto eq=[&](double y){return kappa*ratio*y-std::log(E*y)-kappa*P;};
 double lo=1/(kappa*ratio),hi=2*lo;
 while(eq(hi)<0)hi*=2;
 for(int i=0;i<80;++i){double mid=.5*(lo+hi);if(eq(mid)>0)hi=mid;else lo=mid;}
 double yt=.5*(lo+hi),worst=0,recordWorst=0;int count=0;
 while(std::getline(in,line)){
  std::replace(line.begin(),line.end(),',',' ');std::istringstream s(line);
  int f,c;double y,rho,mu,cp,k,u,uw,grad,conductivity,correction,record;
  if(!(s>>f>>c>>y>>rho>>mu>>cp>>k>>u>>uw>>grad>>conductivity>>correction>>record))return 2;
  const double us=std::pow(Cmu,.25)*std::sqrt(std::max(k,0.)),yp=us*y*rho/mu;
  const bool viscous=yp<yt;
  const double tp=viscous?Pr*yp:Prt*(std::log(E*yp)/kappa+P);
  const double uc=viscous?0:us/kappa*std::log(E*yt)-uw;
  const double C=.5*rho*us*(viscous?Pr*u*u:Prt*u*u+(Pr-Prt)*uc*uc);
  const double alpha=std::max(0.,rho*us*y/tp-mu/Pr);
  const double conduction=cp*(mu/Pr+alpha)*grad,q=conduction-C/tp;
  auto a=ugkpwall::sstJayatillekeThermalTransport(rho,cp,mu,Pr,Prt,Cmu,kappa,E,
    ugkpwall::jayatillekeSmoothP(ratio),ugkpwall::jayatillekeThermalYPlus(ratio),k,y,u,uw,grad,false);
  if(!a.valid)return 3;
  double error=std::abs(a.heatFlux+q)/std::max(1.,std::abs(q));
  worst=std::max(worst,error);recordWorst=std::max(recordWorst,std::abs(q-record));++count;
  out<<f<<','<<c<<','<<conduction<<','<<-C/tp<<','<<q<<','<<record<<','<<-a.heatFlux<<','<<error<<'\n';
 }
 std::cout<<std::setprecision(17)<<"faces="<<count<<" matched_vs_record_max_absolute="<<recordWorst<<" ugkp_vs_matched_max_scaled="<<worst<<'\n';
 return count==336 && worst<1e-10 && recordWorst<1e-8?0:4;
}
