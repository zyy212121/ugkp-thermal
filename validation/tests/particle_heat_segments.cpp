#include "ParticleCaloricHeatResponse.H"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
void require(bool ok,const char* message){if(!ok){std::fprintf(stderr,"%s\n",message);std::exit(1);}}
int main(){
 std::mt19937_64 random(938372);std::uniform_real_distribution<double> unit(0.,1.);
 long queries=0,inside=0;
 for(int trial=0;trial<20000;++trial){
  ugkwp::ParticleCaloric law;
  law.phaseChange=trial%2;law.meltingTemperature=1800+2000*unit(random);
  law.mushyRange=1+100*unit(random);law.solidCp=500+1500*unit(random);law.latentHeat=300000;
  const double oldT=100+10000*unit(random);
  const double f=trial%7==0?0.:(trial%7==1?1e-15:unit(random));
  ugkwp::ParticleCaloricHeatResponse response{law,oldT,f,100.,10000.};
  std::vector<double> starts{100.,3000.,1000+2500*unit(random)};
  if(f>0.)for(double cut:{100.,10000.,2327.,4000.,law.solidus(),law.liquidus()}){
   const double t=oldT+(cut-oldT)/f;
   starts.push_back(t);starts.push_back(std::nextafter(t,-INFINITY));starts.push_back(std::nextafter(t,INFINITY));
  }
  for(double t:starts){
   double q,d;response(t,q,d);const auto seg=response.segment(t);++queries;
   if(!(t>seg.lower&&t<seg.upper))continue;
   for(double change:{-10.,-.001,0.,.001,10.}){
    const double x=t+change;
    if(!(x>seg.lower&&x<seg.upper))continue;
    double exact,derivative;response(x,exact,derivative);
    const double dx=x-t,predicted=q+dx*(d+.5*seg.curvature*dx);
    const double scale=std::max({1.,std::abs(q),std::abs(exact),std::abs(dx*d)});
    if(std::abs(predicted-exact)>2e-11*scale){
     std::printf("trial %d T %.17g x %.17g f %.17g q %.17g exact %.17g predicted %.17g bounds %.17g %.17g\n",trial,t,x,f,q,exact,predicted,seg.lower,seg.upper);
     require(false,"segment enthalpy mismatch");
    }
    require(std::abs(d+seg.curvature*dx-derivative)<=2e-12*std::max(1.,std::abs(derivative)),"segment derivative mismatch");
    ++inside;
   }
  }
 }
 require(inside>100000,"insufficient interior queries");
 std::printf("PASS %ld reference segments, %ld interior polynomial comparisons including clipping, phase cuts, tiny/zero fractions and 10000 K particles\n",queries,inside);
}
