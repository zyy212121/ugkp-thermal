#include "common/GpuPrecisionTypes.H"
#include "common/gasNumerics/RiemannGasFlux.cuh"
#include "common/gasTransport/MixtureThermo.H"
#include <array>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <cstdlib>
using R=GpuReal;
constexpr int Ns=4;
using Q=std::array<R,5+Ns>;
struct Thermo {
 ugkwp::SpeciesThermoData<R> species[Ns];R coefficients[3*Ns];ugkwp::SpeciesThermoView<R,Ns> view;
 Thermo(bool variable,bool sod=false){
  for(int k=0;k<Ns;++k){species[k].molarMass=sod?ugkwp::universalGasConstant<R>()/R(287):R(.028+.004*k);species[k].coefficientOffset=3*k;species[k].minTemperature=R(.001);species[k].maxTemperature=R(50000);species[k].referencePressure=101325;
   coefficients[3*k]=sod?R(1004.5):R(1200+100*k);coefficients[3*k+1]=variable?R(.1+.02*k):R(0);coefficients[3*k+2]=sod?R(0):R((k-1)*1e6);}
  view.species=species;view.coefficients=coefficients;view.coefficientCount=3*Ns;
 }
};
struct State{ugkpriemann::Primitive p;R sound,T;};
void require(bool ok,const char*message){if(!ok){std::cerr<<message<<'\n';std::exit(2);}}
State primitive(const Q&q,const Thermo&t){
 State r{};require(q[0]>0,"nonpositive density");R partial[Ns];for(int k=0;k<Ns;++k){partial[k]=q[5+k];require(partial[k]>=-1e-12*q[0],"negative species");}
 R internal=q[4]-(q[1]*q[1]+q[2]*q[2]+q[3]*q[3])/(2*q[0]);r.T=700;
 ugkwp::ThermoInversionControls<R> controls;controls.relativeEnergyTolerance=1e-12;controls.relativeTemperatureTolerance=1e-12;
 require(ugkwp::invertMixtureEnergy(partial,internal,t.view,controls,r.T)==ugkwp::ThermoStatus::Success,"EOS inversion failed");
 R gasR=ugkwp::mixtureGasConstant(partial,t.view),cv=ugkwp::mixtureHeatCapacity(partial,r.T,t.view)/q[0];
 r.p={q[0],q[1]/q[0],q[2]/q[0],q[3]/q[0],q[0]*gasR*r.T};r.sound=std::sqrt((cv+gasR)/cv*gasR*r.T);require(r.T>0&&r.p.p>0&&std::isfinite(r.sound),"nonphysical recovered state");return r;
}
Q state(R rho,R p,R u,R v,R y0,const Thermo&t){Q q{};q[0]=rho;q[1]=rho*u;q[2]=rho*v;R partial[Ns];for(int k=0;k<Ns;++k){partial[k]=rho*(k==0?y0:(1-y0)/(Ns-1));q[5+k]=partial[k];}R T=p/(rho*ugkwp::mixtureGasConstant(partial,t.view));q[4]=ugkwp::mixtureEnergy(partial,T,t.view)+R(.5)*rho*(u*u+v*v);return q;}
Q flux(const Q&l,const Q&r,const Thermo&t,int scheme){auto a=primitive(l,t),b=primitive(r,t);ugkpriemann::Conservative ul{},ur{};for(int k=0;k<5;++k){ul.q[k]=l[k];ur.q[k]=r[k];}
 auto f=scheme==4?ugkpriemann::hllcFluxUnitNormal(a.p,ul,a.sound,b.p,ur,b.sound,R(1),R(0),R(0)):ugkpriemann::hllKurganovFluxUnitNormal(a.p,ul,a.sound,b.p,ur,b.sound,R(1),R(0),R(0),false);require(f.valid,"nonfinite Riemann flux");Q out{};for(int k=0;k<5;++k)out[k]=f.flux[k];
 if(scheme==4)for(int k=0;k<Ns;++k)out[5+k]=f.flux[0]*(f.flux[0]>=0?l[5+k]/l[0]:r[5+k]/r[0]);
 else for(int k=0;k<Ns;++k)out[5+k]=ugkpriemann::hllConservativeComponentFlux(a.p.ux*l[5+k],b.p.ux*r[5+k],l[5+k],r[5+k],a.p.ux,b.p.ux,a.sound,b.sound);
 R sum=0;for(int k=0;k<Ns-1;++k)sum+=out[5+k];out[5+Ns-1]=out[0]-sum;return out;
}
// Independent ideal-gas exact Riemann reference, gamma=1.4. No production
// approximate Riemann solver participates in this pressure root or sampling.
struct Exact {
 R rhoL,pL,uL,rhoR,pR,uR,g=1.4,ps,us,al,ar;
 R curve(R p,R rho,R pk,R a){if(p>pk)return (p-pk)*std::sqrt(2/((g+1)*rho)/(p+(g-1)/(g+1)*pk));return 2*a/(g-1)*(std::pow(p/pk,(g-1)/(2*g))-1);}
 Exact(R rl,R pl,R ul,R rr,R pr,R ur):rhoL(rl),pL(pl),uL(ul),rhoR(rr),pR(pr),uR(ur){al=std::sqrt(g*pL/rhoL);ar=std::sqrt(g*pR/rhoR);R lo=0,hi=std::max(pL,pR);auto f=[&](R p){return curve(p,rhoL,pL,al)+curve(p,rhoR,pR,ar)+uR-uL;};while(f(hi)<0)hi*=2;for(int j=0;j<150;++j){R p=(lo+hi)/2;if(f(p)>0)hi=p;else lo=p;}ps=(lo+hi)/2;us=(uL+uR+curve(ps,rhoR,pR,ar)-curve(ps,rhoL,pL,al))/2;}
 std::array<R,3> sample(R xi){bool left=xi<=us;R rho=left?rhoL:rhoR,p=left?pL:pR,u=left?uL:uR,a=left?al:ar,sign=left?-1:1;
  if(ps>p){R shock=u+sign*a*std::sqrt((g+1)/(2*g)*ps/p+(g-1)/(2*g));if((xi-shock)*sign>=0)return {rho,u,p};R ratio=ps/p;return {rho*(ratio+(g-1)/(g+1))/((g-1)/(g+1)*ratio+1),us,ps};}
  R astar=a*std::pow(ps/p,(g-1)/(2*g)),head=u+sign*a,tail=us+sign*astar;
  if((xi-head)*sign>=0)return {rho,u,p};if((xi-tail)*sign<=0)return {rho*std::pow(ps/p,1/g),us,ps};
  R uf=2/(g+1)*(xi-sign*a+(g-1)/2*u),af=sign*(xi-uf);return {rho*std::pow(af/a,2/(g-1)),uf,p*std::pow(af/a,2*g/(g-1))};
 }
};
int main(int argc,char**argv){require(argc==6,"scenario scheme cells output-directory direction");std::string scenario=argv[1];int scheme=std::atoi(argv[2]),n=std::atoi(argv[3]);std::string out=argv[4];int direction=std::atoi(argv[5]);bool sod=scenario=="sod",rarefaction=scenario=="rarefaction",contact=!sod&&!rarefaction,variable=scenario=="variable-contact";Thermo thermo(variable,sod||rarefaction);
 R u=contact?R(direction*30):0,dx=R(1)/n,stop=contact?R(.002):R(.2)/std::sqrt(R(1e5)),time=0;std::vector<Q>q(n),initial(n);Q left,right;
 if(contact){left=state(1,1e5,u,50,.85,thermo);right=state(.6,1e5,u,-40,.15,thermo);}
 else if(sod){left=state(1,1e5,0,0,.85,thermo);right=state(.125,1e4,0,0,.15,thermo);}
 else {left=state(1,4e4,-2*std::sqrt(R(1e5)),0,.5,thermo);right=state(1,4e4,2*std::sqrt(R(1e5)),0,.5,thermo);}
 for(int c=0;c<n;++c)q[c]=((c+R(.5))*dx<.5?left:right);initial=q;
 std::array<long double,5+Ns>before{},boundary{};for(auto&v:q)for(int k=0;k<5+Ns;++k)before[k]+=dx*v[k];int steps=0;
 while(time<stop){R speed=0;for(auto&v:q){auto p=primitive(v,thermo);speed=std::max(speed,std::abs(p.p.ux)+p.sound);}R dt=std::min(stop-time,R(.4)*dx/speed);auto old=q;
  for(int stage=0;stage<2;++stage){std::vector<Q>f(n+1);for(int i=0;i<=n;++i)f[i]=flux(i==0?left:q[i-1],i==n?right:q[i],thermo,scheme);
   for(int k=0;k<5+Ns;++k)boundary[k]+=R(.5)*dt*(f[0][k]-f[n][k]);
   for(int c=0;c<n;++c)for(int k=0;k<5+Ns;++k){R trial=q[c][k]-dt/dx*(f[c+1][k]-f[c][k]);q[c][k]=stage==0?trial:R(.5)*(old[c][k]+trial);}
  }time+=dt;++steps;require(steps<200000,"step count limit in test");
 }
 std::array<long double,5+Ns>after{};R minT=1e100,minRho=1e100,pmax=0,pL1=0,rhoL1=0,uL1=0,minSpecies=1,closure=0;
 Exact exact(sod?R(1):R(1),sod?R(1e5):R(4e4),sod?R(0):-2*std::sqrt(R(1e5)),sod?R(.125):R(1),sod?R(1e4):R(4e4),sod?R(0):2*std::sqrt(R(1e5)));
 std::ofstream fields(out+"/fields.csv");fields<<std::setprecision(17)<<"x,rho,u,v,p,T,Y0\n";
 for(int c=0;c<n;++c){auto p=primitive(q[c],thermo);R x=(c+R(.5))*dx;minT=std::min(minT,p.T);minRho=std::min(minRho,p.p.rho);for(int k=0;k<Ns;++k)minSpecies=std::min(minSpecies,q[c][5+k]/q[c][0]);
  if(contact){R density=x<.5+u*time?left[0]:right[0];rhoL1+=dx*std::abs(q[c][0]-density);pmax=std::max(pmax,std::abs(p.p.p-1e5)/1e5);pL1+=dx*std::abs(p.p.p-1e5)/1e5;uL1+=dx*std::abs(p.p.ux-u);}
  else {auto want=exact.sample((x-.5)/time);rhoL1+=dx*std::abs(p.p.rho-want[0]);pL1+=dx*std::abs(p.p.p-want[2])/1e5;uL1+=dx*std::abs(p.p.ux-want[1])/std::sqrt(R(1e5));}
  for(int k=0;k<5+Ns;++k)after[k]+=dx*q[c][k];fields<<x<<','<<p.p.rho<<','<<p.p.ux<<','<<p.p.uy<<','<<p.p.p<<','<<p.T<<','<<q[c][5]/q[c][0]<<'\n';
 }
 for(int k=0;k<5+Ns;++k)closure=std::max(closure,R(std::abs(after[k]-before[k]-boundary[k])/std::max(1.L,std::abs(before[k])+std::abs(boundary[k]))));
 require(closure<1e-11,"global conservative inventory budget");
 std::cout<<std::setprecision(17)<<"scenario,scheme,cells,direction,steps,minT,minRho,minY,rhoL1,pL1,uL1,pRelativeMax,conservativeClosure\n"<<scenario<<','<<scheme<<','<<n<<','<<direction<<','<<steps<<','<<minT<<','<<minRho<<','<<minSpecies<<','<<rhoL1<<','<<pL1<<','<<uL1<<','<<pmax<<','<<closure<<'\n';
}
