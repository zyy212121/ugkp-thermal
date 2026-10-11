#include "common/gasWall/WallModel.H"
#include "common/chemistry/mechanisms/GeneratedH2O2.H"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <type_traits>
using namespace ugkwp;
using L=long double;
constexpr L RU=8.31446261815324L;
struct Values {L cp,h,s;};
Values nasa(const SpeciesThermoView<float,10>& t,int i,L T){
 const auto& sp=t.species[i];const float* c=t.coefficients+sp.coefficientOffset+(T>L(sp.midTemperature)?7:0);
 L cp=0,h=L(c[5])/T,s=L(c[6])+L(c[0])*logl(T);
 for(int j=0;j<5;++j){cp+=L(c[j])*powl(T,j);h+=L(c[j])*powl(T,j)/L(j+1);if(j)s+=L(c[j])*powl(T,j)/L(j);}
 return {RU/L(sp.molarMass)*cp,RU/L(sp.molarMass)*T*h,RU/L(sp.molarMass)*s};
}
L arr(const GasArrheniusData<float>& a,L T){return L(a.preExponential)*powl(T,L(a.temperatureExponent))*expl(-L(a.activationEnergy)/(RU*T));}
L product(const GasStoichTerm<float>* terms,int offset,int n,const double* c){L p=1;for(int i=offset;i<offset+n;++i)p*=powl(L(c[terms[i].species]),L(terms[i].coefficient));return p;}
L reference(int r,L T,const double* c,const SpeciesThermoView<float,10>& t,const GasMechanismView<float,10>& m){
 const auto& a=m.reactions[r];L k=arr(a.highRate,T);
 if(a.type!=GasReactionType::Elementary){L collider=0;for(int i=0;i<10;++i){L eff=a.defaultEfficiency;for(int j=a.efficiencyOffset;j<a.efficiencyOffset+a.efficiencyCount;++j)if(m.efficiencies[j].species==i)eff=m.efficiencies[j].efficiency;collider+=eff*L(c[i]);}
  if(a.type==GasReactionType::ThirdBody)k*=collider;
  else {const L low=arr(a.lowRate,T),pr=low*collider/k;L F=1;
   if(a.type==GasReactionType::Troe){L center=(1-L(a.troe.alpha))*expl(-T/L(a.troe.T3))+L(a.troe.alpha)*expl(-T/L(a.troe.T1));if(a.troe.hasT2)center+=expl(-L(a.troe.T2)/T);
    const L logF=log10l(center),ct=-.4L-.67L*logF,nt=.75L-1.27L*logF,shifted=log10l(pr)+ct,shape=shifted/(nt-.14L*shifted);F=powl(10.L,logF/(1+shape*shape));}
   k=low*collider/(1+pr)*F;}}
 const L f=k*product(m.reactants,a.reactantOffset,a.reactantCount,c);L b=0;
 if(a.reversible){L dg=0,dnu=0;for(int side=0;side<2;++side){const auto* terms=side?m.products:m.reactants;int off=side?a.productOffset:a.reactantOffset,n=side?a.productCount:a.reactantCount;for(int j=off;j<off+n;++j){const auto& term=terms[j];L sign=(side?1.L:-1.L)*L(term.coefficient);auto v=nasa(t,term.species,T);dg+=sign*L(t.species[term.species].molarMass)*(v.h-T*v.s);dnu+=sign;}}
  const L kc=expl(-dg/(RU*T))*powl(L(m.referencePressure)/(RU*T),dnu);b=k/kc*product(m.products,a.productOffset,a.productCount,c);}
 return f-b;
}
int main(){GeneratedH2O2<float> model;auto t=model.thermoView();auto m=model.mechanismView();static_assert(std::is_same<decltype(speciesCp(0,700.,t)),double>::value,"mixed table property narrows");L maxProp=0,maxRate=0;int checks=0,troe=0;
 for(double T:{700.,999.99999,1000.,1000.00001,1500.,3000.}){
  for(int i=0;i<10;++i){auto v=nasa(t,i,L(T));double values[]={speciesCp(i,T,t),speciesH(i,T,t),speciesEntropyStandard(i,T,t),speciesGibbsStandard(i,T,t)};L refs[]={v.cp,v.h,v.s,v.h-L(T)*v.s};for(int j=0;j<4;++j){L e=fabsl(L(values[j])-refs[j])/std::max(1.L,fabsl(refs[j]));maxProp=std::max(maxProp,e);assert(e<1e-11L);++checks;}}
  for(double scale:{1e-4,1.,1e4}){double c[10];for(int i=0;i<10;++i)c[i]=scale*.02*(i+1);for(int r=0;r<m.reactionCount;++r){double q;assert(gasChemistryDetail::progressRate(r,T,c,t,m,q));L ref=reference(r,L(T),c,t,m);L e=fabsl(L(q)-ref)/std::max(1.L,fabsl(ref));maxRate=std::max(maxRate,e);if(!(e<1e-10L)){std::cerr<<r<<' '<<T<<' '<<double(e)<<'\n';return 2;}if(m.reactions[r].type==GasReactionType::Troe)++troe;++checks;}}}
 std::cout<<std::setprecision(17)<<"checks="<<checks<<" troe="<<troe<<" max_property_relative="<<double(maxProp)<<" max_rate_relative="<<double(maxRate)<<'\n';
}
