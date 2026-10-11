// New current-source regression coverage; not a recovered historical test suite.
#include "tests/TestSupport.H"
#include "materials/Reaction.H"
#include "coupling/ExchangeLedger.H"
using namespace chmt;using namespace chmt_test;
int main(){
 ModelConfig model;std::string error;check(validateSpeciesSet(model,error),"compiled species match");model.speciesNames[0]="wrong";check(!validateSpeciesSet(model,error),"species identity rejects mismatch");
 auto p=physics();const auto w=gas(p);
 for(double volume:{1e-12,1.,1e12}){
  auto q=conservativeGas(w,volume,p);GasPrimitive recovered;check(recoverGas(q,volume,p,recovered),"mixture inversion across volume scales");
  near(recovered.temperature,w.temperature,"gas temperature inversion");near(recovered.pressure,w.pressure,"gas EOS");
  near(q.energy,.5*q.mass*dot(w.velocity,w.velocity)+q.species[0]*speciesE(p.species[0],w.temperature)+q.species[1]*speciesE(p.species[1],w.temperature),"formation-inclusive energy");
  auto bad=q;bad.species[0]=-.1;recovered.temperature=987;check(!recoverGas(bad,volume,p,recovered),"negative species rejects");check(recovered.temperature==987,"failed recovery preserves output");
 }
 for(double T:{100.,300.,1700.,2000.}){double x=0;const double e=caloricIntegral(1e10,1000,.3,T);check(invertCaloric(1e10,1000,.3,e,100,2000,p.tolerances,x),"caloric offset inversion");near(x,T,"formation offset temperature",1e-10,1e-8);}
 double sentinel=456;check(!invertCaloric(0,1000,0,99999,100,2000,p.tolerances,sentinel),"out-of-range caloric energy rejects");check(sentinel==456,"invalid caloric leaves output");
 p.enableReactions=true;p.nReactions=1;auto& r=p.reactions[0];r.A=2;r.order[0]=1;r.condensedNu[0]=-1;r.condensedNu[1]=.6;r.gasNu[0]=.4;
 check(validateReaction(r,p,0,error),"stoichiometric mass and element closure");auto bad=r;bad.gasNu[0]=.5;check(!validateReaction(bad,p,0,error),"imbalanced mechanism rejects");
 SolidQ q;q.condensed[0]=.8;q.condensed[1]=.1;q.pore[0]=.01;q.porosity=.2;
 q.energy=.8*condensedE(p.condensed[0],700)+.1*condensedE(p.condensed[1],700)+.01*speciesE(p.species[0],700);
 const double U=q.energy;check(applyReactionExtent(r,0,.2,q),"reaction extent applied");near(q.condensed[0],.6,"donor extent");near(q.condensed[1],.22,"product extent");near(q.pore[0],.09,"pore product");check(q.energy==U,"reaction total U unchanged");near(q.progress[0],.2,"molar progress");
 auto prior=q;check(!applyReactionExtent(r,0,1,q),"donor overdraw rejects");check(q.condensed[0]==prior.condensed[0]&&q.energy==prior.energy,"rejected extent atomic");
 ExchangePacket packet;packet.kind=ExchangeKind::GasSolid;packet.gasCell=packet.solidCell=0;packet.mass=.2;packet.species[0]=.2;packet.condensed[0]=.2;packet.energy=30;packet.advective=10;packet.conductive=20;
 check(validatePacket(packet,p,error),"physical packet decomposition");packet.pressureWork=1;check(!validatePacket(packet,p,error),"unbalanced packet energy rejects");
 std::cout<<"core host regressions: species identity, calorics, reaction inventories, packet validation passed\n";
}
