#include "tests/TestSupport.H"
#include "coupling/ExchangeLedger.H"
#include "gpu/ResourceTransaction.H"
#include <memory>
using namespace chmt;using namespace chmt_test;
struct Fault{std::string message;};struct Resource{int id;Fault* fault;Resource(int i,Fault* f):id(i),fault(f){}void rebindFault(Fault& f)noexcept{fault=&f;}};
int main(){
 std::string error;GasQ gas;gas.mass=1;gas.species[0]=1;gas.energy=1e5;SolidQ solid;solid.condensed[0]=2;solid.energy=4e5;
 ExchangePacket p;p.kind=ExchangeKind::GasSolid;p.gasCell=p.solidCell=0;p.mass=.25;p.species[0]=p.condensed[0]=.25;p.energy=p.advective=1000;p.momentum={3,4,5};
 check(consumePacketTransaction(p,&gas,&solid,nullptr,nullptr,error),"paired packet transaction");near(gas.mass,1.25,"gas receives mass");near(solid.condensed[0],1.75,"solid loses mass");near(gas.energy+solid.energy,5e5,"paired physical energy");check(p.consumerMask==(ConsumeGas|ConsumeSolid),"once-only masks");
 near(gas.energy,101000,"gas receives packet energy");near(solid.energy,399000,"solid loses packet energy");near(gas.momentum.x,3,"gas receives packet impulse");near(gas.species[0],1.25,"gas receives packet species");
 const GasQ g0=gas;const SolidQ s0=solid;
 check(!consumePacketTransaction(p,&gas,&solid,nullptr,nullptr,error),"duplicate refuses");check(gas.mass==g0.mass&&solid.energy==s0.energy,"duplicate atomic");
 p.consumerMask=0;p.mass=p.species[0]=p.condensed[0]=3;
 check(!consumePacketTransaction(p,&gas,&solid,nullptr,nullptr,error),"receiver-valid donor-overdraw refuses");check(gas.mass==g0.mass&&solid.condensed[0]==s0.condensed[0]&&p.consumerMask==0,"failure does not publish first consumer");
 FilmQ film;film.mass=1;film.species[0]=1;film.enthalpy=110;
 FilmPressureVolumeUpdate storage;storage.filmFace=0;storage.oldPV=10;storage.newPV=30;
 check(consumeFilmPressureVolume(storage,film,error),"physical pressure-volume conversion");
 near(film.enthalpy,130,"pV increment updates enthalpy");near(film.enthalpy-storage.newPV,100,"pV preserves thermal inventory");
 check(storage.consumed,"storage once-only marker");check(!consumeFilmPressureVolume(storage,film,error),"duplicate storage refuses");near(film.enthalpy,130,"duplicate storage atomic");
 ExchangePacket phase;phase.kind=ExchangeKind::SolidFilm;phase.solidCell=phase.filmFace=0;phase.mass=.25;phase.species[0]=phase.condensed[0]=.25;phase.energy=phase.advective=25;
 FilmPressureVolumeUpdate paired;paired.filmFace=0;paired.oldPV=30;paired.newPV=40;
 const Real solidBefore=solid.energy,thermalBefore=film.enthalpy-paired.oldPV;
 check(consumePacketTransaction(phase,nullptr,&solid,&film,&paired,error),"phase with physical storage transaction");
 near(film.mass,1.25,"phase film mass");near(film.species[0],1.25,"phase film species");near(solid.condensed[0],1.5,"phase solid donor");
 near(film.enthalpy-paired.newPV,thermalBefore+25,"phase actual thermal inventory");near(solid.energy,solidBefore-25,"phase donor energy");
 check(paired.consumed&&phase.consumerMask==(ConsumeSolid|ConsumeFilm),"phase consumer markers");
 Fault active,candidate;std::unique_ptr<Resource>a(new Resource(1,&active)),b(new Resource(2,&active)),c(new Resource(3,&active));
 std::unique_ptr<Resource>x(new Resource(4,&candidate)),y(new Resource(5,&candidate)),z;
 check(!publishResourceReplacement(a,b,c,x,y,z,active,candidate,error),"incomplete resources reject");check(a->id==1&&x->id==4,"incomplete resources preserve owners");
 z.reset(new Resource(6,&candidate));candidate.message="injected prepare failure";check(!publishResourceReplacement(a,b,c,x,y,z,active,candidate,error),"candidate fault refuses");check(a->id==1&&b->id==2&&c->id==3,"fault preserves active owners");
 candidate.message.clear();check(publishResourceReplacement(a,b,c,x,y,z,active,candidate,error),"prepared resources publish");check(a->id==4&&b->id==5&&c->id==6&&a->fault==&active&&b->fault==&active&&c->fault==&active,"all resources and fault bindings publish together");
 std::cout<<"runtime transaction regressions: packet and resource atomicity passed\n";
}
