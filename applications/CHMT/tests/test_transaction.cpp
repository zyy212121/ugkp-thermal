#include "tests/TestSupport.H"
#include "coupling/ExchangeLedger.H"
#include "gpu/ResourceTransaction.H"
#include <memory>
using namespace chmt;using namespace chmt_test;
struct Fault{std::string message;};struct Resource{int id;Fault* fault;Resource(int i,Fault* f):id(i),fault(f){}void rebindFault(Fault& f)noexcept{fault=&f;}};
int main(){
 std::string error;GasQ gas;gas.mass=1;gas.species[0]=1;gas.energy=1e5;SolidQ solid;solid.condensed[0]=2;solid.energy=4e5;Budget budget;
 ExchangePacket p;p.kind=ExchangeKind::GasSolid;p.gasCell=p.solidCell=0;p.mass=.25;p.species[0]=p.condensed[0]=.25;p.energy=p.advective=1000;
 check(consumePacketTransaction(p,&gas,&solid,nullptr,nullptr,budget,error),"paired packet transaction");near(gas.mass,1.25,"gas receives mass");near(solid.condensed[0],1.75,"solid loses mass");near(gas.energy+solid.energy,5e5,"paired physical energy");check(p.consumerMask==(ConsumeGas|ConsumeSolid)&&budget.consumedPackets==1,"once-only masks");
 const GasQ g0=gas;const SolidQ s0=solid;const Budget b0=budget;
 check(!consumePacketTransaction(p,&gas,&solid,nullptr,nullptr,budget,error),"duplicate refuses");check(gas.mass==g0.mass&&solid.energy==s0.energy&&budget.consumedPackets==b0.consumedPackets,"duplicate atomic");
 p.consumerMask=0;p.mass=p.species[0]=p.condensed[0]=3;
 check(!consumePacketTransaction(p,&gas,&solid,nullptr,nullptr,budget,error),"receiver-valid donor-overdraw refuses");check(gas.mass==g0.mass&&solid.condensed[0]==s0.condensed[0]&&budget.exchangeMass[0]==b0.exchangeMass[0]&&p.consumerMask==0,"failure does not publish first consumer");
 Fault active,candidate;std::unique_ptr<Resource>a(new Resource(1,&active)),b(new Resource(2,&active)),c(new Resource(3,&active));
 std::unique_ptr<Resource>x(new Resource(4,&candidate)),y(new Resource(5,&candidate)),z;
 check(!publishResourceReplacement(a,b,c,x,y,z,active,candidate,error),"incomplete resources reject");check(a->id==1&&x->id==4,"incomplete resources preserve owners");
 z.reset(new Resource(6,&candidate));candidate.message="injected prepare failure";check(!publishResourceReplacement(a,b,c,x,y,z,active,candidate,error),"candidate fault refuses");check(a->id==1&&b->id==2&&c->id==3,"fault preserves active owners");
 candidate.message.clear();check(publishResourceReplacement(a,b,c,x,y,z,active,candidate,error),"prepared resources publish");check(a->id==4&&b->id==5&&c->id==6&&a->fault==&active&&b->fault==&active&&c->fault==&active,"all resources and fault bindings publish together");
 std::cout<<"runtime transaction regressions: packet and resource atomicity passed\n";
}
