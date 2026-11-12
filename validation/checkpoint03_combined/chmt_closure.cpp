#include "ablation/InterfaceMath.H"
#include "ablation/DrySurfaceInterfaceMath.H"
#include "film/FilmMath.H"
#include "coupling/ExchangeLedger.H"
#include "chmt_support.H"
namespace {
using namespace chmt;using namespace chmt_accept;
static_assert(sizeof(decltype(GasQ{}.mass))==sizeof(E),"physical mass uses selected precision");
static_assert(sizeof(decltype(SolidQ{}.condensed[0]))==sizeof(E),"material inventory uses selected precision");
static_assert(sizeof(decltype(FilmQ{}.mass))==sizeof(E),"film inventory uses selected precision");
static_assert(sizeof(decltype(ExchangePacket{}.energy))==sizeof(E),"packet physical precision");
static_assert(sizeof(decltype(FilmQ{}.enthalpy))==sizeof(E),"film physical precision");
static_assert(sizeof(decltype(SurfacePhysicsResult{}.thermalResidual))==sizeof(E),"real closure residual precision");
PhysicsConfig physics(){
 PhysicsConfig p;p.wallModel.family=ugkwp::gaswall::WallFamily::LowRe;
 p.material.enableMelting=false;p.material.enableSurfaceReactions=false;p.material.enablePoreOutflow=false;
 p.enableReactions=false;p.enableFilm=false;p.enableRadiation=true;p.emissivity=.65;p.ambientTemperature=650;
 p.gasConductivity=2;p.gasViscosity=1.8e-5;
 for(int i=0;i<Ns;++i){auto& t=p.species[i];t.R=287+11*i;t.cp0=1100+30*i;t.Tmin=100;t.Tmax=2000;}
 for(int i=0;i<Nc;++i){auto& t=p.condensed[i];t.rho=1000+50*i;t.cp0=1000;t.conductivity=3;t.Tmin=100;t.Tmax=2000;}
 p.liquid=p.condensed[0];p.liquidReferencePressure=100000;
 return p;
}
double sumEnergy(const GasQ& g,const SolidQ& s,const FilmQ& f){return double(g.energy)+double(s.energy)+double(f.enthalpy);}
void thermal(bool hot){
 auto p=physics();const E Tg=hot?E(1000):E(400),Ts=hot?E(600):E(900);
 p.ambientTemperature=hot?750:700;
 SurfacePhysicsInput in;in.hasSolid=true;in.hasFilm=false;in.ablating=false;
 // Data literals enter production field types once. Physical mass/composition
 // follow selected precision; only owner-declared geometry/time boundaries remain double.
 in.area=in.gasArea=in.solidArea=.125;in.gasDistance=.02;in.solidDistance=.03;in.dt=.01;
 in.normal=in.gasNormal=in.solidNormal={1,0,0};in.solidVolume=.001;
 in.solid.condensed[0]=1;in.solid.energy=E(1000)*Ts;in.solidBase=in.solid;
 in.bulk.rho=1.234567890123;in.bulk.Y[0]=.7;in.bulk.Y[1]=.3;in.bulk.temperature=Tg;
 in.bulk.pressure=E(in.bulk.rho)*E(.7*287+.3*298)*Tg;
 in.gasInventory=conservativeGas(in.bulk,1,p);
 SurfacePhysicsResult full,dry;DeviceStatus status;bool numerical=false;
 const std::string label=hot?"thermal_hot":"thermal_cold";
 require(solveSurfaceInterface(in,p,full,&status,&numerical),label+" real surface solve");
 require(solveDrySurfaceInterface(in,p,dry,&status,&numerical),label+" real dry solve");
 const double lo=std::min(double(Tg),double(Ts)),hi=std::max(double(Tg),double(Ts));
 for(const auto* r:{&full,&dry}){
  const double t=r->topTemperature;
  require(std::isfinite(t)&&t>lo&&t<hi,label+" bounded nontrivial thermal direction");
  require(hot?r->gasConductive<0:r->gasConductive>0,label+" signed heat direction");
  require(r->radiationFlux!=0,label+" independently nonzero radiative contribution");
  const double solidHeat=100*(double(Ts)-t),gasHeat=100*(t-double(Tg));
  const double ambient=p.ambientTemperature;
  const double rad=5.670374419e-8*.65*(ambient*ambient*ambient*ambient-t*t*t*t);
  const double scale=std::abs(gasHeat)+std::abs(solidHeat)+std::abs(rad);
  const double old=4*(p.tolerances.absoluteEnergy/(in.dt*in.area)+p.tolerances.relativeEnergy*scale);
  near(label+(r==&full?".full_balance":".dry_balance"),"W/m2",gasHeat-solidHeat-rad,0,scale,old,256);
  near(label+(r==&full?".full_residual":".dry_residual"),"W/m2",r->thermalResidual,0,scale,old,256);
 }
 near(label+".adapter_temperature","K",dry.topTemperature,full.topTemperature,hi,1e-7,64);
 SurfacePacketIdentity id;id.gasCell=0;id.solidCell=0;id.filmFace=0;id.face=47;
 ExchangePacket packets[3];require(assembleSurfacePackets(in,dry,id,p,packets,&status),label+" physical packet assembly");
 GasQ g=in.gasInventory;SolidQ s=in.solid;FilmQ f;
 const double before=sumEnergy(g,s,f),mass=double(g.mass)+s.condensed[0],oldGas=g.energy,oldSolid=s.energy;
 require(applyPacketTransaction(packets[0],&g,&s,nullptr,nullptr,p.tolerances,&status),label+" packet consumption");
 near(label+".ledger_energy","J",sumEnergy(g,s,f)-before,0,std::abs(before)+std::abs(double(packets[0].energy)),1e-9,8);
 near(label+".ledger_mass","kg",double(g.mass)+s.condensed[0],mass,mass,1e-14,4);
 const double transfer=packets[0].energy;
 require(std::abs(transfer)>16*double(std::numeric_limits<E>::epsilon())*std::max(std::abs(oldGas),std::abs(oldSolid)),label+" exchange is resolvable above state ULP");
 near(label+".gas_transfer","J",double(g.energy)-oldGas,transfer,std::abs(oldGas)+std::abs(transfer),1e-9,2);
 near(label+".material_transfer","J",double(s.energy)-oldSolid,-transfer,std::abs(oldSolid)+std::abs(transfer),1e-9,2);
 require(hot?(g.energy<oldGas&&s.energy>oldSolid):(g.energy>oldGas&&s.energy<oldSolid),label+" inventory exchange actually executed");
 const double expected=double(dry.gasEnergy)*in.area*in.dt;
 near(label+".packet_units","J",packets[0].energy,expected,std::abs(expected),1e-10,8);
 require(packets[0].consumerMask==(ConsumeGas|ConsumeSolid),label+" consumers");
}
void filmAdvection(){
 FilmQ q;q.mass=.12;q.species[0]=.084;q.species[1]=.036;q.enthalpy=E(42000.1234);
 for(E velocity:{E(.037),E(-.019)}){
  const FilmQ flux=filmAdvectiveFlux(q,velocity);const double expected=double(q.enthalpy)*velocity;
  near(velocity>0?"film_advection.forward":"film_advection.reverse","W/m",flux.enthalpy,expected,std::abs(expected),1e-10,4);
  near("film_advection.mass","kg/(m s)",flux.mass,double(q.mass)*velocity,std::abs(double(q.mass)*velocity),1e-14,4);
  require((flux.enthalpy>0)==(velocity>0),"film advective direction");
 }
}
void pressureVolume(){
 const E p0=E(100123.125),p1=E(107321.75),v0=E(.00012345),v1=E(.00015678);
 const E delta=pressureVolumeProduct(p0,v0,p1,v1);
 const double expected=double(p1)*v1-double(p0)*v0,scale=std::abs(double(p1)*v1)+std::abs(double(p0)*v0);
 near("pressure_volume.identity","J",delta,expected,scale,1e-12,16);
 FilmQ f;f.mass=.1;f.species[0]=.1;f.enthalpy=E(5000.125);
 FilmPressureVolumeUpdate storage;storage.filmFace=0;storage.oldPV=p0*v0;storage.newPV=p1*v1;
 const double before=f.enthalpy;std::string error;
 require(consumeFilmPressureVolume(storage,f,error),error);
 near("pressure_volume.storage","J",double(f.enthalpy)-before,double(storage.newPV)-storage.oldPV,
 std::abs(before)+std::abs(double(storage.oldPV))+std::abs(double(storage.newPV)),1e-10,8);
 require(storage.consumed,"physical storage consumed");
}
void interfaceChannels(){
 const E J=E(.0123),h=E(120034.5),p=E(103421.5),speed=E(.00017),visc=E(1.75),heat=E(-53.25);
 const decltype(GasPrimitive{}.velocity) velocity{2.5,-1.25,.375};
 const E speciesH=J*h;
 const E output=physicalInterfaceEnergy(J,speciesH,velocity,p,speed,visc,heat);
 const double kinetic=.5*double(J)*(2.5*2.5+1.25*1.25+.375*.375);
 const double expected=double(speciesH)+kinetic+double(p)*speed-visc+heat;
 const double scale=std::abs(double(speciesH))+std::abs(kinetic)+std::abs(double(p)*speed)+std::abs(double(visc))+std::abs(double(heat));
 near("interface_channels.total","W/m2",output,expected,scale,1e-10,32);
 const E liquid=liquidInterfaceHeat(J,h,velocity,p,speed,visc,output);
 near("interface_channels.liquid_heat","W/m2",liquid,-double(heat),scale,1e-10,48);
 const E solid=solidPhaseEnergy(J,h,p,speed,E(1000),heat);
 near("interface_channels.solid_phase","W/m2",solid,double(J)*(double(h)+double(p)/1000)+double(p)*speed+heat,scale,1e-10,24);
 require(output>0&&liquid>0,"nonzero convective and conductive directions");
}
void radiation(){
 auto p=physics();p.ambientTemperature=700;
 for(E t:{E(450),E(950)}){
  const E q=grayRadiation(t,p);const double d=t,a=p.ambientTemperature;
  const double expected=5.670374419e-8*.65*(a*a*a*a-d*d*d*d);
  near(t<700?"radiation.heating":"radiation.cooling","W/m2",q,expected,std::abs(expected),1e-8,16);
  require((q>0)==(t<700),"gray radiation hot/cold direction");
 }
}
void wetLedger(){
 auto config=physics();GasQ g;g.mass=1;g.species[0]=.7;g.species[1]=.3;g.energy=E(700001.25);
 SolidQ s;s.condensed[0]=1;s.energy=E(900003.5);
 FilmQ f;f.mass=.1;f.species[0]=.07;f.species[1]=.03;f.enthalpy=E(30001.75);
 const double initialGas=g.energy,initialSolid=s.energy,initialFilm=f.enthalpy,initialGasMass=g.mass,initialSolidMass=s.condensed[0];
 double gasTransfer=0,solidTransfer=0,filmTransfer=0,massTransfer=0;
 double observation=sumEnergy(g,s,f),mass=double(g.mass)+s.condensed[0]+f.mass,expected=observation,scale=std::abs(observation);
 // Single production scalar updates only. The double total below is read-only
 // test observation; it is never fed back to g, s, f or any packet.
 for(int i=0;i<64;++i){
  ExchangePacket x;x.step=i;x.face=9;x.filmFace=x.gasCell=x.solidCell=0;
  x.kind=i%2?ExchangeKind::SolidFilm:ExchangeKind::GasFilm;
  x.mass=.00001;x.species[0]=.000007;x.species[1]=.000003;
  if(x.kind==ExchangeKind::SolidFilm)x.condensed[0]=x.mass;
  x.conductive=E(-1.125);x.advective=E(3.375);x.pressureWork=E(.625);
  x.viscousWork=E(-.1875);x.radiation=E(.3125);x.liquidKineticAdvection=E(.015625);
  x.energy=x.conductive+x.advective+x.pressureWork+x.viscousWork+x.radiation;
  FilmPressureVolumeUpdate pv;pv.step=x.step;pv.face=x.face;pv.filmFace=0;
  pv.oldPV=E(10.125);pv.newPV=E(10.15625);
  DeviceStatus status;require(applyPacketTransaction(x,&g,&s,&f,&pv,config.tolerances,&status),"wet ledger transaction");
  if(x.kind==ExchangeKind::GasFilm){gasTransfer+=x.energy;filmTransfer-=x.energy;massTransfer+=x.mass;}
  else{solidTransfer-=x.energy;filmTransfer+=x.energy;}
  filmTransfer+=double(x.liquidKineticAdvection)+double(pv.newPV)-pv.oldPV;
  expected+=double(x.liquidKineticAdvection)+double(pv.newPV)-pv.oldPV;
  scale+=std::abs(double(x.energy))+std::abs(double(x.liquidKineticAdvection))+std::abs(double(pv.newPV)-pv.oldPV);
  require(x.consumerMask==requiredConsumers(x.kind)&&pv.consumed,"wet ledger ownership");
 }
 near("wet_ledger.cumulative_energy","J",sumEnergy(g,s,f),expected,scale,1e-8,256);
 near("wet_ledger.cumulative_mass","kg",double(g.mass)+s.condensed[0]+f.mass,mass,mass,1e-12,128);
 near("wet_ledger.gas_transfer","J",double(g.energy)-initialGas,gasTransfer,std::abs(initialGas)+std::abs(gasTransfer),1e-8,32);
 near("wet_ledger.solid_transfer","J",double(s.energy)-initialSolid,solidTransfer,std::abs(initialSolid)+std::abs(solidTransfer),1e-8,32);
 near("wet_ledger.film_transfer","J",double(f.enthalpy)-initialFilm,filmTransfer,std::abs(initialFilm)+std::abs(filmTransfer),1e-8,128);
 require(std::abs(filmTransfer)>allowance(std::abs(initialFilm)+std::abs(filmTransfer),1e-8,128),"film trend cannot hide inside epsilon budget");
 near("wet_ledger.gas_mass_transfer","kg",double(g.mass)-initialGasMass,massTransfer,initialGasMass,1e-12,64);
 near("wet_ledger.solid_mass_transfer","kg",double(s.condensed[0])-initialSolidMass,-massTransfer,initialSolidMass,1e-12,64);
 require(g.mass>initialGasMass&&s.condensed[0]<initialSolidMass,"physical mass exchange actually executed");
 require(f.enthalpy!=E(30001.75)&&g.energy!=E(700001.25)&&s.energy!=E(900003.5),"all inventories nonzero updates");
}
void phaseRate(){
 for(E power:{E(1357.25),E(-987.75)}){
  const E latent=E(234567.5),jump=E(.75);
  const E rate=phaseRateFromHeat(power,latent,jump);
  const double r=rate,energy=r*(double(latent)+.5*r*r*double(jump)*jump);
  near(power>0?"phase_rate.heating":"phase_rate.cooling","W/m2",energy,power,std::abs(double(power)),1e-9,48);
  require((rate>0)==(power>0)&&std::isfinite(r),"phase solve signed finite root");
 }
}
}
int main(){try{thermal(true);thermal(false);filmAdvection();pressureVolume();interfaceChannels();radiation();wetLedger();phaseRate();finish("chmt_closure");return 0;}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
