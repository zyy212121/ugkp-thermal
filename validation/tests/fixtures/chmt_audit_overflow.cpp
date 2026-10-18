#include "coupling/ExchangeLedger.H"
#include "io/CoupledOutput.H"
#include <cassert>
#include <cfenv>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#ifdef CHMT_NATIVE_FOAM
#include "DiagnosticsFoam.H"
#include "sigFpe.H"
#endif
using namespace chmt;
using namespace ugkp::diagnostics;
PhysicsConfig physics(DeviceContext& d){PhysicsConfig p;p.diagnostics=&d;p.tolerances.diagnostics=&d;
 for(int s=0;s<Ns;++s){p.species[s].R=287;p.species[s].cp0=1000;p.species[s].Tmin=200;p.species[s].Tmax=4000;}
 p.liquid.rho=1e305;p.liquid.cp0=1e-305;p.liquid.Tmin=1;p.liquid.Tmax=1000;return p;}
void show(DeviceContext& d){
#ifdef CHMT_NATIVE_FOAM
 dispatch(takeSnapshot(d,{{3812,"applications/CHMT/materials/Thermodynamics.H","recoverFilm",1,"film inventory audit"},{4302,"applications/CHMT/coupling/ExchangeLedger.C","validatePacketMath",1,"packet conservation audit"},{3858,"applications/CHMT/materials/Thermodynamics.H","recoverGas",1,"gas physical state"}},"native CHMT helper"),makeFoamSink());
#endif
}
int main(int argc,char** argv){assert(argc==2);const std::string mode=argv[1];
#ifdef CHMT_NATIVE_FOAM
 Foam::sigFpe trap;trap.set(true);assert((fegetexcept()&(FE_OVERFLOW|FE_DIVBYZERO|FE_INVALID))==(FE_OVERFLOW|FE_DIVBYZERO|FE_INVALID));
#endif
 DeviceContext d{};reset(d,8,99,4,.25);auto p=physics(d);DeviceStatus status{};
 if(mode=="film"){
  FilmQ q;q.mass=1e308;q.species[0]=q.species[1]=q.mass;q.enthalpy=3e5;const auto before=q;FilmAux out;
  assert(recoverFilm(q,1,0,p,out,&status,7));assert(std::abs(out.temperature-300)<1e-10&&std::abs(out.thickness-1000)<1e-10);assert(!d.fatalClaim&&d.warnings[0].count==1&&status.code==0);assert(q.mass==before.mass&&q.enthalpy==before.enthalpy&&q.species[1]==before.species[1]);show(d);
  reset(d,8,99,4,.25);q.species[1]=0;assert(recoverFilm(q,1,0,p,out,&status,7));assert(!d.fatalClaim&&d.warnings[0].count==0&&std::abs(out.temperature-300)<1e-10);
  q.species[0]=std::numeric_limits<Real>::infinity();assert(!recoverFilm(q,1,0,p,out,&status,7));
 }else if(mode=="packet"){
  ExchangePacket q;q.gasCell=q.solidCell=0;const Real big=std::numeric_limits<Real>::max()*.75;
  q.conductive=big;q.advective=-big;q.pressureWork=big;q.viscousWork=-big;const auto before=q;
  assert(validatePacketMath(q,p.tolerances,&status));assert(!d.fatalClaim&&d.warnings[0].count==1);assert(q.conductive==before.conductive&&q.advective==before.advective);show(d);
  reset(d,8,99,4,.25);q.conductive=2;q.advective=-2;q.pressureWork=q.viscousWork=0;assert(validatePacketMath(q,p.tolerances,&status));assert(!d.fatalClaim&&d.warnings[0].count==0);
  reset(d,8,99,4,.25);q.conductive=q.advective=big;q.pressureWork=q.viscousWork=-big;assert(validatePacketMath(q,p.tolerances,&status));assert(!d.fatalClaim&&d.warnings[0].count==1);
  reset(d,8,99,4,.25);q.conductive=q.advective=q.pressureWork=q.viscousWork=0;q.species[0]=big;q.species[1]=-big;assert(validatePacketMath(q,p.tolerances,&status));assert(!d.fatalClaim&&d.warnings[0].count==1);
  reset(d,8,99,4,.25);q.species[0]=q.species[1]=0;q.pore[0]=big;q.pore[1]=-big;assert(validatePacketMath(q,p.tolerances,&status));assert(!d.fatalClaim&&d.warnings[0].count==1);
  reset(d,8,99,4,.25);q.pore[0]=q.pore[1]=0;q.conductive=2;q.advective=-2;q.energy=1;auto loose=p.tolerances;loose.absoluteEnergy=loose.relativeEnergy=std::numeric_limits<Real>::max();assert(validatePacketMath(q,loose,&status));assert(!d.fatalClaim&&d.warnings[0].count==1);
  reset(d,8,99,4,.25);GasQ gasInventory;gasInventory.mass=1e308;gasInventory.energy=1;gasInventory.species[0]=gasInventory.species[1]=1e308;assert(validGasInventory(gasInventory,&d));assert(!d.fatalClaim&&d.warnings[0].count==1);
  reset(d,8,99,4,.25);FilmQ filmInventory;filmInventory.mass=1e308;filmInventory.enthalpy=1;filmInventory.species[0]=filmInventory.species[1]=1e308;assert(validFilmInventory(filmInventory,false,&d));assert(!d.fatalClaim&&d.warnings[0].count==1);
  reset(d,8,99,4,.25);SolidQ solidInventory;solidInventory.energy=1;solidInventory.condensed[0]=solidInventory.condensed[1]=1e308;assert(validSolidInventory(solidInventory,&d));assert(!d.fatalClaim&&d.warnings[0].count==1);
  q.species[0]=std::numeric_limits<Real>::infinity();assert(!validatePacketMath(q,p.tolerances,&status));assert(d.firstFatal.code==Code::NonFinite);
 }else if(mode=="gas"){
  // Both component inventories and physical caloric/density results stay finite;
  // only their optional total-inventory audit exceeds binary64's range.
  auto largePhysics=p;for(auto& species:largePhysics.species){species.R=1e-305;species.cp0=1e-304;}
  GasQ large;large.mass=1e308;large.species[0]=large.species[1]=large.mass;large.energy=5.4e6;GasPrimitive largeOut;
  assert(recoverGas(large,1e308,largePhysics,largeOut,nullptr,3));assert(!d.fatalClaim&&d.warnings[0].count==1);assert(std::abs(largeOut.temperature-300)<1e-10&&largeOut.rho==1);
  reset(d,8,99,4,.25);
  ModelConfig model;model.physics=p;HostState state;state.gas.resize(2);state.gasMesh.volumes={1,1};
  for(auto& q:state.gas){q.mass=1;q.species[0]=1;q.energy=(1000-287)*300;}
  std::ostringstream good;std::string error;assert(chmt::io::writeAcceptedGasFields(good,state,model,error));assert(!d.fatalClaim);
  reset(d,8,99,4,.25);state.gasMesh.volumes[1]=1e-310;std::ostringstream bad;assert(!chmt::io::writeAcceptedGasFields(bad,state,model,error));assert(d.fatalClaim&&d.firstFatal.code==Code::NonFinite);assert(std::isinf(d.firstFatal.observed)&&d.firstFatal.observed>0);assert(d.firstFatal.entityKind==EntityKind::Cell&&d.firstFatal.entityIndex==1&&d.firstFatal.quantityId==Quantity::Density);assert(d.firstFatal.step==4&&d.firstFatal.simulationTime==.25);
#ifdef CHMT_NATIVE_FOAM
 Foam::FatalError.throwExceptions();bool caught=false;try{show(d);}catch(const Foam::error& e){caught=true;assert(e.message().find("observed=inf")!=std::string::npos);assert(e.message().find("entity=1")!=std::string::npos);assert(e.message().find("density")!=std::string::npos);}assert(caught);
#endif
 }else return 2;
 std::cout<<"verified "<<mode<<"\n";
}
