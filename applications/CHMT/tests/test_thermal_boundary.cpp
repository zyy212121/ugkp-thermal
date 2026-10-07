#include "tests/TestSupport.H"
#include "gas/BoundarySample.H"
#include "gas/SstMath.H"
#include "core/ThermalBoundary.H"
using namespace chmt;
int main(){
 auto p=chmt_test::physics();p.gasConductivity=5;p.gasViscosity=.001;p.enableSst=true;
 auto owner=chmt_test::gas(p),prescribed=owner;owner.velocity=prescribed.velocity={};
 // Stored imported zeroGradient value is 730 K. The adjacent cell then changes.
 owner.temperature=900;GasPrimitive recovered;chmt_test::check(recoverGas(conservativeGas(owner,1,p),1,p,recovered),"changed owner EOS");owner=recovered;
 BoundaryKind kind=BoundaryKind::NoSlip;Vec3 area{1,0,0};Real swept=0;GasView gas;gas.primitive=&owner;
 GeometryView geometry;geometry.boundaryKind=&kind;geometry.boundaryPrimitive=&prescribed;geometry.areaVector=&area;geometry.sweptVolume=&swept;
 chmt_test::near(boundarySample(0,0,gas,geometry,1).temperature,owner.temperature,"zeroGradient sample follows changed owner despite positive stored primitive");
 ThermalBoundaryKind policy=ThermalBoundaryKind::ZeroGradient;geometry.thermalBoundary=&policy;
 GasGradient gradient;gradient.temperature={(prescribed.temperature-owner.temperature)/.25,3,5};
 for(Real nut:{0.,.01}) {
  chmt_test::near(impermeableWallDiffusiveFlux(owner,prescribed,{},gradient,area,false,true,p,nut).energy,0,"adiabatic laminar/SST heat flux is exactly zero");
  // Resolved low-Re wall sets eddy conductivity to zero at the wall.
  const Real exact=p.gasConductivity*(owner.temperature-prescribed.temperature)/.25;
  chmt_test::near(impermeableWallDiffusiveFlux(owner,prescribed,{},gradient,area,true,true,p,nut).energy,exact,"fixed-temperature conductive flux independent reference");
 }
 policy=ThermalBoundaryKind::FixedValue;
 chmt_test::near(boundarySample(0,0,gas,geometry,1).temperature,2*prescribed.temperature-owner.temperature,"fixedValue retains reflected thermal gradient");
 GasPrimitive trace;Real omega=0;
 chmt_test::check(wallThermodynamicTrace(owner,prescribed,false,p,trace),"adiabatic endpoint EOS trace");
 chmt_test::near(trace.temperature,owner.temperature,"SST endpoint trace follows owner temperature");
 chmt_test::near(trace.rho,owner.rho,"SST endpoint trace follows owner density");
 chmt_test::check(wallOmegaTarget(trace,.1,p,omega)&&omega>0,"positive SST target from adiabatic trace");
 chmt_test::check(wallThermodynamicTrace(owner,prescribed,true,p,trace),"fixed endpoint EOS trace");
 chmt_test::near(trace.temperature,prescribed.temperature,"fixed trace prescribed temperature");
 chmt_test::near(trace.pressure,owner.pressure,"fixed trace owner pressure EOS");
 chmt_test::check(importedThermalBoundary("zeroGradient",kind,true)==ThermalBoundaryKind::ZeroGradient,"native patch policy decoding");
 chmt_test::check(importedThermalBoundary("fixedValue",kind,false)==ThermalBoundaryKind::FixedValue,"solid fixed patch policy decoding");
 for(const auto& type:{"fixedGradient","mixed","inletOutlet","calculated"}) {
  bool refused=false;try{(void)importedThermalBoundary(type,kind,true);}catch(const std::runtime_error&){refused=true;}
  chmt_test::check(refused,"unsupported thermal operator is refused");
 }
 HostMesh mesh;mesh.owner={0};mesh.boundaryKind={kind};mesh.boundaryPrimitive={prescribed};std::string error;
 chmt_test::check(validateThermalBoundaries(mesh,error),"empty homogeneous zeroGradient shorthand");
 mesh.thermalBoundary={ThermalBoundaryKind::FixedValue};chmt_test::check(validateThermalBoundaries(mesh,error),"positive fixed temperature valid");
 mesh.thermalBoundary.push_back(policy);chmt_test::check(!validateThermalBoundaries(mesh,error),"wrong policy count refused");
 mesh.thermalBoundary={static_cast<ThermalBoundaryKind>(99)};chmt_test::check(!validateThermalBoundaries(mesh,error),"invalid policy enum refused");
 mesh.thermalBoundary={policy};mesh.boundaryPrimitive[0].temperature=0;chmt_test::check(!validateThermalBoundaries(mesh,error),"fixed nonpositive temperature refused");
 mesh.boundaryPrimitive[0]=prescribed;mesh.boundaryKind[0]=BoundaryKind::Interface;chmt_test::check(!validateThermalBoundaries(mesh,error),"interface thermal policy conflict refused");
 std::cout<<"thermal boundary host regressions: evolving zeroGradient/fixedValue, independent flux, SST EOS, invalid policy passed\n";
}
