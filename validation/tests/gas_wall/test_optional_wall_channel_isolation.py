"""Wall diagnostics preserve physical results and the standalone bool failure channel.

Host-only probes compile real production headers; no CUDA behavior is implied.
"""
from pathlib import Path
import os
import shlex
import subprocess

import pytest
from validation.tools.paths import SOURCE_ROOT

ROOT = Path(os.environ.get("UGKP_SOURCE_ROOT", str(SOURCE_ROOT)))

PROBES = {
    'owner_integral': r'''


#include "common/gasWall/WallModel.H"
#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>
using namespace ugkwp;
using namespace ugkwp::gaswall;
struct Model {
 SpeciesThermoData<double> sp[2]; double coef[6]={1000,0,0,1000,0,0};
 double elements[2]={1,1}, basis[2]={-1,1};
 GasReactionData<double> rx[1]; GasStoichTerm<double> re[1],pr[1];
 WallInput<double,2> in;
 Model(){for(int s=0;s<2;++s){sp[s].coefficientOffset=3*s;sp[s].molarMass=.028;sp[s].minTemperature=200;sp[s].maxTemperature=4000;sp[s].referencePressure=101325;sp[s].referenceTemperature=300;sp[s].referenceEntropy=1500;sp[s].hasEntropyReference=true;}
 auto& t=in.model.thermo;t.species=sp;t.coefficients=coef;t.coefficientCount=6;t.elementCount=1;t.elementComposition=elements;t.speciesOrderHash=1;
 rx[0].reactantCount=1;rx[0].productCount=1;rx[0].highRate.preExponential=2;re[0].species=0;re[0].coefficient=1;pr[0].species=1;pr[0].coefficient=1;
 auto& m=in.model.mechanism;m.reactions=rx;m.reactionCount=1;m.reactants=re;m.reactantTermCount=1;m.products=pr;m.productTermCount=1;m.referencePressure=101325;m.speciesOrderHash=1;m.stoichiometricBasis=basis;m.independentRank=1;
 in.integrateOwnerKSource=true; // This fixture explicitly tests auxiliary source integrals.
 in.pressure=101325;in.temperature=300;in.normal[1]=1;in.matchingDistance=.01;in.ownerDistance=.004;
 in.matching.temperature=500;in.matching.velocity[0]=10;in.matching.massFraction[0]=.7;in.matching.massFraction[1]=.3;
 in.model.viscosity=2e-5;in.model.conductivity=.04;in.model.diffusivity[0]=in.model.diffusivity[1]=1e-5;
 }
};
bool near(double a,double b,double r=1e-10){return std::abs(a-b)<1e-10+r*std::abs(b);}

#include <cstring>
template<class T> void same(T a,T b){assert(std::memcmp(&a,&b,sizeof(T))==0);}
template<class T,int N> void sameOutput(const WallOutput<T,N>& a,const WallOutput<T,N>& b){
 same(a.traceTemperature,b.traceTemperature);same(a.traceDensity,b.traceDensity);
 same(a.conductiveHeatFlux,b.conductiveHeatFlux);same(a.integratedKSource,b.integratedKSource);
 same(a.ownerOmega,b.ownerOmega);same(a.wallKFlux,b.wallKFlux);same(a.residual,b.residual);
 same(a.chemistryMismatchAvailable,b.chemistryMismatchAvailable);same(a.iterations,b.iterations);
 for(int d=0;d<3;++d){same(a.traceVelocity[d],b.traceVelocity[d]);same(a.traction[d],b.traction[d]);}
 for(int j=0;j<N;++j){same(a.traceMassFraction[j],b.traceMassFraction[j]);
  same(a.wallSpeciesFlux[j],b.wallSpeciesFlux[j]);same(a.matchingSpeciesFlux[j],b.matchingSpeciesFlux[j]);
  same(a.reactionIntegral[j],b.reactionIntegral[j]);same(a.chemistryMismatch[j],b.chemistryMismatch[j]);
  same(a.speciesBalanceResidual[j],b.speciesBalanceResidual[j]);}
}

template<class R> void physicalCast(const WallOutput<R,2>& out,const WallOutput<double,2>& exact){
#define FIELD(name) assert(out.name==R(exact.name)&&std::isfinite(out.name))
 FIELD(traceTemperature);FIELD(traceDensity);FIELD(conductiveHeatFlux);FIELD(ownerOmega);FIELD(wallKFlux);
#undef FIELD
 for(int d=0;d<3;++d){assert(out.traceVelocity[d]==R(exact.traceVelocity[d])&&std::isfinite(out.traceVelocity[d]));
  assert(out.traction[d]==R(exact.traction[d])&&std::isfinite(out.traction[d]));}
 for(int j=0;j<2;++j){
#define ARRAY(name) assert(out.name[j]==R(exact.name[j])&&std::isfinite(out.name[j]))
  ARRAY(traceMassFraction);ARRAY(wallSpeciesFlux);ARRAY(matchingSpeciesFlux);ARRAY(reactionIntegral);
#undef ARRAY
 }
 assert(out.iterations==exact.iterations);
}
int main(){

 Model m;ugkp::diagnostics::DeviceContext diagnostics{};m.in.diagnostics=&diagnostics;m.in.temperature=m.in.matching.temperature=500;m.in.matching.velocity[0]=300;
 m.in.matching.k=.5;m.in.matching.omega=200;m.in.integrateOwnerKSource=false;
 WallModelConfig<double> c;c.enableSst=true;c.nodes=24;c.maxIterations=100;
 WallWorkspace<double,2,24> w;WallOutput<double,2> plain,out;WallStatus status;
 assert(evaluateWallModel(m.in,c,w,plain,status));
 auto in=detail::wallComputationInput(m.in);auto controls=detail::wallComputationConfig(c);
 detail::LayerContext<double,2> ctx;detail::makeContext(in,controls,ctx);
 int index=1;double largest=0;
 for(int i=1;i<c.nodes;++i){detail::LayerPoint<double,2> point;assert(detail::evaluatePoint(in,controls,ctx,w.state,w.y,i,point));
  if(std::abs(point.sourceK)>std::abs(largest)){index=i;largest=point.sourceK;}}
 assert(std::abs(largest)>1);
 double distance=double(w.y[index]),weight=std::numeric_limits<double>::max();
 m.in.quadrature={&distance,&weight,1,double(1),double(1)};m.in.integrateOwnerKSource=true;
 std::memset(static_cast<void*>(&out),0xA5,sizeof(out));assert(evaluateWallModel(m.in,c,w,out,status));
 assert(status.code==WallCode::Success&&!std::isfinite(out.integratedKSource));
 assert(!ugkp::diagnostics::hasFatal(&diagnostics)&&diagnostics.warnings[1].count>0);
 // Optional source work changes no required output or residual trajectory.
 out.integratedKSource=plain.integratedKSource;sameOutput(out,plain);
 in=detail::wallComputationInput(m.in);WallOutput<double,2> exact;
 assert(detail::publishLayer(in,controls,ctx,w,status.iteration,status.residual,exact));
 WallOutput<double,2> converted;assert(detail::publishWallPrecision(m.in,exact,converted));
 assert(!std::isfinite(converted.integratedKSource));physicalCast(converted,exact);
 assert(converted.integratedKSource==double(exact.integratedKSource)
  ||(std::isnan(converted.integratedKSource)&&std::isnan(exact.integratedKSource)));


 // A singular consumed owner omega reports itself, not the finite heat flux.
 diagnostics={};in.ownerDistance=0;in.integrateOwnerKSource=false;
 assert(!detail::publishLayer(in,controls,ctx,w,0,0.,exact));
 assert(diagnostics.firstFatal.sourceSiteId==31301);
 assert(diagnostics.firstFatal.component==-1&&std::isinf(diagnostics.firstFatal.observed)&&diagnostics.firstFatal.observed>0);
 // Invalid physical state still returns failure and records the consumed value.
 diagnostics={};w.state[0][2]=std::numeric_limits<double>::quiet_NaN();
 assert(!detail::publishLayer(in,controls,ctx,w,0,0.,exact));
 assert(ugkp::diagnostics::hasFatal(&diagnostics)&&diagnostics.firstFatal.sourceSiteId==2410);
}


''',
    'species_audit': r'''


#include "common/gasWall/WallModel.H"
#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>
using namespace ugkwp;
using namespace ugkwp::gaswall;
struct Model {
 SpeciesThermoData<double> sp[2]; double coef[6]={1000,0,0,1000,0,0};
 double elements[2]={1,1}, basis[2]={-1,1};
 GasReactionData<double> rx[1]; GasStoichTerm<double> re[1],pr[1];
 WallInput<double,2> in;
 Model(){for(int s=0;s<2;++s){sp[s].coefficientOffset=3*s;sp[s].molarMass=.028;sp[s].minTemperature=200;sp[s].maxTemperature=4000;sp[s].referencePressure=101325;sp[s].referenceTemperature=300;sp[s].referenceEntropy=1500;sp[s].hasEntropyReference=true;}
 auto& t=in.model.thermo;t.species=sp;t.coefficients=coef;t.coefficientCount=6;t.elementCount=1;t.elementComposition=elements;t.speciesOrderHash=1;
 rx[0].reactantCount=1;rx[0].productCount=1;rx[0].highRate.preExponential=2;re[0].species=0;re[0].coefficient=1;pr[0].species=1;pr[0].coefficient=1;
 auto& m=in.model.mechanism;m.reactions=rx;m.reactionCount=1;m.reactants=re;m.reactantTermCount=1;m.products=pr;m.productTermCount=1;m.referencePressure=101325;m.speciesOrderHash=1;m.stoichiometricBasis=basis;m.independentRank=1;
 in.integrateOwnerKSource=true; // This fixture explicitly tests auxiliary source integrals.
 in.pressure=101325;in.temperature=300;in.normal[1]=1;in.matchingDistance=.01;in.ownerDistance=.004;
 in.matching.temperature=500;in.matching.velocity[0]=10;in.matching.massFraction[0]=.7;in.matching.massFraction[1]=.3;
 in.model.viscosity=2e-5;in.model.conductivity=.04;in.model.diffusivity[0]=in.model.diffusivity[1]=1e-5;
 }
};
bool near(double a,double b,double r=1e-10){return std::abs(a-b)<1e-10+r*std::abs(b);}


int main(){
 Model m; ugkp::diagnostics::DeviceContext diagnostics{};m.in.diagnostics=&diagnostics;
 m.in.integrateOwnerKSource=false;m.in.temperature=m.in.matching.temperature=500;m.in.matching.velocity[0]=0;
 m.in.matchingDistance=1;m.in.ownerDistance=.5;m.in.massFlux[0]=-1.6e308;m.in.massFlux[1]=1.6e308;
 m.in.model.diffusivity[0]=m.in.model.diffusivity[1]=1e308;
 m.coef[2]=m.coef[5]=-500000;
 WallModelConfig<double> c;c.nodes=3;c.maxIterations=0;WallWorkspace<double,2,3> w;WallOutput<double,2> out;
 detail::LayerContext<double,2> ctx;assert(detail::makeContext(m.in,c,ctx));
 for(int i=0;i<3;++i){w.y[i]=.5*i;w.state[i][3]=.2+.3*i;}
 const bool ok=detail::publishLayer(m.in,c,ctx,w,0,0.,out);

 assert(ok);assert(std::isinf(out.speciesBalanceResidual[0]));
 assert(!ugkp::diagnostics::hasFatal(&diagnostics)&&diagnostics.warnings[0].count>0);
 for(int j=0;j<2;++j){assert(std::isfinite(out.wallSpeciesFlux[j])&&std::isfinite(out.matchingSpeciesFlux[j])&&std::isfinite(out.reactionIntegral[j]));}
 assert(std::isfinite(out.conductiveHeatFlux)&&std::isfinite(out.traceDensity));
 WallOutput<double,2> converted;assert(detail::publishWallPrecision(m.in,out,converted));
 assert(converted.speciesBalanceResidual[0]==out.speciesBalanceResidual[0]);
 // Finite nodal sources whose required integrated flux overflows still fail.
 diagnostics={};m.in.massFlux[0]=m.in.massFlux[1]=0;
 m.in.model.diffusivity[0]=m.in.model.diffusivity[1]=1e-5;
 m.in.model.mode=GasMode::MixtureChemistry;m.rx[0].highRate.preExponential=1e307;
 m.in.matchingDistance=100;m.in.ownerDistance=50;ctx={};
 assert(detail::makeContext(m.in,c,ctx));
 for(int i=0;i<3;++i){w.y[i]=50*i;w.state[i][3]=.5;}
 assert(!detail::publishLayer(m.in,c,ctx,w,0,0.,out));
 assert(ugkp::diagnostics::hasFatal(&diagnostics)&&diagnostics.firstFatal.sourceSiteId==31303);
 assert(diagnostics.firstFatal.component==0&&std::isinf(diagnostics.firstFatal.observed)&&std::signbit(diagnostics.firstFatal.observed));
}


''',
    'failure_channel': r'''


#include "common/gasWall/WallModel.H"
#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>
using namespace ugkwp;
using namespace ugkwp::gaswall;
struct Model {
 SpeciesThermoData<double> sp[2]; double coef[6]={1000,0,0,1000,0,0};
 double elements[2]={1,1}, basis[2]={-1,1};
 GasReactionData<double> rx[1]; GasStoichTerm<double> re[1],pr[1];
 WallInput<double,2> in;
 Model(){for(int s=0;s<2;++s){sp[s].coefficientOffset=3*s;sp[s].molarMass=.028;sp[s].minTemperature=200;sp[s].maxTemperature=4000;sp[s].referencePressure=101325;sp[s].referenceTemperature=300;sp[s].referenceEntropy=1500;sp[s].hasEntropyReference=true;}
 auto& t=in.model.thermo;t.species=sp;t.coefficients=coef;t.coefficientCount=6;t.elementCount=1;t.elementComposition=elements;t.speciesOrderHash=1;
 rx[0].reactantCount=1;rx[0].productCount=1;rx[0].highRate.preExponential=2;re[0].species=0;re[0].coefficient=1;pr[0].species=1;pr[0].coefficient=1;
 auto& m=in.model.mechanism;m.reactions=rx;m.reactionCount=1;m.reactants=re;m.reactantTermCount=1;m.products=pr;m.productTermCount=1;m.referencePressure=101325;m.speciesOrderHash=1;m.stoichiometricBasis=basis;m.independentRank=1;
 in.integrateOwnerKSource=true; // This fixture explicitly tests auxiliary source integrals.
 in.pressure=101325;in.temperature=300;in.normal[1]=1;in.matchingDistance=.01;in.ownerDistance=.004;
 in.matching.temperature=500;in.matching.velocity[0]=10;in.matching.massFraction[0]=.7;in.matching.massFraction[1]=.3;
 in.model.viscosity=2e-5;in.model.conductivity=.04;in.model.diffusivity[0]=in.model.diffusivity[1]=1e-5;
 }
};
bool near(double a,double b,double r=1e-10){return std::abs(a-b)<1e-10+r*std::abs(b);}


int main(){
 Model m;m.in.integrateOwnerKSource=false;
 WallModelConfig<double> c;c.model=BoundaryLayerModel::ConstantTransport;
 WallWorkspace<double,2,24> w;WallOutput<double,2> out;WallStatus status;
 assert(evaluateWallModel(m.in,c,w,out,status));
 m.in.temperature=-1;
 assert(!evaluateWallModel(m.in,c,w,out,status));
 ugkp::diagnostics::DeviceContext diagnostics{};m.in.diagnostics=&diagnostics;
 assert(!evaluateWallModel(m.in,c,w,out,status));
 assert(ugkp::diagnostics::hasFatal(&diagnostics));
 assert(diagnostics.firstFatal.code==ugkp::diagnostics::Code::NegativeTemperature);
 assert(diagnostics.firstFatal.observed==-1);
 const auto first=diagnostics.firstFatal;
 assert(!detail::wallFatal(m.in,ugkp::diagnostics::Code::NonFinite,99999));
 assert(diagnostics.firstFatal.sourceSiteId==first.sourceSiteId);
 diagnostics={};m.in.temperature=300;m.in.model.thermo.coefficients=nullptr;
 assert(!evaluateWallModel(m.in,c,w,out,status));
 assert(ugkp::diagnostics::hasFatal(&diagnostics)&&diagnostics.firstFatal.code==ugkp::diagnostics::Code::MissingOperator);
}


''',
    'opt_in': r'''


#include "common/gasWall/WallModel.H"
#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>
using namespace ugkwp;
using namespace ugkwp::gaswall;
struct Model {
 SpeciesThermoData<GpuReal> sp[2]; GpuReal coef[6]={1000,0,0,1000,0,0};
 GpuReal elements[2]={1,1}, basis[2]={-1,1};
 GasReactionData<GpuReal> rx[1]; GasStoichTerm<GpuReal> re[1],pr[1];
 WallInput<GpuReal,2> in;
 Model(){for(int s=0;s<2;++s){sp[s].coefficientOffset=3*s;sp[s].molarMass=.028;sp[s].minTemperature=200;sp[s].maxTemperature=4000;sp[s].referencePressure=101325;sp[s].referenceTemperature=300;sp[s].referenceEntropy=1500;sp[s].hasEntropyReference=true;}
 auto& t=in.model.thermo;t.species=sp;t.coefficients=coef;t.coefficientCount=6;t.elementCount=1;t.elementComposition=elements;t.speciesOrderHash=1;
 rx[0].reactantCount=1;rx[0].productCount=1;rx[0].highRate.preExponential=2;re[0].species=0;re[0].coefficient=1;pr[0].species=1;pr[0].coefficient=1;
 auto& m=in.model.mechanism;m.reactions=rx;m.reactionCount=1;m.reactants=re;m.reactantTermCount=1;m.products=pr;m.productTermCount=1;m.referencePressure=101325;m.speciesOrderHash=1;m.stoichiometricBasis=basis;m.independentRank=1;
 in.integrateOwnerKSource=true; // This fixture explicitly tests auxiliary source integrals.
 in.pressure=101325;in.temperature=300;in.normal[1]=1;in.matchingDistance=.01;in.ownerDistance=.004;
 in.matching.temperature=500;in.matching.velocity[0]=10;in.matching.massFraction[0]=.7;in.matching.massFraction[1]=.3;
 in.model.viscosity=2e-5;in.model.conductivity=.04;in.model.diffusivity[0]=in.model.diffusivity[1]=1e-5;
 }
};
bool near(GpuReal a,GpuReal b,GpuReal r=1e-10){return std::abs(a-b)<1e-10+r*std::abs(b);}

#include <cstring>
int main(){Model m;m.in.temperature=m.in.matching.temperature=500;m.in.matching.velocity[0]=30;m.in.matching.k=.5;m.in.matching.omega=200;
 GpuReal y[2]={.002,.006},weight[2]={2e-7,2e-7};m.in.quadrature={y,weight,2,4e-7,1.6e-9};
 WallModelConfig<GpuReal> c;c.enableSst=true;c.nodes=32;c.maxIterations=100;
 WallWorkspace<GpuReal,2,48> w;WallOutput<GpuReal,2> with,without;WallStatus s;
 m.in.integrateOwnerKSource=true;assert(evaluateWallModel(m.in,c,w,with,s));assert(std::isfinite(with.integratedKSource));
 // No valid quadrature pointers: this path cannot touch even one point.
 m.in.integrateOwnerKSource=false;m.in.quadrature.distance=nullptr;m.in.quadrature.volumeWeight=nullptr;
 assert(evaluateWallModel(m.in,c,w,without,s));assert(std::isnan(without.integratedKSource));
 assert(with.traceTemperature==without.traceTemperature&&with.traceDensity==without.traceDensity&&with.ownerOmega==without.ownerOmega&&with.wallKFlux==without.wallKFlux&&with.conductiveHeatFlux==without.conductiveHeatFlux&&with.residual==without.residual&&with.iterations==without.iterations);
 for(int d=0;d<3;++d)assert(with.traction[d]==without.traction[d]&&with.traceVelocity[d]==without.traceVelocity[d]);
 for(int j=0;j<2;++j)assert(with.traceMassFraction[j]==without.traceMassFraction[j]&&with.wallSpeciesFlux[j]==without.wallSpeciesFlux[j]&&with.matchingSpeciesFlux[j]==without.matchingSpeciesFlux[j]&&with.reactionIntegral[j]==without.reactionIntegral[j]);
 m.in.integrateOwnerKSource=true;assert(!evaluateWallModel(m.in,c,w,without,s));
 ugkp::diagnostics::DeviceContext diagnostics{};m.in.diagnostics=&diagnostics;
 assert(!evaluateWallModel(m.in,c,w,without,s));
 assert(ugkp::diagnostics::hasFatal(&diagnostics)&&diagnostics.firstFatal.sourceSiteId==10017);
 const auto first=diagnostics.firstFatal;
 assert(!detail::wallFatal(m.in,ugkp::diagnostics::Code::NonFinite,99999));
 assert(diagnostics.firstFatal.sourceSiteId==first.sourceSiteId);
 WallInput<GpuReal,2> defaults;assert(!defaults.integrateOwnerKSource);
}


''',
    'publication_provenance': r'''

#include "common/gasWall/WallModel.H"
#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>
using namespace ugkwp;
using namespace ugkwp::gaswall;
struct Model {
 SpeciesThermoData<double> sp[2]; double coef[6]={1000,0,0,1000,0,0};
 double elements[2]={1,1}, basis[2]={-1,1};
 GasReactionData<double> rx[1]; GasStoichTerm<double> re[1],pr[1];
 WallInput<double,2> in;
 Model(){for(int s=0;s<2;++s){sp[s].coefficientOffset=3*s;sp[s].molarMass=.028;sp[s].minTemperature=200;sp[s].maxTemperature=4000;sp[s].referencePressure=101325;sp[s].referenceTemperature=300;sp[s].referenceEntropy=1500;sp[s].hasEntropyReference=true;}
 auto& t=in.model.thermo;t.species=sp;t.coefficients=coef;t.coefficientCount=6;t.elementCount=1;t.elementComposition=elements;t.speciesOrderHash=1;
 rx[0].reactantCount=1;rx[0].productCount=1;rx[0].highRate.preExponential=2;re[0].species=0;re[0].coefficient=1;pr[0].species=1;pr[0].coefficient=1;
 auto& m=in.model.mechanism;m.reactions=rx;m.reactionCount=1;m.reactants=re;m.reactantTermCount=1;m.products=pr;m.productTermCount=1;m.referencePressure=101325;m.speciesOrderHash=1;m.stoichiometricBasis=basis;m.independentRank=1;
 in.integrateOwnerKSource=true; // This fixture explicitly tests auxiliary source integrals.
 in.pressure=101325;in.temperature=300;in.normal[1]=1;in.matchingDistance=.01;in.ownerDistance=.004;
 in.matching.temperature=500;in.matching.velocity[0]=10;in.matching.massFraction[0]=.7;in.matching.massFraction[1]=.3;
 in.model.viscosity=2e-5;in.model.conductivity=.04;in.model.diffusivity[0]=in.model.diffusivity[1]=1e-5;
 }
};
bool near(double a,double b,double r=1e-10){return std::abs(a-b)<1e-10+r*std::abs(b);}


int main(){
 Model m;ugkp::diagnostics::DeviceContext diagnostics{};m.in.diagnostics=&diagnostics;
 m.in.integrateOwnerKSource=false;m.in.temperature=m.in.matching.temperature=500;m.in.matching.velocity[0]=0;
 m.coef[2]=m.coef[5]=-500000;
 m.in.matchingDistance=1000;m.in.ownerDistance=500;
 m.in.model.mode=GasMode::MixtureChemistry;m.rx[0].highRate.preExponential=1e307;
 WallModelConfig<double> c;c.nodes=3;WallWorkspace<double,2,3> w;WallOutput<double,2> out;
 detail::LayerContext<double,2> ctx;assert(detail::makeContext(m.in,c,ctx));
 for(int i=0;i<3;++i){w.y[i]=500*i;w.state[i][3]=.5;}
 assert(!detail::publishLayer(m.in,c,ctx,w,0,0.,out));
 assert(diagnostics.firstFatal.sourceSiteId==31302&&diagnostics.firstFatal.component==0);
 assert(std::isinf(diagnostics.firstFatal.observed)&&std::signbit(diagnostics.firstFatal.observed));
 // Required heat overflow from finite species enthalpies/mass fluxes.
 diagnostics={};m.in.model.mode=GasMode::MixtureFrozen;
 m.coef[2]=-499999;m.coef[5]=-500001;
 m.in.massFlux[0]=1e308;m.in.massFlux[1]=-1e308;ctx={};
 assert(detail::makeContext(m.in,c,ctx));
 assert(!detail::publishLayer(m.in,c,ctx,w,0,0.,out));
 assert(diagnostics.firstFatal.sourceSiteId==2426&&diagnostics.firstFatal.component==-1);
 assert(std::isinf(diagnostics.firstFatal.observed)&&std::signbit(diagnostics.firstFatal.observed));
}

''',
}
@pytest.mark.parametrize("name", PROBES)
def test_optional_wall_channels(tmp_path, name):
    source = tmp_path / f"{name}.cpp"
    source.write_text(PROBES[name])
    binary = tmp_path / name
    command = shlex.split(os.environ.get("CXX", "g++")) + [
        "-std=c++17", "-O1", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
        "-I" + str(ROOT), "-I" + str(ROOT / "common"), str(source), "-o", str(binary),
    ]
    build = subprocess.run(command, capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
