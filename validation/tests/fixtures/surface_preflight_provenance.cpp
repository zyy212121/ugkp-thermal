#include "surface_terminal/fixture.hpp"
#include <fenv.h>
#include <iomanip>
#include <limits>
using namespace ugkp::diagnostics;
int main(int argc,char** argv){if(argc!=2)return 2;feenableexcept(FE_OVERFLOW|FE_INVALID|FE_DIVBYZERO);const int mask=fegetexcept();const std::string mode=argv[1];auto p=physics();auto h=host(p);auto in=input(h,p);in.hasSolid=false;in.hasFilm=true;in.film.mass=1;in.film.species[0]=1;in.film.enthalpy=liquidH(p,800,in.bulk.pressure);in.filmBase=in.film;in.aux.pressure=in.bulk.pressure;
DeviceContext ctx{};reset(ctx,12,13,17,.12);p.diagnostics=&ctx;p.tolerances.diagnostics=&ctx;const int species=1;const Real inf=std::numeric_limits<Real>::infinity();
if(mode=="R-inf"||mode=="R-inf-primitive")p.species[species].R=inf;
else if(mode=="R-negative-primitive")p.species[species].R=-17;
else if(mode=="model-invalid")p.species[species].model=static_cast<ugkwp::SpeciesThermoModel>(999);
else if(mode=="D-inf"||mode=="inactive"||mode=="boundary-preflight")p.gasDiffusivity[species]=inf;
else if(mode!="valid")return 3;
if(mode=="inactive"){in.hasFilm=false;SurfacePhysicsResult out;assert(!solveSurfaceInterface(in,p,out)&&!hasFatal(&ctx));return 0;}
if(mode=="boundary-preflight"){p.wallModel.family=ugkwp::gaswall::WallFamily::BoundaryLayer;p.wallModel.model=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport;assert(surfaceTemperatureConfiguration(in,p)&&!hasFatal(&ctx));return 0;}
SurfacePhysicsResult out;out.topTemperature=-123;out.gasSpecies[0]=456;GasPrimitive primitive;primitive.temperature=-123;const bool ok=(mode=="R-inf-primitive"||mode=="R-negative-primitive")?makeInterfacePrimitive(in.bulk.Y,800,in.bulk.pressure,{},in.normal,0,0,p,primitive):solveSurfaceInterface(in,p,out);const auto& error=ctx.firstFatal;std::cout<<std::setprecision(17)<<mode<<" ok="<<ok<<" fatal="<<hasFatal(&ctx)<<" site="<<error.sourceSiteId<<" observed="<<error.observed<<" reference="<<error.reference<<" component="<<error.component<<std::endl;
if(mode=="valid")assert(ok&&!hasFatal(&ctx));
else{assert(!ok&&hasFatal(&ctx)&&out.topTemperature==-123&&out.gasSpecies[0]==456);assert(error.component==species);if(mode=="R-inf"||mode=="R-inf-primitive"){assert(error.sourceSiteId==3843&&error.code==Code::NonFinite&&error.observed==inf&&error.reference==inf);if(mode=="R-inf-primitive")assert(primitive.temperature==-123);}
else if(mode=="R-negative-primitive")assert(error.sourceSiteId==3843&&error.observed==-17&&error.reference==-17);else if(mode=="D-inf")assert(error.sourceSiteId==3842&&error.code==Code::NonFinite&&error.observed==inf);else assert(error.sourceSiteId==3843&&error.code==Code::MissingOperator&&error.observed==static_cast<Real>(p.species[species].model)&&error.reference==0);}
assert(fegetexcept()==mask&&fetestexcept(FE_OVERFLOW|FE_INVALID|FE_DIVBYZERO)==0);return 0;}
