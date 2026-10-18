"""Compile and exercise finite extrapolation and diagnostic-only geometry bounds."""
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def test_material_domain_and_film_geometry(tmp_path):
    source=tmp_path/'probe.cpp'
    source.write_text(r'''
#include "film/CpuFilmDriver.C"
#include "materials/Reaction.H"
#include "gpu/SingleGasContract.H"
#include "materials/MaterialCaloric.H"
#include <cassert>
using namespace chmt;
int main(){
 ugkp::diagnostics::DeviceContext ctx{};PhysicsConfig p;p.diagnostics=&ctx;p.tolerances.diagnostics=&ctx;
 p.liquid.rho=1000;p.liquid.cp0=1000;p.liquid.Tmin=100;p.liquid.Tmax=1000;p.liquidViscosity=1;p.enableFilm=true;
 FilmQ q;q.mass=1;q.species[0]=1;q.enthalpy=1500000;FilmAux a;
 assert(recoverFilm(q,1,0,p,a)&&a.temperature==1500);
 auto curved=p;curved.liquid.cp1=-1;curved.liquid.Tmax=2000;
 FilmQ physical=q;physical.enthalpy=255000;FilmAux physicalAux;
 assert(recoverFilm(physical,1,0,curved,physicalAux)&&physicalAux.temperature==300);
 q.enthalpy=-1000;assert(!recoverFilm(q,1,0,p,a));assert(ugkp::diagnostics::hasFatal(&ctx));ctx={};
 p.condensed[0]=p.liquid;p.material.minimumPorosity=.2;p.material.maximumPorosity=.3;
 SolidQ s;s.condensed[0]=500;s.energy=150000000;MaterialPrimitive m;
 assert(recoverMaterial(s,1,p,m)&&m.porosity==.5);
 HostState h;h.surface.area={1,1};h.surface.normal={{0,0,1.00001},{0,0,1}};h.film.resize(2);h.filmAux.resize(2);
 h.surface.edgeOwner={0};h.surface.edgeNeighbour={1};h.surface.edgeLength={1};h.surface.edgeConormal={{1,0,.00001}};
 h.surface.edgeOwnerOffset={{.5,0,0}};h.surface.edgeNeighbourOffset={{-.5,0,0}};
 std::vector<CpuFilmForcing> f(2);std::vector<CpuFilmDrive>d(2);std::string error;for(int i=0;i<2;++i){d[i].pressure=0;h.filmAux[i].pressure=0;}
 assert(validate(h,p,.01,f,d,error,nullptr));
 assert(!ugkp::diagnostics::hasFatal(&ctx));
 for(auto& species:p.species){species.R=287;species.cp0=1000;species.Tmin=100;species.Tmax=1000;}
 GasPrimitive gas;gas.rho=1;gas.temperature=1500;gas.Y[0]=1;
 const auto conservative=conservativeGas(gas,1,p);GasPrimitive recovered;
 assert(recoverGas(conservative,1,p,recovered)&&recovered.temperature==1500);
 p.material.minimumPorosity=2;p.material.maximumPorosity=-2;assert(validateMaterialPolicy(p,error));
 p.material.enableMelting=true;p.material.phaseCondensed=0;p.material.phaseFilmY[0]=1.1;
 assert(validateMaterialPolicy(p,error)&&ctx.warnings[2].count>0);
 assert(physicalFraction(p,1.1)==1&&physicalFraction(p,-.1)==0);
 CondensedThermo shifted=p.liquid;shifted.e0=1e30;assert(validCondensed(shifted));
 Real scaledT;
 assert(invertCaloric(0,1e155,1,3e157,100,1000,p.tolerances,scaledT)&&std::abs(scaledT-300)<1e-10);
 assert(invertCaloric(0,1e-167,1e-170,3.45e-165,100,1000,p.tolerances,scaledT)&&std::abs(scaledT-300)<1e-10);
 assert(invertCaloric(0,-1e-200,1e-300,1e100,100,1000,p.tolerances,scaledT)&&std::isfinite(scaledT)&&scaledT>1e200);
 Real shiftedT=300;const auto count=ctx.warnings[1].count;
 assert(invertCaloric(1e30,1000,0,1e30,100,1000,p.tolerances,shiftedT));assert(ctx.warnings[1].count>count);
 SpeciesThermo reference=p.species[0];reference.cp0=100;reference.cp1=1;reference.Tmin=200;reference.Tmax=500;
 assert(validSpecies(reference));
 MaterialCaloric caloric;p.condensed[0]=shifted;assert(materialCaloric(s,p,caloric,error));

 ModelConfig model;model.physics.singleGasSpecies=0;HostState state;state.gas.resize(1);state.gas[0].species[1]=.1;
 assert(!validateSingleGasInventory(model,state,error));assert(error.find("source=gas species=1 entity=0")!=std::string::npos);
}
''')
    exe=tmp_path/'probe'
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-fsanitize=undefined','-fno-sanitize-recover=undefined','-I'+str(ROOT/'applications/CHMT'),'-I'+str(ROOT/'common'),str(source),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)

def test_surface_retains_finite_unbracketed_state(tmp_path):
    source=tmp_path/'surface.cpp'
    source.write_text(r'''
#include "fixture.hpp"
int main(){
 auto p=physics();ugkp::diagnostics::DeviceContext diagnostics{};
 p.diagnostics=&diagnostics;p.tolerances.diagnostics=&diagnostics;
 auto h=host(p);auto in=input(h,p);MaterialPrimitive material;
 assert(recoverMaterial(in.solid,in.solidVolume,p,material));
 SurfacePhysicsResult result;material.temperature=100.25;SurfaceRootTrace trace;
 assert(findFeasibleSurfaceRoot<true>(in,p,material,p.meltTemperature,100,101,result,&trace));
 assert(result.topTemperature!=material.temperature&&result.topTemperature==trace.completedState.topTemperature);
 assert(result.topTemperature==101&&std::isfinite(result.thermalResidual));
 assert(diagnostics.warnings[1].count>0&&!diagnostics.fatalClaim);
 in.normal={1.000001,0,0};in.gasNormal={1.000001,0,0};in.solidNormal={1.000001,0,0};
 assert(solveSurfaceInterface(in,p,result));assert(std::isfinite(result.topTemperature));
 assert(!diagnostics.fatalClaim);
 Real Y[Ns]{},J[Ns]{},residual[Ns]{};Y[0]=1;
 assert(!compositionResidual(Y,Y,J,std::numeric_limits<Real>::quiet_NaN(),300,1,p,residual,nullptr));
 assert(diagnostics.firstFatal.code==ugkp::diagnostics::Code::NonFinite&&diagnostics.firstFatal.component==1);
 diagnostics={};
 assert(!compositionResidual(Y,Y,J,101325,300,0,p,residual,nullptr));
 assert(diagnostics.firstFatal.observed==0&&diagnostics.firstFatal.component==3);
}
''')
    exe=tmp_path/'surface'
    subprocess.run(['g++','-std=c++17','-O1','-Wall','-Wextra','-Wno-misleading-indentation','-fsanitize=undefined','-fno-sanitize-recover=undefined',
      '-I'+str(ROOT/'validation/tests/fixtures/surface_terminal'),'-I'+str(ROOT/'applications/CHMT'),'-I'+str(ROOT/'common'),str(source),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
