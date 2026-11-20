"""Single-pass material numerics, independent of CUDA or OpenFOAM runtime."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "applications" / "CHMT"


def test_material_reaction_is_one_full_euler(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ unavailable")
    controller = (APP / "materials/ReactionStepControl.H").read_text()
    controller = controller.replace("advanceLocalReactions(", "countedEuler(")
    wrapper = r'''
#include "materials/Reaction.H"
namespace chmt {
int calls=0;
bool countedEuler(const SolidQ& q,Real v,const PhysicsConfig& p,Real dt,
    SolidQ& out,Real* rates,DeviceStatus* status,int index,bool* numerical) {
    ++calls;return advanceLocalReactions(q,v,p,dt,out,rates,status,index,numerical);
}
}
'''
    (tmp_path / "controller.H").write_text(wrapper + controller)
    probe = r'''
#include "controller.H"
#include <cassert>
#include <cmath>
using namespace chmt;
int main() {
    PhysicsConfig p;p.enableReactions=true;p.nReactions=1;
    ugkp::diagnostics::DeviceContext context{};p.diagnostics=&context;
    for(auto& c:p.condensed){c.rho=1000;c.cp0=1000;c.cp1=2;c.Tmin=100;c.Tmax=3000;}
    for(auto& s:p.species){s.R=300;s.cp0=1000;s.Tmin=100;s.Tmax=3000;}
    p.reactions[0].A=10;p.reactions[0].order[0]=1;
    p.reactions[0].condensedNu[0]=-1;p.reactions[0].condensedNu[1]=1;
    SolidQ q;q.condensed[0]=1;q.energy=caloricIntegral(0,1000,2,500);
    for(Real requested:{.001,.1,1.}) {
        ReactionStepControl control;assert(reactionStepControl(q,.001,p,.4,control));
        const Real expectedDt=minValue(requested,control.dtLimit);
        SolidQ expected,out;Real rates[Nr]{};
        assert(advanceLocalReactions(q,.001,p,expectedDt,expected,rates));
        Real dt=0;bool active=false;calls=0;
        assert(reactionStepCandidate(q,.001,p,requested,.4,out,dt,active));
        assert(active&&calls==1&&dt==expectedDt);
        assert(out.energy==expected.energy);
        for(int c=0;c<Nc;++c)assert(out.condensed[c]==expected.condensed[c]);
        for(int r=0;r<Nr;++r)assert(out.progress[r]==expected.progress[r]);
        q=out;
    }
    Real dt=17;bool active=true;SolidQ out=q;calls=0;
    assert(!reactionStepCandidate(q,.001,p,0,.4,out,dt,active));
    assert(calls==0&&dt==17&&out.energy==q.energy);
    assert(context.fatalClaim&&context.firstFatal.sourceSiteId==4117);
}
'''
    source = tmp_path / "probe.cpp"
    source.write_text(probe)
    binary = tmp_path / "probe"
    subprocess.run([compiler, "-std=c++17", "-O0", "-DUGKWP_GAS_SPECIES=2",
                    "-I" + str(APP), "-I" + str(ROOT / "common"),
                    str(source), "-o", str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


def test_material_and_window_sources_have_no_replay_or_terminal_route():
    reaction = (APP / "materials/ReactionStepControl.H").read_text()
    driver = (APP / "materials/CpuMaterialDriver.C").read_text()
    window = (APP / "io/MultirateEvolution.C").read_text()
    transport = (APP / "materials/MaterialTransport.C").read_text()
    assert "advanceLocalReactions" in reaction
    for token in ("retry", "refined", "half", "completeReactionStep"):
        assert token not in reaction
    for token in ("dt*=.5", "lastRetry", "terminalMaterial", "terminalLast", "detail::complete"):
        assert token not in driver
    for token in ("rollbackGasWindow", "controller.replay", "controller.shorten", "ConductionContinuation"):
        assert token not in window
    assert not (APP / "materials/ConductionContinuation.H").exists()
    assert "bound*=.5" not in transport
    assert "completeMaterial" not in transport
    assert "while(elapsed<windowEnd)" in driver
    assert "while(gasElapsed<intervalDuration(interval))" in window
    assert "for(int iteration=" in driver
    assert "advanceMaterialReactionsMath(" in transport
    reaction_substeps = (APP / "materials/MaterialReactionMath.H").read_text()
    assert "while(elapsed<interval)" in reaction_substeps
    assert window.index("bindImplicitGasWindowMaterial(") < window.index("enqueuePreparedGasWindow(")
    assert "collectMaterialWindowExchange" not in window
    implicit = (APP / "materials/ImplicitMaterialWindow.inl").read_text()
    assert ".residual()" not in implicit and "equation.solve(" not in implicit
    assert "material.advanceCandidate(" not in window


def test_implicit_reaction_uses_the_window_inventory_once(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ unavailable")
    probe = r'''
#include "materials/ImplicitMaterialMass.H"
#include <cassert>
#include <vector>
using namespace chmt;
int main(){
    HostState base;
    base.solid.resize(1);
    base.solid[0].condensed[0]=1;
    base.solidMesh.volumes={.001};
    base.solidMesh.cellCentres={{0,0,0}};
    base.solidMesh.cellFaceOffsets={0,0};
    base.surface.solidFace={0};
    base.surface.solidCell={0};

    PhysicsConfig physics;
    physics.enableReactions=true;
    physics.nReactions=1;
    for(auto& phase:physics.condensed){
        phase.rho=1000;phase.cp0=1000;phase.Tmin=100;phase.Tmax=3000;
    }
    auto& reaction=physics.reactions[0];
    reaction.A=12;reaction.order[0]=1;
    reaction.condensedNu[0]=-1;reaction.condensedNu[1]=1;

    MaterialWindowExchange input(base);
    ExchangePacket accepted;
    accepted.kind=ExchangeKind::GasSolid;
    accepted.solidCell=0;
    accepted.condensed[0]=PhysicalReal(.25);
    std::string error;
    assert(input.append(0,accepted,base.surface,error));

    ImplicitMaterialMass mass;
    assert(mass.begin(base,base.solidMesh,input,physics.tolerances,error));
    std::vector<PhysicalReal> temperature{500};
    std::vector<SolidQ> endpoint(1);
    std::vector<MaterialCaloric> caloric(1);
    std::vector<MaterialPrimitive> primitive(1);
    for(int i=0;i<4;++i){
        mass.reconstruct(physics);
        PhysicalReal residual=0;
        assert(mass.react(base.solidMesh,physics,temperature,.25,residual,error));
        assert(mass.prepare(temperature,base.solidMesh.volumes,physics,endpoint,caloric,primitive,error));
        const auto& state=endpoint[0];
        assert(std::abs(state.condensed[0]-.1875)<1e-7&&std::abs(state.condensed[1]-.5625)<1e-7);
        assert(std::abs(state.progress[0]-.5625)<1e-7);
        if(i>0)assert(residual<=1);
        temperature[0]+=25;
    }
    assert(base.solid[0].condensed[0]==1&&base.solid[0].progress[0]==0);
}
'''
    source = tmp_path / "implicit-window.cpp"
    source.write_text(probe)
    binary = tmp_path / "implicit-window"
    subprocess.run([compiler, "-std=c++17", "-O0", "-DUGKWP_GAS_SPECIES=2",
                    "-I" + str(APP), "-I" + str(ROOT / "common"), str(source),
                    str(APP / "materials/MaterialTransport.C"), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_implicit_material_face_transfers_close_mass_and_energy_together(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ unavailable")
    probe = r'''
#include "materials/ImplicitMaterialMass.H"
#include <cassert>
#include <cmath>
#include <type_traits>

using namespace chmt;

bool near(PhysicalReal x,PhysicalReal y){
    const auto scale=std::max(PhysicalReal(1),std::abs(y));
    return std::abs(x-y)<std::max(PhysicalReal(2e-10),PhysicalReal(32)*std::numeric_limits<PhysicalReal>::epsilon())*scale;
}

template<class Number>
void checkWindowSum(){
    MaterialWindowSum<Number> sum;
    const Number large=Number(16777216);
    sum.add(large);
    sum.add(Number(1));
    sum.add(-large);
    static_assert(std::is_same_v<decltype(sum.value()),Number>);
    if constexpr(std::is_same_v<Number,float>){
        static_assert(sizeof(sum)==sizeof(float));
        assert(sum.value()==0);
    }else assert(sum.value()==1);
}

void solve(bool darcy,bool diffusion,bool reverse,PhysicalReal tableMin=10){
    PhysicsConfig p;
    p.spatialOrder=1;
    p.gasConductivity=0;
    p.permeability=darcy?PhysicalReal(1)/PhysicalReal(300):PhysicalReal(0);
    p.poreViscosity=1;
    for(auto& phase:p.condensed){phase.rho=2;phase.cp0=1;phase.Tmin=tableMin;phase.Tmax=6000;}
    for(auto& gas:p.species){gas.R=1;gas.cp0=2;gas.Tmin=tableMin;gas.Tmax=6000;}
    if(diffusion){
        p.poreEffectiveDiffusivity[0]=p.poreEffectiveDiffusivity[1]=darcy?PhysicalReal(.1):PhysicalReal(.5);
        p.species[1].cp0=3;
    }

    HostState initial;
    auto& mesh=initial.solidMesh;
    mesh.volumes={1,1};
    mesh.cellCentres={{.5,.5,.5},{1.5,.5,.5}};
    mesh.faceCentres={{1,.5,.5}};
    mesh.areaVectors={{1,0,0}};
    mesh.owner={0};mesh.neighbour={1};mesh.periodicPartner={-1};
    mesh.boundaryKind={BoundaryKind::Internal};
    mesh.cellFaceOffsets={0,1,2};mesh.cellFaces={0,0};mesh.cellFaceSigns={1,-1};
    initial.solid.resize(2);
    const int donor=reverse?1:0,receiver=1-donor;
    initial.solid[donor].pore[0]=1;
    initial.solid[receiver].pore[diffusion?1:0]=darcy?PhysicalReal(.1):PhysicalReal(1);
    for(auto& cell:initial.solid){
        cell.condensed[0]=1;
        cell.energy=EnergyReal(300)+(EnergyReal(cell.pore[0])+EnergyReal(2)*EnergyReal(cell.pore[1]))*EnergyReal(300);
    }
    const std::vector<EnergyReal> fixed{initial.solid[0].energy,initial.solid[1].energy};
    const std::vector<EnergyReal> noConduction(2);
    MaterialWindowExchange accepted(initial);
    ImplicitMaterialMass balance;
    std::string error;
    assert(balance.begin(initial,mesh,accepted,p.tolerances,error));
    std::vector<PhysicalReal> temperature(2,300);
    std::vector<SolidQ> state(2);
    std::vector<MaterialCaloric> caloric(2);
    std::vector<MaterialPrimitive> primitive(2);
    bool converged=false;
    for(int i=0;i<50;++i){
        balance.reconstruct(p);
        assert(balance.prepare(temperature,mesh.volumes,p,state,caloric,primitive,error));
        MaterialTransportResult rates;
        assert(evaluateMaterialTransport(mesh,state,p,{},1,PhysicalReal(.4),rates,error,nullptr,&primitive));
        PhysicalReal massResidual=0;
        assert(balance.transport(mesh,{},p,primitive,rates,PhysicalReal(1),massResidual,error));
        assert(balance.prepare(temperature,mesh.volumes,p,state,caloric,primitive,error));
        auto target=fixed;
        EnergyReal boundary=0;
        assert(balance.correctEnergy(p,temperature,caloric,noConduction,target,boundary,error));
        assert(boundary==0);
        for(int c=0;c<2;++c){
            state[c].energy=target[c];
            assert(recoverSolid(state[c],p,temperature[c]));
            assert(temperature[c]>0);
        }
        EnergyReal energyResidual=0;
        assert(balance.checkEnergy(p,temperature,state,EnergyReal(1e-10),EnergyReal(1e-10),energyResidual,error));
        assert(near(target[0]+target[1],fixed[0]+fixed[1]));
        for(int species=0;species<Ns;++species){
            assert(near(state[0].pore[species]+state[1].pore[species],
                initial.solid[0].pore[species]+initial.solid[1].pore[species]));
        }
        if(i==0&&darcy&&!diffusion){
            assert(near(state[donor].pore[0],PhysicalReal(5)/PhysicalReal(7)));
            assert(near(target[donor],450));
            assert(near(temperature[donor],PhysicalReal(262.5)));
        }
        if(massResidual<=1&&energyResidual<=1){converged=true;break;}
    }
    assert(converged);
    MaterialTransportResult physicalRates;
    assert(evaluateMaterialTransport(mesh,state,p,{},1,PhysicalReal(.4),physicalRates,error));
    for(int species=0;species<Ns;++species)
        assert(near(initial.solid[0].pore[species]-state[0].pore[species],physicalRates.faceFlux[0].pore[species]));
    assert(near(fixed[0]-state[0].energy,physicalRates.faceFlux[0].energy));
    if(!darcy){
        assert(near(state[donor].energy,700));
        assert(near(state[receiver].energy,800));
        assert(near(temperature[0],300)&&near(temperature[1],300));
    }
}

void checkCoolingLimits(){
    MaterialCaloric law;
    law.a0=-600;law.a1=-10;law.a2=EnergyReal(.1);
    law.minimumTemperature=300;
    implicitMaterial::CellThermal thermal;
    assert(implicitMaterial::prepareThermalCell(law,PhysicalReal(300),thermal));
    assert(near(thermal.minimum,law.energy(100)));
    assert(thermal.minimum<law.energy(0));

    law={};law.a1=1;law.pore[0]=1;
    auto& gas=law.gas[0];
    gas.model=ugkwp::SpeciesThermoModel::NASA7;gas.R=1;gas.Tmin=300;gas.Tmid=1000;gas.Tmax=6000;
    gas.nasa7[0]=2;gas.nasa7[1]=EnergyReal(-.03);gas.nasa7[2]=EnergyReal(.0001);
    for(int i=0;i<7;++i)gas.nasa7[7+i]=gas.nasa7[i];
    assert(implicitMaterial::prepareThermalCell(law,PhysicalReal(300),thermal));
    assert(near(thermal.minimum,law.energy(200)));
    assert(thermal.minimum<thermal.anchor);
}

int main(){
    checkCoolingLimits();
    checkWindowSum<PhysicalReal>();
    checkWindowSum<EnergyReal>();
    solve(true,false,false);
    solve(true,false,true);
    solve(true,false,false,PhysicalReal(300));
    solve(true,false,true,PhysicalReal(300));
    solve(false,true,false);
    solve(true,true,false);
}
'''
    source = tmp_path / "implicit-face.cpp"
    source.write_text(probe)
    binary = tmp_path / "implicit-face"
    subprocess.run([compiler, "-std=c++17", "-O2", "-DUGKWP_GAS_SPECIES=2",
                    "-I" + str(APP), "-I" + str(ROOT / "common"), str(source),
                    str(APP / "materials/MaterialTransport.C"), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
