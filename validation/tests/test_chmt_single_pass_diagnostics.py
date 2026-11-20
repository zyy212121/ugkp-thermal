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
    assert window.index("collectMaterialWindowExchange(") < window.index("advanceImplicitWindow(")
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
    std::vector<Real> temperature{500};
    std::vector<SolidQ> endpoint(1);
    std::vector<MaterialCaloric> caloric(1);
    std::vector<MaterialPrimitive> primitive(1);
    for(int i=0;i<4;++i){
        mass.reconstruct(physics);
        Real residual=0;
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
