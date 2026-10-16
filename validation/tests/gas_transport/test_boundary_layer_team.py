"""Real slot preparation/publication and persistent-workspace continuation, CPU teams vs HEAD."""
from pathlib import Path
import shutil
import subprocess
import pytest
from test_mixture_state import ROOT,HERE
from test_mixture_transport import fixture
from validation.tests.gas_wall.test_wall_team_equivalence import TEAM

BODY=r"""
#include <iostream>
// Fault injection at the publication boundary still invokes the real publisher:
// unavailable thermochemistry must return false, rather than a mocked status.
#define publishGasBoundaryLayerOutput originalPublishGasBoundaryLayerOutput
#include "gasTransport/GasBoundaryLayerEvaluation.H"
#undef publishGasBoundaryLayerOutput
namespace ugkwp {
template<class S,class R,int N> bool publishGasBoundaryLayerOutput(const S& s,int slot,
 const gaswall::WallInput<R,N>& in,const gaswall::WallOutput<R,N>& out,R area,R sweep){
 auto supplied=in;if(s.testPublishFailureSlot==slot)supplied.model.thermo.coefficients=nullptr;
 return originalPublishGasBoundaryLayerOutput(s,slot,supplied,out,area,sweep);
}}
FIXTURE
TEAM_TYPES
int main(int argc,char**argv){
 using SlotEntry=bool(*)(State&,int,int);
 SlotEntry slotEntry=&ugkwp::evaluateGasBoundaryLayerSlot<false,State>;
 using WorkspaceEntry=bool(*)(const ugkwp::gaswall::WallInput<Real,2>&,const ugkwp::gaswall::WallModelConfig<Real>&,ugkwp::gaswall::WallWorkspace<Real,2,24>&,ugkwp::gaswall::WallOutput<Real,2>&,ugkwp::gaswall::WallStatus&,Real*);
 WorkspaceEntry workspaceEntry=&ugkwp::evaluateGasBoundaryLayerWorkspace<false,Real,2,24>;
 ck(slotEntry&&workspaceEntry,"original entrypoint types unavailable");
 int mode=std::atoi(argv[1]),lanes=std::atoi(argv[2]);State s;initialise(s);
 s.gasMu=.02;s.gasThermalConductivity=.5;s.Uy[1]=10;
 Real diffusion[2]={.01,.01};s.gasSpecies.diffusivity=diffusion;
 auto&w=s.gasBoundaryLayer;w.enabled=true;w.count=2;int fs[3]={-1,0,1},os[2]={0,1},status[2]={0,0};
 ugkwp::GasBoundaryLayerExchange<Real> exchange[2];ugkwp::GasBoundaryLayerSstClosure<Real> closure[2];Real speciesFlux[4]{};
 w.faceSlot=fs;w.ownerSlot=os;w.status=status;w.exchange=exchange;w.sst=closure;w.speciesFlux=speciesFlux;
 auto&m=s.gasBoundaryLayerModel;m.useSuppliedWallState=true;m.config.nodes=24;
 m.config.model=mode==3?ugkwp::gaswall::BoundaryLayerModel::ConstantTransport:ugkwp::gaswall::BoundaryLayerModel::ReactingSst;
 if(mode==4)m.config.model=static_cast<ugkwp::gaswall::BoundaryLayerModel>(99);
 ugkwp::gaswall::WallInput<Real,2> input[2];ugkwp::gaswall::WallOutput<Real,2> output[2];ugkwp::gaswall::WallStatus modelStatus[2];
 auto* workspace=new ugkwp::gaswall::WallWorkspace<Real,2,24>[1];std::memset(workspace,0xff,sizeof(*workspace));
 int faces[2]={1,2},offsets[3]={0,1,2},matching[2]={1,0};Real weights[2]={1,1};
 m.input=input;m.output=output;m.status=modelStatus;m.workspace=workspace;m.workspaceCount=1;m.workspaceCapacity=24;m.faces=faces;m.matchingOffsets=offsets;m.matchingCells=matching;m.matchingWeights=weights;
 for(int i=0;i<2;++i){input[i].temperature=500+20*i;input[i].normal[0]=1;input[i].matchingDistance=1.5;input[i].ownerDistance=.5;input[i].quadrature.volume=1;}
 if(mode==1)matching[0]=s.nCells;
 if(mode==2)s.testPublishFailureSlot=0;
 bool result[32][2]{};
#if WALL_TEAM
 TeamStorage storage(lanes);std::vector<std::thread> threads;
 for(int lane=0;lane<lanes;++lane)threads.emplace_back([&,lane]{CpuTeam team{storage,lane,lanes};for(int slot=0;slot<2;++slot)result[lane][slot]=ugkwp::evaluateGasBoundaryLayerSlot(s,slot,0,team);});
 for(auto& thread:threads)thread.join();
 for(int lane=1;lane<lanes;++lane)for(int slot=0;slot<2;++slot)ck(result[lane][slot]==result[0][slot],"team members returned different slot status");
#else
 (void)lanes;for(int slot=0;slot<2;++slot)result[0][slot]=ugkwp::evaluateGasBoundaryLayerSlot(s,slot,0);
#endif
 std::cout<<std::hexfloat;
 for(int slot=0;slot<2;++slot){auto&e=exchange[slot];auto&o=output[slot];auto&st=modelStatus[slot];
 std::cout<<result[0][slot]<<' '<<status[slot]<<' '<<e.ready<<' '<<e.massReady<<' '<<int(st.code)<<' '<<st.node<<' '<<st.iteration<<' '<<st.residual<<' ';
 for(double value:{e.mass,e.momentumX,e.momentumY,e.momentumZ,e.energy,e.k,e.matchingPressure,closure[slot].wallShearStress,closure[slot].ownerOmega,closure[slot].volume,closure[slot].ownerDistance,o.conductiveHeatFlux,o.residual})std::cout<<value<<' ';
 }
 for(int i=0;i<4;++i)std::cout<<speciesFlux[i]<<' ';
 for(int i=0;i<2;++i)std::cout<<s.rho[i]<<' '<<s.rhoE[i]<<' '<<s.gasSpecies.rho[i]<<' '<<s.gasSpecies.cellStatus[i]<<' ';
 for(int i=0;i<3;++i)std::cout<<s.gasSpecies.faceStatus[i]<<' ';
 std::cout<<'\n';
 if(mode==1||mode==2){ck(!result[0][0]&&!exchange[0].ready,"failed first slot was published");ck(result[0][1]&&exchange[1].ready,"next slot did not continue on reused workspace");}
 if(mode==0||mode==3)ck(result[0][0]&&result[0][1],"valid slot failed");
 if(mode==4)ck(!result[0][0]&&!result[0][1],"unsupported model became a solve");
 delete[] workspace;
}
"""

@pytest.fixture(scope='module')
def slot_programs(tmp_path_factory):
    tmp=tmp_path_factory.mktemp('wall-slot-team');oracle=tmp/'oracle'
    shutil.copytree(ROOT/'common',oracle/'common')
    for name in ('common/gasWall/ReactingWallLayer.H','common/gasTransport/GasBoundaryLayerModelState.H'):
        original=subprocess.run(['git','show','HEAD:'+name],cwd=ROOT,check=True,capture_output=True,text=True).stdout
        (oracle/name).write_text(original)
    src=fixture().replace('ugkwp::GasSpeciesState<Real,2> gasSpecies;};','ugkwp::GasSpeciesState<Real,2> gasSpecies;ugkwp::GasBoundaryLayerModelState<Real,2> gasBoundaryLayerModel;int testPublishFailureSlot=-1;};')
    src='#include "gasTransport/GasBoundaryLayerModelState.H"\n'+src
    # Define the fault boundary wrapper before ModelState is first instantiated.
    types=TEAM.split('template<int Capacity>')[0].replace('#include "common/gasTransport/GasBoundaryLayerModelState.H"','')
    body=BODY.replace('FIXTURE',src).replace('TEAM_TYPES',types)
    pre=(HERE/'gas_state_probe.cpp').read_text().split('void hashBytes')[0]
    cpp=tmp/'slot.cpp';cpp.write_text(pre+body);programs=[]
    for root,enabled in ((oracle,0),(ROOT,1)):
        exe=tmp/('slot-'+str(enabled));cmd=['g++','-std=c++17','-O2','-pthread','-DWALL_TEAM='+str(enabled),'-I'+str(HERE),'-I'+str(root),'-I'+str(root/'common'),'-I'+str(root/'common/gasNumerics'),str(cpp),'-o',str(exe)]
        compiled=subprocess.run(cmd,capture_output=True,text=True);assert compiled.returncode==0,compiled.stderr;programs.append(exe)
    return programs

@pytest.mark.parametrize('mode',range(5))
@pytest.mark.parametrize('lanes',[1,4,32])
def test_slot_team_matches_original_and_reuses_workspace_after_failure(slot_programs,mode,lanes):
    baseline,team=slot_programs
    expected=subprocess.run([str(baseline),str(mode),'1'],capture_output=True,text=True,timeout=60);assert expected.returncode==0,expected.stderr
    actual=subprocess.run([str(team),str(mode),str(lanes)],capture_output=True,text=True,timeout=60);assert actual.returncode==0,actual.stderr
    assert actual.stdout==expected.stdout
