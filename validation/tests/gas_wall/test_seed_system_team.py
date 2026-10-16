"""Independent SST systems use disjoint recurrence scratch and distinct team lanes."""
import shutil
import subprocess
from pathlib import Path
import pytest
from validation.tools.paths import SOURCE_ROOT
from validation.tests.gas_wall.test_wall_team_equivalence import TEAM

CPP=r"""
#include <iostream>
#include <iomanip>
#include <thread>
int jobOwner[6]={};thread_local int currentLane=0;
void recordSeedJob(int variable){jobOwner[variable]=currentLane;}
#include "common/gasWall/WallModel.H"
namespace ugkwp { namespace gaswall { namespace detail {
ORIGINAL_SEED_FUNCTION
}}}
TEAM_TYPES
int main(int argc,char**argv){
 using namespace ugkwp::gaswall;int lanes=std::atoi(argv[1]),mode=std::atoi(argv[2]);
 WallModelConfig<double> c;c.nodes=mode==4?2:(mode==5?65:24);c.enableSst=true;
 WallWorkspace<double,2,128> serial{},parallel{};
 for(int i=0;i<c.nodes;++i){serial.y[i]=.01*i/(c.nodes-1);
  for(int j=0;j<6;++j)serial.state[i][j]=parallel.state[i][j]=.1*(i+j+1);
  for(int j=0;j<8;++j)serial.seed[i][j]=.2+.1*j;}
 if(mode==1)for(int i=0;i<c.nodes;++i)serial.seed[i][1]=-.3;
 if(mode==2)for(int i=0;i<c.nodes;++i)serial.seed[i][2]=-.4;
 parallel=serial;const double mass=mode==3?-1e8:.001,firstOmega=200;
 bool expected=detail::originalSeedScalar(c,serial,0,0,-1,-1,mass,0.,10.)
  &&detail::originalSeedScalar(c,serial,1,0,-1,-1,mass,0.,-3.)
  &&detail::originalSeedScalar(c,serial,4,1,3,4,mass,0.,.5)
  &&detail::originalSeedScalar(c,serial,5,2,5,6,mass,firstOmega,200.,1);
 for(auto& v:jobOwner)v=-1;
 TeamStorage storage(lanes);std::vector<std::thread> threads;bool result[32]{};
 for(int lane=0;lane<lanes;++lane)threads.emplace_back([&,lane]{currentLane=lane;
  result[lane]=detail::solveSeedSystemsWithTeam(c,parallel,mass,10.,-3.,.5,firstOmega,200.,CpuTeam{storage,lane,lanes});});
 for(auto& thread:threads)thread.join();
 for(int lane=0;lane<lanes;++lane)if(result[lane]!=expected)return 2;
 if(expected){for(int job=0;job<4;++job){const int variable=job<2?job:job+2;
   if(jobOwner[variable]!=job%lanes)return 3;
   for(int i=0;i<c.nodes;++i)if(serial.delta[i][variable]!=parallel.delta[i][variable])return 4;}}
 for(int i=0;i<c.nodes;++i)for(int j=0;j<6;++j)if(serial.state[i][j]!=parallel.state[i][j])return 5;
 std::cout<<expected<<'\n';
}
"""

@pytest.fixture(scope='module')
def seed_program(tmp_path_factory):
    tmp=tmp_path_factory.mktemp('seed-team');root=tmp/'instrumented'
    shutil.copytree(SOURCE_ROOT/'common',root/'common')
    path=root/'common/gasWall/ReactingWallLayer.H'
    text=path.read_text()
    marker='    // Each system owns a distinct upper-recurrence row.'
    assert text.count(marker)==1
    text=text.replace(marker,'    ::recordSeedJob(variable);\n'+marker)
    path.write_text(text)
    types=TEAM.split('template<int Capacity>')[0].replace('#include "common/gasTransport/GasBoundaryLayerModelState.H"','')
    original=subprocess.run(['git','show','HEAD:common/gasWall/ReactingWallLayer.H'],cwd=SOURCE_ROOT,check=True,capture_output=True,text=True).stdout
    start=original.index('template<class R,int N,int M> UGKWP_GAS_HD inline bool seedScalar(')
    end=original.index('\n}\n',start)+3
    seed=original[start:end].replace('bool seedScalar(','bool originalSeedScalar(',1)
    source=tmp/'seed.cpp';source.write_text(CPP.replace('TEAM_TYPES',types).replace('ORIGINAL_SEED_FUNCTION',seed))
    exe=tmp/'seed'
    result=subprocess.run(['g++','-std=c++14','-Werror','-O2','-pthread','-I'+str(root),'-I'+str(root/'common'),str(source),'-o',str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    return exe

@pytest.mark.parametrize('lanes',[1,4,7,32])
@pytest.mark.parametrize('mode',range(6))
def test_four_seed_systems_match_original_and_have_disjoint_owners(seed_program,lanes,mode):
    run=subprocess.run([str(seed_program),str(lanes),str(mode)],capture_output=True,text=True,timeout=20)
    assert run.returncode==0,run.stdout+run.stderr


def test_seed_simt_recurrences_share_one_callsite():
    text=(SOURCE_ROOT/'common/gasWall/ReactingWallLayer.H').read_text()
    start=text.index('const Team& team,std::false_type){',text.index('bool solveSeedSystemsWithTeam('))
    end=text.index('\ntemplate<class',start)
    # Different large if/else callsites can serialize an entire warp. Select
    # the system parameters first, then execute one common recurrence body.
    assert text[start:end].count('seedScalarUsingScratch(')==1
