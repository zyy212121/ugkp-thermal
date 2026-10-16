"""Uniform SST boundary physics is evaluated once per team, with original outputs."""
from pathlib import Path
import shutil
import subprocess
import pytest
from validation.tools.paths import SOURCE_ROOT
from validation.tests.gas_wall.test_wall_model import BASE
from validation.tests.gas_wall.test_wall_team_equivalence import TEAM, programs

PREFIX=r"""
#include <atomic>
thread_local int currentLane=0;
std::atomic<int> uniformWallCalls{0},uniformFluxCalls{0},nonleaderCalls{0};
void recordUniformWork(int kind){if(currentLane!=0)++nonleaderCalls;
 if(kind==0)++uniformWallCalls;else ++uniformFluxCalls;}
"""
MAIN=r"""
int main(int argc,char**argv){
 std::cout<<std::hexfloat;const int lanes=std::atoi(argv[1]);run<24>(1,lanes);
 std::cerr<<"uniform_wall_calls="<<uniformWallCalls<<" uniform_flux_calls="<<uniformFluxCalls<<" nonleader_calls="<<nonleaderCalls<<'\n';
 if(nonleaderCalls!=0||uniformWallCalls!=1||uniformFluxCalls<1)return 42;
 return 0;
}
"""

@pytest.fixture(scope='module')
def uniform_program(tmp_path_factory):
    tmp=tmp_path_factory.mktemp('uniform-work');root=tmp/'instrumented'
    shutil.copytree(SOURCE_ROOT/'common',root/'common')
    header=root/'common/gasWall/ReactingWallLayer.H';text=header.read_text()
    for kind,call in enumerate([
        'pointState(in,c,ctx,w.state[0],wall)',
        'intervalFlux(in,c,ctx,w.state[0],w.state[1],w.y[0],w.y[1],first)',
    ]):
        assert text.count(call)==1
        text=text.replace(call,'(::recordUniformWork('+str(kind)+'),'+call+')')
    header.write_text(text)
    body=TEAM[:TEAM.index('int main(int argc,char**argv)')]
    marker='threads.emplace_back([&,lane]{bool ok='
    assert body.count(marker)==1
    body=body.replace(marker,'threads.emplace_back([&,lane]{currentLane=lane;bool ok=')
    source=tmp/'uniform.cpp';source.write_text(PREFIX+BASE+body+MAIN);exe=tmp/'uniform'
    compiled=subprocess.run(['g++','-std=c++17','-O2','-pthread','-DWALL_TEAM=1','-I'+str(root),'-I'+str(root/'common'),str(source),'-o',str(exe)],capture_output=True,text=True)
    assert compiled.returncode==0,compiled.stderr
    return exe

@pytest.mark.parametrize('lanes',[4,1,7,32])
def test_uniform_boundary_work_is_leader_owned_and_original_exact(uniform_program,programs,lanes):
    actual=subprocess.run([str(uniform_program),str(lanes)],capture_output=True,text=True,timeout=60)
    assert actual.returncode==0,actual.stdout+actual.stderr
    original=subprocess.run([str(programs[0]),'1','1'],capture_output=True,text=True,timeout=60)
    assert original.returncode==0,original.stderr
    assert actual.stdout==original.stdout
    print(actual.stderr.strip())
