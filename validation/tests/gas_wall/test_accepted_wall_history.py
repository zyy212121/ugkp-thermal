"""Accepted history coordinate and geometry semantics, no solver tolerance changes."""
import subprocess
from validation.tools.paths import SOURCE_ROOT

CPP=r"""
#include <cassert>
#include <cmath>
#include <cstring>
#include "common/gasWall/WallAcceptedHistory.H"
using namespace ugkwp::gaswall;
int main(){
 WallHistoryFrame<3> a;a.valid=true;a.nodes=24;a.distance=.001;a.stretch=2;a.temperature=300;a.speciesOrder=7;
 a.tangent[0][0]=a.tangent[1][2]=1;a.wallU[0]=5;a.wallU[1]=2;a.independent[0]=1;a.independent[1]=2;a.dependent=0;
 double x[7]={12,-3,4,.2,.1,2,.001},y[7]{};transformHistoryRow(x,y,a,a);assert(!std::memcmp(x,y,sizeof(x)));
 auto b=a;b.temperature=302;b.wallU[0]=8;b.wallU[1]=-1;b.independent[0]=0;b.independent[1]=2;b.dependent=1;
 assert(compatibleHistory(a,b));transformHistoryRow(x,y,a,b);
 assert(y[0]==9&&y[1]==0&&y[2]==2&&std::abs(y[3]-.7)<1e-15&&y[4]==.1&&y[5]==2&&y[6]==.001);
 b.distance*=2;assert(!compatibleHistory(a,b));b=a;b.nodes=48;assert(!compatibleHistory(a,b));b=a;b.speciesOrder=8;assert(!compatibleHistory(a,b));b=a;b.normal[0]=1;assert(!compatibleHistory(a,b));b=a;b.valid=false;assert(!compatibleHistory(b,a));
 // A worker's trial coordinate conversion must not alter immutable accepted state.
 assert(x[0]==12&&x[3]==.2);
}
"""
def test_history_coordinates_and_invalidation(tmp_path):
    src=tmp_path/'history.cpp';src.write_text(CPP);exe=tmp_path/'history'
    build=subprocess.run(['g++','-std=c++17','-O2','-I'+str(SOURCE_ROOT),'-I'+str(SOURCE_ROOT/'common'),str(src),'-o',str(exe)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
