"""Bound real accepted-state residual and immutable-table work without a solver timing gate."""
import shutil
import subprocess
from validation.tools.paths import SOURCE_ROOT

def test_accepted_candidate_reuses_residual_and_table_preflight(tmp_path):
    shutil.copytree(SOURCE_ROOT/'common',tmp_path/'common')
    h=tmp_path/'common/gasWall/ReactingWallLayer.H'
    text=h.read_text()
    for name,counter in [('allResidual','residualCalls'),('layerTrialStorageAvailable','storageCalls')]:
        start=text.index(' '+name+'(');body=text.index('){',start)+2
        text=text[:body]+'++::'+counter+';'+text[body:]
    h.write_text(text)
    cpp=tmp_path/'work.cpp'
    cpp.write_text(r"""
long long residualCalls=0,storageCalls=0;
#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
#include <cassert>
#include <iostream>
int main(){
 using namespace ugkwp::gaswall;
 CapturedSstWall m;WallWorkspace<double,2,24> w;WallOutput<double,2> o;WallStatus status;
 ugkp::diagnostics::DeviceContext diagnostics{};m.input.diagnostics=&diagnostics;
 assert(evaluateReactingWallLayer(m.input,m.config,w,o,status));
 assert(o.iterations>0&&o.iterations<m.config.maxIterations&&o.residual<m.config.absoluteTolerance+m.config.relativeTolerance);
 assert(!diagnostics.fatalClaim);
 for(auto& warning:diagnostics.warnings)assert(warning.count==0);
 std::cout<<"residual_calls="<<residualCalls<<" storage_calls="<<storageCalls<<std::endl;
 assert(residualCalls>=o.iterations+1&&residualCalls<=2*o.iterations+3); // Initial/accepted/rejected residuals, without a fixed Newton trajectory.
 assert(storageCalls==0); // Complete immutable tables have no temperature-dependent gaps.
}
""")
    exe=tmp_path/'work'
    b=subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-I'+str(tmp_path),'-I'+str(SOURCE_ROOT),str(cpp),'-o',str(exe)],capture_output=True,text=True)
    assert b.returncode==0,b.stdout+b.stderr
    q=subprocess.run([str(exe)],capture_output=True,text=True)
    assert q.returncode==0,q.stdout+q.stderr
