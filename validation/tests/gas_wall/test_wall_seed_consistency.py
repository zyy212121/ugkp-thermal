"""Catch premature tiny-k Picard termination and a seed for a different operator."""
import subprocess
import pytest
from validation.tools.paths import SOURCE_ROOT

SOURCE = r'''
#include "validation/tests/gas_wall/fixtures/sst_small_k_column_capture.H"
#include <cassert>
#include <cstdio>
#include <cstdlib>
int main(int argc,char**argv){using namespace ugkwp::gaswall;using namespace ugkwp::gaswall::detail;
 const int nodes=std::atoi(argv[1]);const bool tiny=std::atoi(argv[2])!=0;
 CapturedSingularWall small;CapturedSstWall regular;
 auto&in=tiny?small.input:regular.input;auto c=tiny?small.config:regular.config;
 c.nodes=nodes;in.integrateOwnerKSource=false;in.volumeSstClosure=false;
 ugkp::diagnostics::DeviceContext diagnostic{};in.diagnostics=&diagnostic;
 WallWorkspace<double,2,128>w;WallOutput<double,2>out;WallStatus status;
 assert(evaluateWallModel(in,c,w,out,status));
 assert(!diagnostic.fatalClaim);
 // The old warning-and-publish return value was true even while this failed.
 assert(out.residual<=c.absoluteTolerance+c.relativeTolerance);
 LayerContext<double,2>ctx;assert(makeContext(in,c,ctx));double checked=0;
 assert(allResidual(in,c,ctx,w.state,w.y,w.residual,checked));
 assert(checked<=c.absoluteTolerance+c.relativeTolerance);
 assert(checked==out.residual);
 assert(w.state[0][4]==0&&w.state[nodes-1][4]==std::sqrt(in.matching.k));
 assert(w.state[nodes-1][5]==1/std::sqrt(in.matching.omega));
 // Frozen practical workload: a consistent initializer should not require
 // a long Newton repair of its own diffusion/destruction discretization.
 assert(out.iterations<=5);
 std::printf("nodes=%d tiny=%d iterations=%d residual=%.17g tau=%.17g\n",
             nodes,int(tiny),out.iterations,checked,out.traction[0]);
}
'''

@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    path=tmp_path_factory.mktemp("seed-consistency")
    source=path/"probe.cpp";source.write_text(SOURCE)
    binary=path/"probe"
    subprocess.run(["g++","-std=c++17","-O2","-ffp-contract=off",
                    "-I"+str(SOURCE_ROOT),str(source),"-o",str(binary)],
                   check=True,capture_output=True,text=True)
    return binary

@pytest.mark.parametrize("nodes",[24,48,128])
@pytest.mark.parametrize("tiny",[False,True],ids=["restart","tiny-k"])
def test_captured_profiles_reach_verified_residual_without_newton_stagnation(binary,nodes,tiny):
    run=subprocess.run([str(binary),str(nodes),str(int(tiny))],capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
    print(run.stdout,end="")
