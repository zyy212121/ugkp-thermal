"""Both wall entry points require equation convergence before heat publication."""
import subprocess
from validation.tools.paths import SOURCE_ROOT


def test_reference_rejects_unconverged_candidate_without_publishing_heat(tmp_path):
    source = tmp_path / "acceptance.cpp"
    source.write_text(r'''
#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
#include "common/gasWall/WallReferenceSolver.H"
#include <cassert>
int main(){
 using namespace ugkwp::gaswall;
 CapturedSstWall model;model.config.maxIterations=0;
 WallWorkspace<double,2,24> normal{},reference{};
 WallReferenceScratch<2,24> scratch{};WallReferenceStats stats;
 WallOutput<double,2> a{},b{};a.conductiveHeatFlux=b.conductiveHeatFlux=12345;
 WallStatus sa{},sb{};
 assert(!evaluateReactingWallLayer(model.input,model.config,normal,a,sa));
 assert(!evaluateWallReference(model.input,model.config,reference,scratch,
     static_cast<const double*>(nullptr),b,sb,stats));
 assert(sa.code==WallCode::NonConvergence&&sb.code==WallCode::NonConvergence);
 assert(!stats.converged&&a.conductiveHeatFlux==12345&&b.conductiveHeatFlux==12345);
 model.config.maxIterations=100;
 assert(evaluateWallReference(model.input,model.config,reference,scratch,
     static_cast<const double*>(nullptr),b,sb,stats));
 assert(stats.converged&&stats.residual<=model.config.absoluteTolerance+model.config.relativeTolerance);
 assert(b.conductiveHeatFlux!=12345);
}
''')
    binary = tmp_path / "acceptance"
    subprocess.run(["g++", "-std=c++17", "-O2", "-ffp-contract=off", "-I", str(SOURCE_ROOT),
                    str(source), "-o", str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
