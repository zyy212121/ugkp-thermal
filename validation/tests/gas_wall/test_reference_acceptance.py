"""Bounded wall solves publish finite heat and report accuracy at the iteration limit."""
import subprocess
from validation.tools.paths import SOURCE_ROOT


def test_reference_publishes_finite_heat_and_bounds_residual_evaluations(tmp_path):
    source = tmp_path / "acceptance.cpp"
    source.write_text(r'''
#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
#include "common/gasWall/WallReferenceSolver.H"
#include <cassert>
int main(){
 using namespace ugkwp::gaswall;
 using R=WallReal; CapturedSstWallOf<R> model;
 ugkp::diagnostics::DeviceContext diagnostics{};model.input.diagnostics=&diagnostics;
 model.config.maxIterations=0;
 WallWorkspace<R,2,24> normal{},reference{};
 WallReferenceScratch<2,24> scratch{};WallReferenceStats stats;
 WallOutput<R,2> a{},b{};a.conductiveHeatFlux=b.conductiveHeatFlux=12345;
 WallStatus sa{},sb{};
 assert(evaluateReactingWallLayer(model.input,model.config,normal,a,sa));
 assert(evaluateWallReference(model.input,model.config,reference,scratch,
     static_cast<const R*>(nullptr),b,sb,stats));
 assert(sa.code==WallCode::NonConvergence&&sb.code==WallCode::NonConvergence);
 assert(!stats.converged&&std::isfinite(a.conductiveHeatFlux)&&std::isfinite(b.conductiveHeatFlux));
 assert(a.conductiveHeatFlux!=12345&&b.conductiveHeatFlux!=12345&&stats.residualEvaluations==1);
 assert(!diagnostics.fatalClaim);
 const auto slot=static_cast<unsigned>(ugkp::diagnostics::Code::AccuracyResidual)-101;
 assert(diagnostics.warnings[slot].count>=2);
 model.input.diagnostics=nullptr;
 model.config.maxIterations=1;
 assert(evaluateWallReference(model.input,model.config,reference,scratch,
     static_cast<const R*>(nullptr),b,sb,stats));
 assert(stats.iterations<=1&&stats.residualEvaluations<=3&&std::isfinite(b.conductiveHeatFlux));
 model.config.absoluteTolerance=stats.residual*R(2)+R(1);
 assert(evaluateWallReference(model.input,model.config,reference,scratch,
     &reference.state[0][0],b,sb,stats));
 assert(stats.converged&&stats.iterations==0&&stats.residualEvaluations==1);
}
''')
    binary = tmp_path / "acceptance"
    subprocess.run(["g++", "-std=c++17", "-O2", "-ffp-contract=off", "-I", str(SOURCE_ROOT),
                    str(source), "-o", str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
