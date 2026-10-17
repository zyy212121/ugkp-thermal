"""PR19: diagnostic ratios cannot veto geometry; invalid inputs retain provenance."""
import subprocess
from validation.tools.paths import SOURCE_ROOT


def build_run(tmp_path, code, geometry=False):
    source = tmp_path / "review.cpp"
    source.write_text(code)
    exe = tmp_path / "review"
    command = ["g++", "-std=c++14", "-O2", "-I" + str(SOURCE_ROOT / "applications/CHMT"), "-I" + str(SOURCE_ROOT / "common"), str(source)]
    if geometry:
        command += [str(SOURCE_ROOT / "applications/CHMT/mesh/Geometry.C")]
    subprocess.run(command + ["-o", str(exe)], check=True, capture_output=True, text=True)
    result = subprocess.run([str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_geometry_diagnostic_ratio_overflow_preserves_finite_geometry(tmp_path):
    fixture = SOURCE_ROOT / "validation/solvers/CHMT/devtools/multirate/test_sweep_constraints.cpp"
    build_run(tmp_path, '#define main original_main\n#include "' + str(fixture) + '"\n#undef main\n' + r''' 
#include "DiagnosticsHost.H"
#include <cassert>
int main(){
    HostMesh mesh=twoHexahedra(0),out;
    std::vector<Vec3> zero(mesh.points.size());
    std::vector<unsigned char> fixed(mesh.points.size(),1);
    std::vector<Real> sweeps;SweepConstraintControls controls;
    controls.maximumIterations=1;
    controls.faceVolumeTolerance={std::numeric_limits<Real>::denorm_min()};
    SweepConstraintReport report;std::string error;
    ugkp::diagnostics::DeviceContext ctx{};Tolerances tol;tol.diagnostics=&ctx;
    assert(constrainFaceSweeps(mesh,{0},{.01},zero,fixed,zero,1,out,sweeps,report,error,controls,tol));
    assert(sweeps[0]==0 && out.points[0].x==mesh.points[0].x);
    assert(!ugkp::diagnostics::hasFatal(&ctx));
    assert(ctx.warnings[0].count+ctx.warnings[1].count>0);
    controls.faceVolumeTolerance={-1};
    assert(!constrainFaceSweeps(mesh,{0},{.01},zero,fixed,zero,1,out,sweeps,report,error,controls,tol));
    controls.faceVolumeTolerance={std::numeric_limits<Real>::quiet_NaN()};
    assert(!constrainFaceSweeps(mesh,{0},{.01},zero,fixed,zero,1,out,sweeps,report,error,controls,tol));
    controls.faceVolumeTolerance={std::numeric_limits<Real>::denorm_min()};
    zero[0].x=std::numeric_limits<Real>::quiet_NaN();
    assert(!constrainFaceSweeps(mesh,{0},{.01},zero,fixed,zero,1,out,sweeps,report,error,controls,tol));
}
''', True)


def test_gas_recovery_records_actual_bad_input_and_cell(tmp_path):
    build_run(tmp_path, r'''
#include "materials/Thermodynamics.H"
#include <cassert>
using namespace chmt;
using namespace ugkp::diagnostics;
int main(){
    DeviceContext context{};reset(context,8,99,4,.25);
    PhysicsConfig p;p.diagnostics=&context;p.tolerances.diagnostics=&context;
    GasQ q{};q.mass=-123;GasPrimitive out{};
    assert(!recoverGas(q,1,p,out,nullptr,7));
    assert(hasFatal(&context));
    assert(context.firstFatal.observed==-123);
    assert(context.firstFatal.entityKind==EntityKind::Cell && context.firstFatal.entityIndex==7);
    assert(context.firstFatal.step==4 && context.firstFatal.simulationTime==.25);
    reset(context,8,99,4,.25);q.mass=1;q.momentum.y=std::numeric_limits<Real>::infinity();
    assert(!recoverGas(q,1,p,out,nullptr,9));
    assert(context.firstFatal.code==Code::NonFinite && context.firstFatal.component==1);
    assert(context.firstFatal.observed==q.momentum.y && context.firstFatal.entityIndex==9);
    reset(context,8,99,4,.25);q.momentum={};q.species[1]=std::numeric_limits<Real>::quiet_NaN();
    assert(!recoverGas(q,1,p,out,nullptr,11));
    assert(context.firstFatal.code==Code::NonFinite && context.firstFatal.component==1);
    assert(std::isnan(context.firstFatal.observed) && context.firstFatal.entityIndex==11);
}
''')
