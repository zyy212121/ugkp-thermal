#define main historical_geometry_main
#include "../../../solvers/CHMT/devtools/multirate/test_sweep_constraints.cpp"
#undef main
#include "DiagnosticsHost.H"
#include <limits>
int main() {
    actualNonuniformDrySweep();
    rejectInvalidTrajectory();
    rebuildActualTrajectorySurface();
    rejectInteriorBoundaryDof();
    retainMovingOffsetRejection();
    scaledRotatedDryMotion();
    planarWetMotionStillSupported();
    ugkp::diagnostics::DeviceContext diagnostics{};
    ugkp::diagnostics::reset(diagnostics,1,34);
    Tolerances tolerance; tolerance.diagnostics=&diagnostics;
    HostMesh mesh=twoHexahedra(0),out;
    std::vector<Vec3> zero(mesh.points.size());
    std::vector<unsigned char> fixed(mesh.points.size(),1);
    std::vector<Real> sweeps;
    SweepConstraintControls controls; controls.maximumIterations=1;
    SweepConstraintReport report; std::string error;
    require(constrainFaceSweeps(mesh,{0},{.01},zero,fixed,zero,1,out,sweeps,report,error,controls,tolerance),error);
    require(report.iterations==1 && report.lineSearchTrials==0,"exactly one full correction, no candidate search");
    require(sweeps[0]==0 && out.points[0].x==mesh.points[0].x,"finite unmet target retains actual unchanged geometry");
    auto snapshot=ugkp::diagnostics::takeSnapshot(diagnostics);
    require(!snapshot.hasFatal && !snapshot.warnings.empty(),"unmet finite residual has warning");
    zero[0].x=std::numeric_limits<Real>::quiet_NaN();
    require(!constrainFaceSweeps(mesh,{0},{.01},zero,fixed,zero,1,out,sweeps,report,error,controls,tolerance),"NaN fails rather than terminal completion");
    HostState base; base.solidMesh=mesh; base.surface.solidFace={0}; base.solidSweepRemainder={.25};
    require(validateMaterialSweepRemainder(base,error,&diagnostics),"finite remainder warns");
    require(base.solidSweepRemainder[0]==.25,"warning preserves remainder");
    base.solidSweepRemainder[0]=std::numeric_limits<Real>::quiet_NaN();
    require(!validateMaterialSweepRemainder(base,error,&diagnostics),"NaN remainder fails");
    std::cout << "PASS: single-pass geometry, finite residual warnings, physical failures\n";
}
