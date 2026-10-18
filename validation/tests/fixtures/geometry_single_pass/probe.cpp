#define main historical_geometry_main
#include "../../../solvers/CHMT/devtools/multirate/test_sweep_constraints.cpp"
#undef main
#include "DiagnosticsHost.H"
#include <limits>
int main() {
    actualNonuniformDrySweep();
    // An unattainable finite target is an unresolved accuracy residual; the
    // published geometry itself must remain valid and its residual visible.
    {const HostState base=coupledPair();HostMesh gas,solid;SurfaceMesh surface;
     ugkp::diagnostics::DeviceContext d{};Tolerances t;t.diagnostics=&d;
     SweepConstraintReport r;std::string e;
     require(moveCoupledMeshesConstrained(base,base.filmAux,{-3,-3},1,gas,solid,surface,r,e,SweepConstraintControls{},t),e);
     for(Real v:gas.volumes)require(v>0&&std::isfinite(v),"invalid gas endpoint published");
     for(Real v:solid.volumes)require(v>0&&std::isfinite(v),"invalid material endpoint published");
     const auto snap=ugkp::diagnostics::takeSnapshot(d);
     require(r.maximumSweepResidual>1&&!snap.hasFatal&&!snap.warnings.empty(),"finite impossible target was silently called converged");}
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
    require(report.rank==0 && report.lineSearchTrials==0,"zero correction does not waste candidate evaluations");
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
    std::cout << "PASS: fixed-interval geometry, finite residual warnings, physical failures\n";
}
