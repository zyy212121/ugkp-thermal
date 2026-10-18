#ifndef CHMT_MESH_SWEEP_CONSTRAINTS_H
#define CHMT_MESH_SWEEP_CONSTRAINTS_H
#include "mesh/Motion.H"
#include "mesh/GeometryDiagnostics.H"
namespace chmt {
// Fixed-topology Newton solve. Invalid geometry and resource failures return
// failure; finite conservation/accuracy residuals are warnings, with no replay.
struct SweepConstraintControls {
    Real absoluteVolumeTolerance=1e-14, relativeVolumeTolerance=1e-10;
    // Optional per-face accuracy thresholds used to report residual warnings.
    std::vector<Real> faceVolumeTolerance{};
    Real rankTolerance=1e-11;
    int maximumIterations=24;
    int maximumLineSearch=24; // Bounded mathematical damping at fixed target/interval.
    // Compatibility-only fields. Allocation is limited by actual representable
    // sizes and allocator availability, never by a configured policy cap.
    int maximumDegreesOfFreedom=0, maximumConstraints=0;
    std::size_t maximumModeEntries=0;
    std::size_t maximumWorkspaceBytes=0;
};
// Bound both cell-inventory arithmetic and endpoint coordinate quantization.
// A thin cell far from the coordinate origin cannot resolve eight epsilon of
// its volume in world-coordinate vertices. Its actual face-area-weighted ULP
// bound is retained in the signed remainder, not discarded or charged to mass.
// Requested accuracy, explicit face ceilings, EOS and GCL are checked separately.
inline bool materialSweepResidualLimits(const HostMesh& mesh,const std::vector<int>& faces,
    std::vector<Real>& output) {
    std::vector<int> count(mesh.volumes.size());std::vector<unsigned char> seen(mesh.owner.size());
    for (int face:faces) {
        if (face< -1)return false;
        if (face== -1)continue;
        if ((static_cast<std::size_t>(face)>=mesh.owner.size()||seen[face]++))return false;
        const int cell=mesh.owner[face];
        if ((cell<0||static_cast<std::size_t>(cell)>=mesh.volumes.size()
            ||!finite(mesh.volumes[cell])||mesh.volumes[cell]<=0))return false;
        ++count[cell];
    }
    std::vector<Real> limits(faces.size(),0);
    for (std::size_t i=0;i<faces.size();++i)if(faces[i]>=0) {
        const int cell=mesh.owner[faces[i]];

        // Inventory arithmetic and endpoint representation are independent
        // error sources. Bound the latter from this face's actual coordinate
        // spacing, rather than treating a global coordinate as cell-local.
        const int f=faces[i],begin=mesh.faceOffsets[f],end=mesh.faceOffsets[f+1];
        Real representation=0;
        auto ulp=[](Real x){return maxValue(absValue(std::nextafter(x,std::numeric_limits<Real>::infinity())-x),
            absValue(std::nextafter(x,-std::numeric_limits<Real>::infinity())-x));};
        for(int k=begin+1;k+1<end;++k){
            const Vec3 a=mesh.points[mesh.facePoints[begin]],b=mesh.points[mesh.facePoints[k]],c=mesh.points[mesh.facePoints[k+1]];
            const Vec3 area=cross(b-a,c-a)*.5;
            for(Vec3 v:{a,b,c})representation+=(absValue(area.x)*ulp(v.x)+absValue(area.y)*ulp(v.y)+absValue(area.z)*ulp(v.z))/3;
        }
        limits[i]=8*std::numeric_limits<Real>::epsilon()*mesh.volumes[cell]/count[cell]+2*representation;

        if ((!(limits[i]>0)||!finite(limits[i])))return false;
    }
    output=std::move(limits);return true;
}
inline bool validateMaterialSweepRemainder(const HostState& state,std::string& error,ugkp::diagnostics::DeviceContext* diagnostics=nullptr) {
    if(state.solidSweepRemainder.empty()){error.clear();return true;}
    std::vector<Real> limits;
    if(state.solidSweepRemainder.size()!=state.surface.solidFace.size()
        ||!materialSweepResidualLimits(state.solidMesh,state.surface.solidFace,limits)){
        error="material swept-volume remainder mapping mismatch";return false;}
    for(std::size_t f=0;f<limits.size();++f) {
        if(!finite(state.solidSweepRemainder[f])) {error="nonfinite material swept-volume remainder";return false;}
        if(absValue(state.solidSweepRemainder[f])>4*limits[f])
            geometryWarning(diagnostics,state.solidSweepRemainder[f],4*limits[f],f);
    }
    error.clear();return true;
}
inline bool compensateMaterialSweepTargets(const HostState& base,const std::vector<Real>& physical,
    std::vector<Real>& output,std::string& error,ugkp::diagnostics::DeviceContext* diagnostics=nullptr) {
    if(physical.size()!=base.surface.solidFace.size()){error="material swept-volume target mapping mismatch";return false;}
    if(!validateMaterialSweepRemainder(base,error,diagnostics))return false;
    std::vector<Real> solve(physical.size());
    for(std::size_t f=0;f<physical.size();++f){
        if((!finite(physical[f])||(base.surface.solidFace[f]<0&&physical[f]!=0))){
            error="invalid physical material swept-volume target";return false;}
        solve[f]=static_cast<Real>(static_cast<long double>(physical[f])
            +(base.solidSweepRemainder.empty()?0:base.solidSweepRemainder[f]));
        if(!finite(solve[f])){error="nonfinite compensated material swept-volume target";return false;}
    }
    output=std::move(solve);error.clear();return true;
}
inline bool updateMaterialSweepRemainder(const HostState& base,const std::vector<Real>& physical,
    const std::vector<Real>& measured,const HostMesh& endpoint,std::vector<Real>& output,std::string& error,ugkp::diagnostics::DeviceContext* diagnostics=nullptr) {
    std::vector<Real> solve,limits;
    if(!compensateMaterialSweepTargets(base,physical,solve,error,diagnostics))return false;
    if((measured.size()!=endpoint.owner.size()||endpoint.owner!=base.solidMesh.owner
        ||endpoint.topologyHash!=base.solidMesh.topologyHash
        ||!materialSweepResidualLimits(endpoint,base.surface.solidFace,limits))){
        error="material swept-volume remainder endpoint mismatch";return false;}
    std::vector<Real> next(physical.size());
    for(std::size_t f=0;f<physical.size();++f){const int face=base.surface.solidFace[f];
        const Real actual=face<0?0:measured[face];
        if(!finite(actual)){error="nonfinite measured material swept volume";return false;}
        next[f]=static_cast<Real>((base.solidSweepRemainder.empty()?0:static_cast<long double>(base.solidSweepRemainder[f]))
            +static_cast<long double>(physical[f])-actual);
        if(!finite(next[f])){error="nonfinite updated material swept-volume remainder";return false;}
        if(absValue(next[f])>4*limits[f])geometryWarning(diagnostics,next[f],4*limits[f],f);
    }
    output=std::move(next);error.clear();return true;
}
enum class SweepConstraintStatus {
    Success, InvalidInput, SizeLimit, Incompatible, InvalidTrajectory, Nonconverged
};
struct SweepConstraintReport {
    SweepConstraintStatus status=SweepConstraintStatus::InvalidInput;
    int iterations=0, rank=0, constraints=0, degreesOfFreedom=0, lineSearchTrials=0;
    Real maximumSweepResidual=0; // m^3, measured from actual vertex sweeps.
    Real maximumPlanarityResidual=0; // dimensionless path-polynomial residual.
    bool streamedModes=false;
    std::size_t peakWorkspaceBytes=0,modeSolves=0;
};
// targetSweeps are actual signed volumes [m^3], outward from mesh.owner.
// prescribedDisplacement is only the initial guess. Each nonzero correction
// direction provides one scalar boundary DOF; prescribed tangential motion is
// preserved. The fixed mask also supplies harmonic interior Dirichlet anchors.
// Corrections minimize lumped incident-face-area-weighted point motion; interior
// modes are harmonic. A 32-epsilon geometric roundoff floor accompanies volume
// tolerances unless a hard face ceiling requires tighter accuracy. Old planar polygons remain planar over the entire linear
// path. Mesh/sweeps outputs change only on success; report also explains failure.
bool constrainFaceSweeps(const HostMesh&, const std::vector<int>& faces,
    const std::vector<Real>& targetSweeps, const std::vector<Vec3>& prescribedDisplacement,
    const std::vector<unsigned char>& prescribed, const std::vector<Vec3>& correctionDirections,
    Real interval, HostMesh& output, std::vector<Real>& actualSweeps,
    SweepConstraintReport&, std::string& error,
    const SweepConstraintControls& controls=SweepConstraintControls{},
    const Tolerances& tolerances=Tolerances{},bool* numericalFailure=nullptr);

// Production multirate entry: solidTargetSweeps are constitutively integrated
// consumed-volume targets, indexed by accepted.surface.solidFace. They must not
// be reconstructed from a macro-end porosity, a front, or a mean mass rate.
// Legacy moveCoupledMeshes instead uses old face area * front increment and is
// retained only for the explicit compatibility path.
bool moveCoupledMeshesConstrained(const HostState&, const std::vector<FilmAux>&,
    const std::vector<Real>& solidTargetSweeps, Real interval,
    HostMesh& gas, HostMesh& solid, SurfaceMesh&, SweepConstraintReport&,
    std::string& error, const SweepConstraintControls& controls=SweepConstraintControls{},
    const Tolerances& tolerances=Tolerances{},bool* numericalFailure=nullptr);
}
#endif
