#ifndef CHMT_MESH_SWEEP_CONSTRAINTS_H
#define CHMT_MESH_SWEEP_CONSTRAINTS_H
#include "mesh/Motion.H"
namespace chmt {
// Bounded dense fixed-topology solve. A limit/rank/trajectory failure is a
// rejection, never permission to rescale physical transfers or prescribe GCL.
struct SweepConstraintControls {
    Real absoluteVolumeTolerance=1e-14, relativeVolumeTolerance=1e-10;
    // Optional hard ceilings, one per constrained face. Neither the requested
    // tolerance nor the geometric floating-point floor can override them.
    std::vector<Real> faceVolumeTolerance{};
    Real rankTolerance=1e-11;
    int maximumIterations=24, maximumLineSearch=24;
    int maximumDegreesOfFreedom=512, maximumConstraints=4096;
    std::size_t maximumModeEntries=2000000;
};
// Allocate a local interface-volume error budget across each cell's constrained
// faces. Eight epsilon leaves room under the unchanged 32-epsilon material EOS
// check; it is a design budget, not a replacement for final EOS/GCL validation.
inline bool materialSweepResidualLimits(const HostMesh& mesh,const std::vector<int>& faces,
    std::vector<Real>& output) {
    std::vector<int> count(mesh.volumes.size());std::vector<unsigned char> seen(mesh.owner.size());
    for (int face:faces) {
        if (face< -1)return false;
        if (face== -1)continue;
        if (static_cast<std::size_t>(face)>=mesh.owner.size()||seen[face]++)return false;
        const int cell=mesh.owner[face];
        if (cell<0||static_cast<std::size_t>(cell)>=mesh.volumes.size()
            ||!finite(mesh.volumes[cell])||mesh.volumes[cell]<=0)return false;
        ++count[cell];
    }
    std::vector<Real> limits(faces.size(),0);
    for (std::size_t i=0;i<faces.size();++i)if(faces[i]>=0) {
        const int cell=mesh.owner[faces[i]];
        limits[i]=8*std::numeric_limits<Real>::epsilon()*mesh.volumes[cell]/count[cell];
        if (!(limits[i]>0)||!finite(limits[i]))return false;
    }
    output=std::move(limits);return true;
}
inline bool validateMaterialSweepRemainder(const HostState& state,std::string& error) {
    if(state.solidSweepRemainder.empty()){error.clear();return true;}
    std::vector<Real> limits;
    if(state.solidSweepRemainder.size()!=state.surface.solidFace.size()
        ||!materialSweepResidualLimits(state.solidMesh,state.surface.solidFace,limits)){
        error="material swept-volume remainder mapping mismatch";return false;}
    for(std::size_t f=0;f<limits.size();++f)if(!finite(state.solidSweepRemainder[f])
        ||absValue(state.solidSweepRemainder[f])>4*limits[f]){
        error="material swept-volume remainder exceeds roundoff bound";return false;}
    error.clear();return true;
}
inline bool compensateMaterialSweepTargets(const HostState& base,const std::vector<Real>& physical,
    std::vector<Real>& output,std::string& error) {
    if(physical.size()!=base.surface.solidFace.size()){error="material swept-volume target mapping mismatch";return false;}
    if(!validateMaterialSweepRemainder(base,error))return false;
    std::vector<Real> solve(physical.size());
    for(std::size_t f=0;f<physical.size();++f){
        if(!finite(physical[f])||(base.surface.solidFace[f]<0&&physical[f]!=0)){
            error="invalid physical material swept-volume target";return false;}
        solve[f]=static_cast<Real>(static_cast<long double>(physical[f])
            +(base.solidSweepRemainder.empty()?0:base.solidSweepRemainder[f]));
        if(!finite(solve[f])){error="nonfinite compensated material swept-volume target";return false;}
    }
    output=std::move(solve);error.clear();return true;
}
inline bool updateMaterialSweepRemainder(const HostState& base,const std::vector<Real>& physical,
    const std::vector<Real>& measured,const HostMesh& endpoint,std::vector<Real>& output,std::string& error) {
    std::vector<Real> solve,limits;
    if(!compensateMaterialSweepTargets(base,physical,solve,error))return false;
    if(measured.size()!=endpoint.owner.size()||endpoint.owner!=base.solidMesh.owner
        ||endpoint.topologyHash!=base.solidMesh.topologyHash
        ||!materialSweepResidualLimits(endpoint,base.surface.solidFace,limits)){
        error="material swept-volume remainder endpoint mismatch";return false;}
    std::vector<Real> next(physical.size());
    for(std::size_t f=0;f<physical.size();++f){const int face=base.surface.solidFace[f];
        const Real actual=face<0?0:measured[face];
        if(!finite(actual)){error="nonfinite measured material swept volume";return false;}
        next[f]=static_cast<Real>((base.solidSweepRemainder.empty()?0:static_cast<long double>(base.solidSweepRemainder[f]))
            +static_cast<long double>(physical[f])-actual);
        if(!finite(next[f])||absValue(next[f])>4*limits[f]){
            error="updated material swept-volume remainder exceeds roundoff bound";return false;}
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
    const SweepConstraintControls& controls=SweepConstraintControls{});

// Production multirate entry: solidTargetSweeps are constitutively integrated
// consumed-volume targets, indexed by accepted.surface.solidFace. They must not
// be reconstructed from a macro-end porosity, a front, or a mean mass rate.
// Legacy moveCoupledMeshes instead uses old face area * front increment and is
// retained only for the explicit compatibility path.
bool moveCoupledMeshesConstrained(const HostState&, const std::vector<FilmAux>&,
    const std::vector<Real>& solidTargetSweeps, Real interval,
    HostMesh& gas, HostMesh& solid, SurfaceMesh&, SweepConstraintReport&,
    std::string& error, const SweepConstraintControls& controls=SweepConstraintControls{});
}
#endif
