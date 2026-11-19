#ifndef CHMT_MESH_SWEEP_CONSTRAINTS_H
#define CHMT_MESH_SWEEP_CONSTRAINTS_H
#include "mesh/Motion.H"
#include "core/NumericalControls.H"
#include "mesh/GeometryDiagnostics.H"
namespace chmt {


struct SweepConstraintControls {
    Real absoluteVolumeTolerance=1e-14, relativeVolumeTolerance=1e-10;

    std::vector<Real> faceVolumeTolerance{};
    Real rankTolerance=1e-11;
    int maximumIterations=24;
    int harmonicMaximumIterations=20000;
    Real harmonicRelativeTolerance=1e-12;
    int maximumLineSearch=24;


    int maximumDegreesOfFreedom=0, maximumConstraints=0;
    std::size_t maximumModeEntries=0;
    std::size_t maximumWorkspaceBytes=0;
};
inline SweepConstraintControls materialSweepControls(const MeshNumericalControls& input) {
    SweepConstraintControls controls;
    controls.absoluteVolumeTolerance=input.absoluteVolumeTolerance;
    controls.relativeVolumeTolerance=input.relativeVolumeTolerance;
    controls.rankTolerance=input.rankTolerance;
    controls.maximumIterations=input.maximumIterations;
    controls.maximumLineSearch=input.maximumLineSearch;
    controls.harmonicMaximumIterations=input.harmonicMaximumIterations;
    controls.harmonicRelativeTolerance=input.harmonicRelativeTolerance;
    return controls;
}





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
    Real maximumSweepResidual=0;
    Real maximumPlanarityResidual=0;
    bool streamedModes=false;
    std::size_t peakWorkspaceBytes=0,modeSolves=0;
};








bool constrainFaceSweeps(const HostMesh&, const std::vector<int>& faces,
    const std::vector<Real>& targetSweeps, const std::vector<Vec3>& prescribedDisplacement,
    const std::vector<unsigned char>& prescribed, const std::vector<Vec3>& correctionDirections,
    Real interval, HostMesh& output, std::vector<Real>& actualSweeps,
    SweepConstraintReport&, std::string& error,
    const SweepConstraintControls& controls=SweepConstraintControls{},
    const Tolerances& tolerances=Tolerances{},bool* numericalFailure=nullptr);








bool moveCoupledMeshesConstrained(const HostState&, const std::vector<FilmAux>&,
    const std::vector<Real>& solidTargetSweeps, std::size_t materialCellCount, Real interval,
    HostMesh& gas, HostMesh& solid, SurfaceMesh&, SweepConstraintReport&,
    std::string& error, const SweepConstraintControls& controls=SweepConstraintControls{},
    const Tolerances& tolerances=Tolerances{},bool* numericalFailure=nullptr);

bool moveCoupledMeshesConstrained(const HostState&, const std::vector<FilmAux>&,
    const std::vector<Real>& solidTargetSweeps, Real interval,
    HostMesh& gas, HostMesh& solid, SurfaceMesh&, SweepConstraintReport&,
    std::string& error, const SweepConstraintControls& controls=SweepConstraintControls{},
    const Tolerances& tolerances=Tolerances{},bool* numericalFailure=nullptr);
}
#endif
