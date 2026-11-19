#pragma once
#include "ablation/CpuSurfaceInterface.H"
#include "coupling/IntervalContracts.H"
#include "mesh/Geometry.H"
#include <algorithm>
#include <utility>
namespace chmt {
class MaterialStepReuse {
    bool previousImposedGeometryStationary_=false;
    WindowTime surfaceTime_{};
    Real surfaceDt_=0;
    CpuSurfaceResult surface_;
public:
    bool useStationaryGeometry(const HostState& state,const std::vector<Vec3>& gasPoints,
        const std::vector<Vec3>& solidPoints,HostState& trial,std::vector<Real>& gasSweep,std::vector<Real>& solidSweep){
        const bool stationary=identicalPoints(state.gasMesh.points,gasPoints)
            &&identicalPoints(state.solidMesh.points,solidPoints);
        if(!std::exchange(previousImposedGeometryStationary_,stationary)||!stationary)return false;
        ++trial.gasMesh.geometryVersion;
        ++trial.solidMesh.geometryVersion;
        gasSweep.assign(state.gasMesh.owner.size(),0);
        solidSweep.assign(state.solidMesh.owner.size(),0);
        return true;
    }

    bool takeSurface(WindowTime time,Real dt,const std::vector<Real>& sideVolume,CpuSurfaceResult& output){
        const Real previousDt=std::exchange(surfaceDt_,0);
        if(dt!=previousDt||time!=surfaceTime_
            ||std::any_of(sideVolume.begin(),sideVolume.end(),[](Real value){return value!=0;}))return false;
        output=std::move(surface_);
        return true;
    }

    void rememberSurface(WindowTime time,Real dt,const GasIntervalRecord& drive,CpuSurfaceResult&& result){
        if(result.wall.empty()||drive.gasAcceptedWall.size()!=result.wall.size())return;
        surfaceTime_=time;
        surfaceDt_=dt;
        surface_=std::move(result);
    }
};
}
