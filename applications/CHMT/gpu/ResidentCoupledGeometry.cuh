#pragma once
#include "gpu/ResidentMeshTrajectory.cuh"
#include "mesh/TrajectorySurface.H"
#include "../../../common/gpu/ResidentWallMatching.H"
#include "../../../common/gasWall/WallGeometryPublication.cuh"
#include "../../../common/operators/publishGasMeshGeometryKernel.cuh"
namespace chmt {
namespace coupled_geometry_detail {
using namespace ugkp::gpu::geometry;
// Candidate owners expose status without forcing a host decision. The consuming
// application records an invalid physical trajectory in its existing fatal latch.
__global__ void consumeStageStatus(const StageGeometryStatus* status,int region,
    ugkp::diagnostics::DeviceContext* diagnostics){
    if(threadIdx.x||blockIdx.x||!ugkp::diagnostics::blockMayProceed(diagnostics)
        ||status->failure==StageGeometryFailure::None)return;
    using namespace ugkp::diagnostics;
    recordFatal(diagnostics,makeRecord(Code::InvalidPhysicalState,3410,34,int(status->phase),
        EntityKind::None,status->index,region,double(int(status->failure)),0,0,Quantity::Conservation));
}
__global__ void consumeSurfaceStatus(const SurfaceGeometryStatus* status,ugkp::diagnostics::DeviceContext* diagnostics){
    if(threadIdx.x||blockIdx.x||!ugkp::diagnostics::blockMayProceed(diagnostics)
        ||status->failure==SurfaceGeometryFailure::None)return;
    using namespace ugkp::diagnostics;
    recordFatal(diagnostics,makeRecord(Code::InvalidPhysicalState,3411,34,0,
        EntityKind::Face,status->index,-1,double(int(status->failure)),0,0,Quantity::Conservation));
}
__global__ void consumeWallStatus(int count,const ugkwp::gaswall::WallMatchingStatus* status,
    ugkp::diagnostics::DeviceContext* diagnostics){
    if(threadIdx.x||blockIdx.x||!ugkp::diagnostics::blockMayProceed(diagnostics))return;
    if(status->invalidFace<0&&status->firstInvalidRow>=count)return;
    using namespace ugkp::diagnostics;
    recordFatal(diagnostics,makeRecord(Code::InvalidPhysicalState,3412,34,0,
        EntityKind::Face,status->invalidFace>=0?status->invalidFace:status->firstInvalidRow,-1,1,0,0,Quantity::Accuracy));
}
__global__ void publishSurfaceInput(int count,const int* gasFace,const SurfaceFaceGeometry* accepted,
    const SurfaceFaceGeometry* endpoint,const Point* support,const double* sweep,CoupledSurfaceInput* inputs,
    ugkp::diagnostics::DeviceContext* diagnostics){
    if(!ugkp::diagnostics::blockMayProceed(diagnostics))return;
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const auto& old=accepted[i];auto& out=inputs[i];
    out.area=old.area;out.gasDistance=old.gasDistance;out.normal={old.normal.x,old.normal.y,old.normal.z};
    out.sweptVolume=-sweep[gasFace[i]];
    const Vec3 base{support[i].x,support[i].y,support[i].z};
    const auto velocity=endpoint[i].velocity;const Vec3 motion{velocity.x,velocity.y,velocity.z};
    out.particleMaterialVelocity=base+out.normal*dot(motion-base,out.normal);
}
__global__ void acceptSurface(int count,const SurfaceFaceGeometry* endpoint,SurfaceFaceGeometry* accepted,
    ugkp::diagnostics::DeviceContext* diagnostics){
    if(!ugkp::diagnostics::blockMayProceed(diagnostics))return;
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)accepted[i]=endpoint[i];
}
}
// Application adapter only: topology, ownership and CHMT's private interface
// identity are supplied here. All mesh, surface and matching mathematics is
// delegated to the shared resident owners.
class ResidentCoupledGeometry {
    using Point=ugkp::gpu::geometry::Point;
    using Face=ugkp::gpu::geometry::SurfaceFaceGeometry;
    CudaFault fault_;
    std::unique_ptr<ugkp::gpu::geometry::ResidentSurfaceSeams> seams_;
    std::unique_ptr<ugkp::gpu::geometry::ResidentSurfaceGeometry> surface_;
    std::unique_ptr<ugkp::gpu::geometry::ResidentWallMatching> wall_;
    Buffer<Face> acceptedSurface_;
    Buffer<Point> support_;
    std::vector<int> solidFaceSlot_;
    std::vector<Face> publishedFaces_;
    std::vector<ugkp::gpu::geometry::EdgeResult> publishedEdges_;
    std::vector<ugkp::gpu::geometry::SurfaceEdgeMap> publishedMaps_;
    std::vector<ugkp::gpu::geometry::SurfaceGeometryStatus> publishedStatus_;
    bool dirty_=false;
    double absoluteTolerance_=0,relativeTolerance_=0;
public:
    ResidentMeshTrajectory gas,solid;
    bool initialize(const HostState& state,const Tolerances& tolerance,SharedGasDeviceStorage& storage,std::string& error){
        using namespace ugkp::gpu::geometry;
        absoluteTolerance_=tolerance.absoluteGeometry;relativeTolerance_=tolerance.relativeGeometry;
        if(!gas.initialize(state.gasMesh,tolerance,storage,error)||!solid.initialize(state.solidMesh,tolerance,storage,error))return false;
        std::vector<SurfaceTopologyEdge> edges;std::vector<SurfaceSeamPatch> patches;
        if(!buildSurfaceSeamTopology(state.solidMesh,state.surface,edges,patches,error))return false;
        seams_.reset(new ResidentSurfaceSeams(state.solidMesh.points.size(),state.solidMesh.owner.size(),state.surface.solidFace,edges,patches));
        std::vector<SurfaceFaceMap> faces;solidFaceSlot_.assign(state.solidMesh.owner.size(),-1);
        for(std::size_t i=0;i<state.surface.area.size();++i){const int s=state.surface.solidFace[i],g=state.surface.gasFace[i];
            faces.push_back({s,g,state.solidMesh.owner[s],g>=0?state.gasMesh.owner[g]:-1});solidFaceSlot_[s]=i;}
        surface_.reset(new ResidentSurfaceGeometry(faces,seams_->view().capacity,state.solidMesh.points.size(),solid.empty()));
        if(storage.hostView().gasBoundaryLayer.enabled){
            WallMatchingDefinition d;const auto& m=state.gasMesh;d.cells=m.volumes.size();d.faceOffsets=m.faceOffsets;
            d.owner=m.owner;d.neighbour=m.neighbour;d.walls=state.surface.gasFace;d.physicalWall.resize(m.owner.size());
            for(std::size_t f=0;f<m.owner.size();++f)d.physicalWall[f]=m.boundaryKind[f]==BoundaryKind::NoSlip||m.boundaryKind[f]==BoundaryKind::Interface;
            wall_.reset(new ResidentWallMatching(d,gas.view().endpointGeometry.topology,false));
        }
        return true;
    }
    bool install(const HostState& state,const WallProgram& program,SharedGasDeviceStorage& storage,
        CoupledSurfaceInput* input,std::string& error){
        if(!gas.importAccepted(state.gasMesh,storage,error)||!solid.importAccepted(state.solidMesh,storage,error)
            ||!gas.installProgram(program,true,storage,error)||!solid.installProgram(program,false,storage,error))return false;
        const auto& s=state.surface;std::vector<Face> faces(s.area.size());std::vector<Point> support(s.area.size());
        std::vector<CoupledSurfaceInput> inputs(s.area.size());
        for(std::size_t i=0;i<faces.size();++i){auto& f=faces[i];f.area=s.area[i];f.gasDistance=s.gasDistance[i];
            f.normal={s.normal[i].x,s.normal[i].y,s.normal[i].z};
            if(!s.baseVelocity.empty())support[i]={s.baseVelocity[i].x,s.baseVelocity[i].y,s.baseVelocity[i].z};
            auto& id=inputs[i].identity;id.face=s.persistentId[i];id.stage=1;id.gasCell=state.gasMesh.owner[s.gasFace[i]];
            id.solidCell=s.solidCell[i];id.filmFace=i;}
        if(!acceptedSurface_.resize(faces.size(),fault_)||!support_.resize(support.size(),fault_)
            ||!storage.write(acceptedSurface_.data(),faces)||!storage.write(support_.data(),support)||!storage.write(input,inputs)){
            error=fault_.message.empty()?storage.error():fault_.message;return false;}
        dirty_=false;return true;
    }
    bool prepare(double interval,const ResidentWallKnotTime* times,int knots,const ResidentWallClock* clock,
        const int* gasFaces,CoupledSurfaceInput* input,SharedGasDeviceStorage& storage,cudaStream_t stream,std::string& error){
        using namespace coupled_geometry_detail;
        auto* diagnostics=storage.hostView().diagnostics;
        if(!gas.prepare(interval,times,knots,clock,diagnostics,stream,error))return false;
        consumeStageStatus<<<1,1,0,stream>>>(gas.view().status,0,diagnostics);
        if(!solid.prepare(interval,times,knots,clock,diagnostics,stream,error))return false;
        consumeStageStatus<<<1,1,0,stream>>>(solid.view().status,1,diagnostics);
        seams_->prepare(solid.previous(),solid.view().oldGeometry.face,stream);
        surface_->prepareWithSeams(solid.view().oldGeometry,solid.view().endpointGeometry,gas.view().endpointGeometry,
            solid.previous(),solid.endpoint(),seams_->view(),&gas.controls()->interval,diagnostics,stream);
        consumeSurfaceStatus<<<1,1,0,stream>>>(surface_->view().status,diagnostics);
        const int count=surface_->view().faces;
        if(count)publishSurfaceInput<<<(count+127)/128,128,0,stream>>>(count,gasFaces,acceptedSurface_.data(),surface_->view().face,
            support_.data(),gas.view().sweptVolume,input,diagnostics);
        auto& g=storage.hostView().gasGeometry;g.enabled=true;g.oldVolume=gas.view().oldVolume;
        g.newVolume=gas.view().newVolume;g.faceSweptVolume=gas.view().sweptVolume;g.interval=interval;
        g.absoluteGeometryTolerance=absoluteTolerance_;g.relativeGeometryTolerance=relativeTolerance_;
        ugkwp::bindGasStageGeometryKernel<<<1,1,0,stream>>>(storage.deviceView(),gas.view(),gas.controls());
        if(!fault_.check(cudaGetLastError(),"prepare resident coupled geometry")){error=fault_.message;return false;}return true;
    }
    ugkp::thermal::ParticleSurfaceGeometry particleGeometry(const SharedGasDeviceStorage& storage,bool endpoint)const{
        const auto mesh=gas.view().endpointGeometry;
        return {mesh.faces,gas.pointCount(),mesh.topology.faceOffsets,mesh.topology.facePoints,storage.hostView().faceOwner,
            endpoint?gas.endpoint():gas.previous()};
    }
    bool finish(SharedGasDeviceStorage& storage,cudaStream_t stream,std::string& error){
        using namespace coupled_geometry_detail;auto* state=storage.deviceView();auto& host=storage.hostView();auto* diagnostics=host.diagnostics;
        const auto mesh=gas.view().endpointGeometry;
        if(wall_){wall_->prepare(gas.endpoint(),mesh.cell,nullptr,nullptr,stream);const auto w=wall_->view();
            consumeWallStatus<<<1,1,0,stream>>>(w.rows,w.status,diagnostics);
            if(w.rows){auto& layer=host.gasBoundaryLayerModel;
                ugkwp::gaswall::publishCellAverageWallGeometryKernel<<<(w.rows+127)/128,128,0,stream>>>(w.rows,w.geometry,w.status,
                    layer.input,const_cast<int*>(layer.matchingCells),const_cast<Real*>(layer.matchingWeights),diagnostics);
                ugkwp::gaswall::resetWallGeometryFramesKernel<<<(w.rows+127)/128,128,0,stream>>>(state,w.status);}}
        if(mesh.cells)ugkwp::publishGasCellGeometryKernel<<<(mesh.cells+127)/128,128,0,stream>>>(state,mesh,gas.view().status);
        if(mesh.faces)ugkwp::publishGasFaceGeometryKernel<<<(mesh.faces+127)/128,128,0,stream>>>(state,mesh,gas.view().status);
        const int count=surface_->view().faces;
        if(count)acceptSurface<<<(count+127)/128,128,0,stream>>>(count,surface_->view().face,acceptedSurface_.data(),diagnostics);
        if(!gas.accept(diagnostics,stream,error)||!solid.accept(diagnostics,stream,error))return false;
        if(!fault_.check(cudaGetLastError(),"publish resident coupled geometry")){error=fault_.message;return false;}
        dirty_=true;return true;
    }
    bool dirty()const{return dirty_;}
    bool queuePublication(SharedGasDeviceStorage& storage,std::string& error){
        if(!dirty_)return true;
        const auto surface=surface_->view();const auto seams=seams_->view();
        if(!gas.queuePublication(storage,error)||!solid.queuePublication(storage,error))return false;
        if(!storage.enqueueRead(surface.face,surface.faces,publishedFaces_)||!storage.enqueueRead(surface.status,1,publishedStatus_)
            ||!storage.enqueueRead(surface.edge,surface.edges,publishedEdges_)||!storage.enqueueRead(seams.edges,seams.capacity,publishedMaps_)){
            error=storage.error();return false;}return true;
    }
    void publish(HostState& state){
        if(!dirty_)return;
        gas.publish(state.gasMesh,state.gasStages[0],state.gasMesh.geometryVersion);state.gasStages[1]=state.gasStages[0];
        solid.publish(state.solidMesh,state.solidStages[0],state.solidMesh.geometryVersion);state.solidStages[1]=state.solidStages[0];
        auto& s=state.surface;const auto count=publishedFaces_.size();
        s.oldArea.resize(count);s.area.resize(count);s.centre.resize(count);s.normal.resize(count);
        s.meshVelocity.resize(count);s.baseVelocity.resize(count);s.gasDistance.resize(count);s.solidDistance.resize(count);
        for(std::size_t i=0;i<count;++i){const auto& f=publishedFaces_[i];s.oldArea[i]=f.oldArea;s.area[i]=f.area;
            s.centre[i]=geometry_device::unpack(f.centre);s.normal[i]=geometry_device::unpack(f.normal);s.meshVelocity[i]=geometry_device::unpack(f.velocity);
            s.gasDistance[i]=f.gasDistance;s.solidDistance[i]=f.solidDistance;}
        const auto edges=publishedStatus_[0].edgeCount;s.edgeOwner.resize(edges);s.edgeNeighbour.resize(edges);s.edgeLength.resize(edges);
        s.edgeConormal.resize(edges);s.sweptEdgeArea.resize(edges);s.edgeOwnerOffset.resize(edges);s.edgeNeighbourOffset.resize(edges);
        for(int i=0;i<edges;++i){const auto& m=publishedMaps_[i];const auto& e=publishedEdges_[i];s.edgeOwner[i]=m.ownerSurface;
            s.edgeNeighbour[i]=m.geometry.neighbourFace<0?-1:solidFaceSlot_[m.geometry.neighbourFace];s.edgeLength[i]=e.length;
            s.edgeConormal[i]=geometry_device::unpack(e.conormal);s.sweptEdgeArea[i]=e.sweep;
            s.edgeOwnerOffset[i]=geometry_device::unpack(e.ownerOffset);s.edgeNeighbourOffset[i]=geometry_device::unpack(e.neighbourOffset);}
        dirty_=false;
    }
};
}
