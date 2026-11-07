#pragma once
#include "gpu/SharedGasDeviceStorage.cuh"
#include "gpu/ResidentWallProgram.H"
#include "mesh/GeometryGpu.H"
#include "mesh/EmptyExtrusion.H"
#include "../../../common/gpu/ResidentStageGeometry.H"
namespace chmt {
namespace resident_mesh_detail {
using Point=ugkp::gpu::geometry::Point;
__global__ void samplePointProgram(int count,const Point* current,const Point* knots,
    const ResidentWallKnotTime* time,int knotCount,const ResidentWallClock* clock,
    double interval,Point* previous,Point* endpoint,ugkp::gpu::geometry::StageGeometryControls* controls){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i==0)controls->interval=interval;
    if(i>=count)return;
    auto stamp=clock->elapsed;stamp.add(interval);
    const auto position=locateResidentGasWallProgram(time,knotCount,clock->origin,stamp);
    const auto a=knots[std::size_t(position.lower)*count+i];previous[i]=current[i];
    if(position.lower==position.upper){endpoint[i]=a;return;}
    const auto b=knots[std::size_t(position.upper)*count+i];
    endpoint[i]={interpolateGasWallValue(a.x,b.x,position.alpha),interpolateGasWallValue(a.y,b.y,position.alpha),interpolateGasWallValue(a.z,b.z,position.alpha)};
}
__global__ void acceptPointTrajectory(int count,const Point* endpoint,Point* current,
    const ugkp::gpu::geometry::StageGeometryStatus* status,ugkp::diagnostics::DeviceContext* diagnostics){
    if(!ugkp::diagnostics::blockMayProceed(diagnostics)||status->failure!=ugkp::gpu::geometry::StageGeometryFailure::None)return;
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)current[i]=endpoint[i];
}
}
// CHMT mesh/state adapter. All geometry mathematics belongs to the shared
// resident stage owner. Accepted coordinates, candidate coordinates and the
// previous accepted stage have distinct stable addresses for graph consumers.
class ResidentMeshTrajectory {
    using Point=ugkp::gpu::geometry::Point;
    using Stage=ugkp::gpu::geometry::ResidentStageGeometry;
    CudaFault fault_;
    Buffer<Point> current_,previous_,endpoint_,program_;
    Buffer<ugkp::gpu::geometry::StageGeometryControls> controls_;
    std::unique_ptr<Stage> stage_;
    int points_=0;bool empty_=false;
    std::uint64_t topologyHash_=0;
    struct Publication {
        std::vector<Point> previous,endpoint,velocity;
        std::vector<ugkp::gpu::geometry::CellResult> oldCells,cells;
        std::vector<ugkp::gpu::geometry::FaceResult> oldFaces,faces;
        std::vector<ugkp::gpu::geometry::WallDistanceResult> distance;
        std::vector<double> sweep;
        std::vector<ugkp::gpu::geometry::StageGeometryControls> controls;
    } publication_;
public:
    bool initialize(const HostMesh& mesh,const Tolerances& tolerance,SharedGasDeviceStorage& storage,std::string& error){
        const auto definition=geometry_device::meshTopology(mesh);
        empty_detail::Extrusion extrusion;if(!empty_detail::build(mesh,extrusion,error))return false;
        points_=mesh.points.size();empty_=extrusion.axis>=0;topologyHash_=mesh.topologyHash;
        stage_.reset(new Stage(definition.topology,points_,definition.walls,mesh.periodicPartner,extrusion.axis,extrusion.mate));
        if(!current_.resize(points_,fault_)||!previous_.resize(points_,fault_)||!endpoint_.resize(points_,fault_)||!controls_.resize(1,fault_)){
            error=fault_.message;return false;}
        const ugkp::gpu::geometry::StageGeometryControls controls{0,tolerance.absoluteGeometry,tolerance.relativeGeometry};
        if(!storage.write(current_.data(),geometry_device::pack(mesh.points))
            ||!storage.write(controls_.data(),std::vector<ugkp::gpu::geometry::StageGeometryControls>{controls})){
            error=storage.error();return false;}
        return true;
    }
    // Transitional publication boundary of the existing material producer.
    // Microsteps consume these arrays directly; no point samples are marshaled
    // by the host. A resident material producer can supply the same knot layout.
    bool installProgram(const WallProgram& program,bool gas,SharedGasDeviceStorage& storage,std::string& error){
        std::vector<Point> points;
        for(const auto& knot:program.knots){const auto& values=gas?knot.gasPoints:knot.solidPoints;
            if(values.size()!=std::size_t(points_)){error="resident point program topology mismatch";return false;}
            for(const auto& p:values)points.push_back({p.x,p.y,p.z});}
        if(!program_.resize(points.size(),fault_)||!storage.write(program_.data(),points)){
            error=fault_.message.empty()?storage.error():fault_.message;return false;}
        return true;
    }
    bool importAccepted(const HostMesh& mesh,SharedGasDeviceStorage& storage,std::string& error){
        if(mesh.topologyHash!=topologyHash_||mesh.points.size()!=std::size_t(points_)){
            error="resident trajectory requires the initialized mesh topology";return false;}
        if(!storage.write(current_.data(),geometry_device::pack(mesh.points))){error=storage.error();return false;}
        return true;
    }
    bool prepare(double interval,const ResidentWallKnotTime* times,int knots,const ResidentWallClock* clock,
        ugkp::diagnostics::DeviceContext* diagnostics,cudaStream_t stream,std::string& error){
        resident_mesh_detail::samplePointProgram<<<points_?(points_+127)/128:1,128,0,stream>>>(
            points_,current_.data(),program_.data(),times,knots,clock,interval,previous_.data(),endpoint_.data(),controls_.data());
        if(!fault_.check(cudaGetLastError(),"sample resident mesh point program")){error=fault_.message;return false;}
        stage_->prepare(previous_.data(),endpoint_.data(),controls_.data(),diagnostics,stream);return true;
    }
    bool accept(ugkp::diagnostics::DeviceContext* diagnostics,cudaStream_t stream,std::string& error){
        if(points_)resident_mesh_detail::acceptPointTrajectory<<<(points_+127)/128,128,0,stream>>>(points_,endpoint_.data(),current_.data(),view().status,diagnostics);
        if(!fault_.check(cudaGetLastError(),"accept resident point trajectory")){error=fault_.message;return false;}return true;
    }
    // Publication only: queue all reads on the owner's existing output stream.
    // Complete them together with gas-field reads before formatting host state.
    bool queuePublication(SharedGasDeviceStorage& storage,std::string& error){
        const auto stage=view();auto& p=publication_;
        if(!storage.enqueueRead(previous_.data(),points_,p.previous)||!storage.enqueueRead(endpoint_.data(),points_,p.endpoint)
            ||!storage.enqueueRead(stage.oldGeometry.cell,stage.oldGeometry.cells,p.oldCells)
            ||!storage.enqueueRead(stage.endpointGeometry.cell,stage.endpointGeometry.cells,p.cells)
            ||!storage.enqueueRead(stage.oldGeometry.face,stage.oldGeometry.faces,p.oldFaces)
            ||!storage.enqueueRead(stage.endpointGeometry.face,stage.endpointGeometry.faces,p.faces)
            ||!storage.enqueueRead(stage.endpointGeometry.distance,stage.endpointGeometry.cells,p.distance)
            ||!storage.enqueueRead(stage.meshVelocity,stage.endpointGeometry.faces,p.velocity)
            ||!storage.enqueueRead(stage.sweptVolume,stage.endpointGeometry.faces,p.sweep)
            ||!storage.enqueueRead(controls_.data(),1,p.controls)){error=storage.error();return false;}
        return true;
    }
    // Called only after the queued publication and existing diagnostics finish.
    // This formats device results; it performs no geometric recomputation.
    void publish(HostMesh& mesh,HostStageGeometry& stage,std::uint64_t version)const{
        const auto& p=publication_;const auto cells=p.cells.size(),faces=p.faces.size();
        geometry_device::unpack(p.previous,mesh.oldPoints);geometry_device::unpack(p.endpoint,mesh.points);
        geometry_device::unpack(p.velocity,mesh.meshVelocity);
        mesh.oldVolumes.resize(cells);mesh.volumes.resize(cells);mesh.cellCentres.resize(cells);mesh.wallDistance.resize(cells);
        stage.cellCentre.resize(cells);stage.areaVector.resize(faces);stage.faceCentre.resize(faces);
        mesh.areaVectors.resize(faces);mesh.faceCentres.resize(faces);
        for(std::size_t c=0;c<cells;++c){mesh.oldVolumes[c]=p.oldCells[c].volume;mesh.volumes[c]=p.cells[c].volume;
            mesh.cellCentres[c]=geometry_device::unpack(p.cells[c].centre);stage.cellCentre[c]=geometry_device::unpack(p.oldCells[c].centre);mesh.wallDistance[c]=p.distance[c].distance;}
        for(std::size_t f=0;f<faces;++f){mesh.areaVectors[f]=geometry_device::unpack(p.faces[f].area);mesh.faceCentres[f]=geometry_device::unpack(p.faces[f].centre);
            stage.areaVector[f]=geometry_device::unpack(p.oldFaces[f].area);stage.faceCentre[f]=geometry_device::unpack(p.oldFaces[f].centre);}
        mesh.geometryVersion=version;stage.geometryVersion=version;stage.topologyHash=mesh.topologyHash;
        stage.interval=p.controls[0].interval;stage.oldVolume=mesh.oldVolumes;stage.newVolume=mesh.volumes;stage.evaluationVolume=mesh.oldVolumes;
        stage.oldPoints=mesh.oldPoints;stage.newPoints=mesh.points;stage.sweptVolume=p.sweep;
    }
    ugkp::gpu::geometry::StageGeometryDeviceView view()const{return stage_->view();}
    const ugkp::gpu::geometry::StageGeometryControls* controls()const{return controls_.data();}
    const Point* previous()const{return previous_.data();}
    const Point* endpoint()const{return endpoint_.data();}
    const Point* current()const{return current_.data();}
    int pointCount()const{return points_;}
    bool empty()const{return empty_;}
};
}
