// Included inside ugkp::gpu::geometry after common geometry/bounds operators.
namespace {
__global__ void resetSurfaceGeometry(const double* interval,int edges,const SurfaceSeamStatus* seams,SurfaceGeometryStatus* status){
    if(threadIdx.x||blockIdx.x)return;*status={};status->first=~0ull;status->edgeCount=seams?seams->count:edges;status->firstWarning=status->edgeCount;
    if(!(*interval>0)||!std::isfinite(*interval))status->failure=SurfaceGeometryFailure::InvalidInterval;
}
__global__ void surfaceFaceMetrics(int count,const SurfaceFaceMap* map,PolyhedralGeometryDeviceView oldSolid,
    PolyhedralGeometryDeviceView solid,PolyhedralGeometryDeviceView gas,const double* interval,
    SurfaceFaceGeometry* output,SurfaceGeometryStatus* status){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count||status->failure==SurfaceGeometryFailure::InvalidInterval)return;
    const auto m=map[i];const auto old=oldSolid.face[m.solidFace],end=solid.face[m.solidFace];
    const auto other=m.gasFace>=0?gas.face[m.gasFace]:FaceResult{};
    const auto cell=m.gasFace>=0?gas.cell[m.gasCell].centre:Point{};
    const auto result=surfaceFaceGeometry(old.area,old.centre,end.area,end.centre,solid.cell[m.solidCell].centre,
        m.gasFace>=0,other.area,other.centre,cell,*interval);output[i]=result;
    if(!(result.solidDistance>0)||(m.gasFace>=0&&!(result.gasDistance>0)))atomicMin(&status->first,static_cast<unsigned long long>(i));
}
__global__ void publishSurfaceSeamFailure(const SurfaceSeamStatus* seams,SurfaceGeometryStatus* status){
    if(threadIdx.x||blockIdx.x||!seams||status->failure!=SurfaceGeometryFailure::None||status->first!=~0ull)return;
    if(seams->failure!=SurfaceSeamFailure::None){status->index=seams->index;
        status->failure=seams->failure==SurfaceSeamFailure::Ambiguous?SurfaceGeometryFailure::AmbiguousSeam:SurfaceGeometryFailure::UnpairedSeam;}
}
__global__ void surfaceEdgeMetrics(int count,TopologyView topology,const Point* old,const Point* endpoint,
    const SurfaceEdgeMap* edges,bool empty,EdgeResult* output,SurfaceGeometryStatus* status){
    const int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=status->edgeCount||status->failure!=SurfaceGeometryFailure::None)return;
    const auto result=surfaceEdge(topology,old,endpoint,edges[e].geometry,empty,0,status->scale);output[e]=result;
    if(result.invalid)atomicMin(&status->first,(1ull<<32)|static_cast<unsigned int>(e));
    else if(!result.converged)atomicMin(&status->firstWarning,e);
}
__device__ int surfaceWarningCutoff(int edges,const SurfaceGeometryStatus& status){
    if(status.first==~0ull)return edges-1;
    return status.first>>32?int(static_cast<unsigned int>(status.first)):-1;
}
__device__ void recordSurfaceEdgeWarning(int edge,const SurfaceEdgeMap* maps,const EdgeResult* results,ugkp::diagnostics::DeviceContext* diagnostics){
    using namespace ugkp::diagnostics;const auto r=results[edge];
    recordWarning(diagnostics,makeRecord(Code::AccuracyResidual,3402,34,0,EntityKind::Face,maps[edge].ownerSurface,-1,
        r.residual,r.tolerance,r.residual,Quantity::Accuracy));
}
__global__ void publishSurfaceFirstWarning(int count,const SurfaceEdgeMap* map,const EdgeResult* results,
    const SurfaceGeometryStatus* status,ugkp::diagnostics::DeviceContext* diagnostics){
    if(threadIdx.x||blockIdx.x||!diagnostics||status->failure!=SurfaceGeometryFailure::None)return;
    count=status->edgeCount;
    const int first=status->firstWarning;if(first<count&&first<=surfaceWarningCutoff(count,*status))recordSurfaceEdgeWarning(first,map,results,diagnostics);
}
__global__ void publishSurfaceOtherWarnings(int count,const SurfaceEdgeMap* map,const EdgeResult* results,
    const SurfaceGeometryStatus* status,ugkp::diagnostics::DeviceContext* diagnostics){
    const int e=blockIdx.x*blockDim.x+threadIdx.x;count=status->edgeCount;
    if(!diagnostics||status->failure!=SurfaceGeometryFailure::None||e>=count||e==status->firstWarning||e>surfaceWarningCutoff(count,*status))return;
    if(!results[e].invalid&&!results[e].converged)recordSurfaceEdgeWarning(e,map,results,diagnostics);
}
__global__ void finishSurfaceGeometry(const SurfaceEdgeMap* edges,const EdgeResult* results,SurfaceGeometryStatus* status){
    if(threadIdx.x||blockIdx.x||status->failure!=SurfaceGeometryFailure::None||status->first==~0ull)return;
    const int i=static_cast<unsigned int>(status->first);
    if(status->first>>32){status->index=edges[i].ownerSurface;
        status->failure=results[i].invalid==1?SurfaceGeometryFailure::DegenerateConormal:SurfaceGeometryFailure::DegenerateHinge;
    }else{status->index=i;status->failure=SurfaceGeometryFailure::InvalidNormalDistance;}
}
}
struct ResidentSurfaceGeometry::Implementation {
    int faces=0,edges=0,points=0;bool empty=false;
    Array<SurfaceFaceMap> mapping;Array<SurfaceFaceGeometry> face;Array<EdgeResult> edge;
    Array<GeometryBounds> bounds;Array<SurfaceGeometryStatus> status;
};
ResidentSurfaceGeometry::ResidentSurfaceGeometry(const std::vector<SurfaceFaceMap>& faces,int edges,int points,bool empty):data_(new Implementation){
    if(edges<0||points<0)throw std::invalid_argument("negative resident surface shape");
    auto& d=*data_;d.faces=faces.size();d.edges=edges;d.points=points;d.empty=empty;
    d.mapping.reserve(faces.size());if(!faces.empty())checked(cudaMemcpy(d.mapping.pointer,faces.data(),faces.size()*sizeof(SurfaceFaceMap),cudaMemcpyHostToDevice));
    d.face.reserve(faces.size());d.edge.reserve(edges);d.status.reserve(1);if(edges)d.bounds.reserve((points+127)/128);
}
ResidentSurfaceGeometry::~ResidentSurfaceGeometry()=default;
SurfaceGeometryDeviceView ResidentSurfaceGeometry::view()const{
    const auto& d=*data_;return {d.faces,d.edges,d.face.pointer,d.edge.pointer,d.status.pointer};
}
void ResidentSurfaceGeometry::prepare(PolyhedralGeometryDeviceView oldSolid,PolyhedralGeometryDeviceView solid,
    PolyhedralGeometryDeviceView gas,const Point* old,const Point* endpoint,const SurfaceEdgeMap* edges,const double* interval,
    ugkp::diagnostics::DeviceContext* diagnostics,void* stream){
    prepareImpl(oldSolid,solid,gas,old,endpoint,edges,interval,nullptr,diagnostics,stream);
}
void ResidentSurfaceGeometry::prepareWithSeams(PolyhedralGeometryDeviceView oldSolid,PolyhedralGeometryDeviceView solid,
    PolyhedralGeometryDeviceView gas,const Point* old,const Point* endpoint,SurfaceSeamDeviceView seams,const double* interval,
    ugkp::diagnostics::DeviceContext* diagnostics,void* stream){
    if(seams.capacity!=data_->edges||!seams.status)throw std::invalid_argument("resident surface seam allocation mismatch");
    prepareImpl(oldSolid,solid,gas,old,endpoint,seams.edges,interval,seams.status,diagnostics,stream);
}
void ResidentSurfaceGeometry::prepareImpl(PolyhedralGeometryDeviceView oldSolid,PolyhedralGeometryDeviceView solid,
    PolyhedralGeometryDeviceView gas,const Point* old,const Point* endpoint,const SurfaceEdgeMap* edges,const double* interval,
    const SurfaceSeamStatus* seams,ugkp::diagnostics::DeviceContext* diagnostics,void* streamHandle){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(streamHandle);
    resetSurfaceGeometry<<<1,1,0,stream>>>(interval,d.edges,seams,d.status.pointer);checked(cudaGetLastError());
    if(d.faces){surfaceFaceMetrics<<<(d.faces+127)/128,128,0,stream>>>(d.faces,d.mapping.pointer,oldSolid,solid,gas,interval,d.face.pointer,d.status.pointer);checked(cudaGetLastError());}
    if(seams){publishSurfaceSeamFailure<<<1,1,0,stream>>>(seams,d.status.pointer);checked(cudaGetLastError());}
    if(d.edges){
        geometryBounds<<<(d.points+127)/128,128,0,stream>>>(d.points,old,d.bounds.pointer);checked(cudaGetLastError());
        geometryScale<<<1,128,0,stream>>>((d.points+127)/128,d.bounds.pointer,d.status.pointer);checked(cudaGetLastError());
        surfaceEdgeMetrics<<<(d.edges+127)/128,128,0,stream>>>(d.edges,oldSolid.topology,old,endpoint,edges,d.empty,d.edge.pointer,d.status.pointer);checked(cudaGetLastError());
        publishSurfaceFirstWarning<<<1,1,0,stream>>>(d.edges,edges,d.edge.pointer,d.status.pointer,diagnostics);checked(cudaGetLastError());
        publishSurfaceOtherWarnings<<<(d.edges+127)/128,128,0,stream>>>(d.edges,edges,d.edge.pointer,d.status.pointer,diagnostics);checked(cudaGetLastError());
    }
    finishSurfaceGeometry<<<1,1,0,stream>>>(edges,d.edge.pointer,d.status.pointer);checked(cudaGetLastError());
}
