// Included inside ugkp::gpu::geometry after the shared allocation utilities.
namespace {
struct PeriodicGeometryFace {int face,partner,firstMatch;};
struct PeriodicGeometryVertex {int vertex,face,partner;};
struct GeometryBounds {Point low,high;};
__global__ void resetGeometryBoundary(GeometryBoundaryStatus* out){
    if(!threadIdx.x&&!blockIdx.x){*out={};out->first=~0ull;}
}
__device__ void boundaryFailure(GeometryBoundaryStatus* out,int index,GeometryBoundaryFailure kind){
    atomicMin(&out->first,static_cast<unsigned long long>(index)*8+static_cast<unsigned int>(kind));
}
__global__ void finishGeometryBoundary(GeometryBoundaryStatus* out){
    if(!threadIdx.x&&!blockIdx.x&&out->first!=~0ull){
        out->failure=static_cast<GeometryBoundaryFailure>(out->first%8);
        out->index=out->failure==GeometryBoundaryFailure::EmptyDuplicate?-1:int(out->first/8);
    }
}
__device__ GeometryBounds mergeBounds(GeometryBounds a,GeometryBounds b){
    a.low.x=a.low.x<b.low.x?a.low.x:b.low.x;a.low.y=a.low.y<b.low.y?a.low.y:b.low.y;a.low.z=a.low.z<b.low.z?a.low.z:b.low.z;
    a.high.x=a.high.x>b.high.x?a.high.x:b.high.x;a.high.y=a.high.y>b.high.y?a.high.y:b.high.y;a.high.z=a.high.z>b.high.z?a.high.z:b.high.z;
    return a;
}
__global__ void geometryBounds(int count,const Point* points,GeometryBounds* partial){
    __shared__ GeometryBounds values[128];const int p=blockIdx.x*blockDim.x+threadIdx.x;
    const Point q=points[p<count?p:count-1];values[threadIdx.x]={q,q};__syncthreads();
    for(int half=64;half;half/=2){if(threadIdx.x<half)values[threadIdx.x]=mergeBounds(values[threadIdx.x],values[threadIdx.x+half]);__syncthreads();}
    if(!threadIdx.x)partial[blockIdx.x]=values[0];
}
__global__ void geometryScale(int count,const GeometryBounds* partial,GeometryBoundaryStatus* out){
    __shared__ GeometryBounds values[128];GeometryBounds value=partial[0];
    for(int i=threadIdx.x;i<count;i+=blockDim.x)value=mergeBounds(value,partial[i]);
    values[threadIdx.x]=value;__syncthreads();
    for(int half=64;half;half/=2){if(threadIdx.x<half)values[threadIdx.x]=mergeBounds(values[threadIdx.x],values[threadIdx.x+half]);__syncthreads();}
    if(!threadIdx.x)out->scale=magnitude(values[0].high-values[0].low);
}
__global__ void matchPeriodicVertices(int count,const PeriodicGeometryVertex* vertices,TopologyView topology,
    const Point* points,const FaceResult* faces,const GeometryBoundaryStatus* status,int* matches){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const auto v=vertices[i];const Point shift=faces[v.partner].centre-faces[v.face].centre;
    matches[i]=periodicVertexMatch(v.vertex,v.partner,topology,points,shift,1e-10*status->scale);
}
__global__ void checkPeriodicFaces(int count,const PeriodicGeometryFace* periodic,TopologyView topology,
    const FaceResult* faces,const int* matches,GeometryBoundaryStatus* status){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const auto p=periodic[i];
    if(!periodicAreaMatches(faces[p.face].area,faces[p.partner].area)){
        boundaryFailure(status,p.face,GeometryBoundaryFailure::PeriodicArea);return;
    }
    const int vertices=topology.faceOffsets[p.face+1]-topology.faceOffsets[p.face];
    for(int j=0;j<vertices;++j){const int value=matches[p.firstMatch+j];
        if(value<0){boundaryFailure(status,p.face,GeometryBoundaryFailure::PeriodicVertices);return;}
        for(int k=0;k<j;++k)if(matches[p.firstMatch+k]==value){
            boundaryFailure(status,p.face,GeometryBoundaryFailure::PeriodicVertices);return;
        }
    }
}
__global__ void checkEmptyTrajectory(int count,int axis,const int* mate,const Point* old,const Point* endpoint,GeometryBoundaryStatus* status){
    const int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=count)return;
    const auto failure=emptyTrajectoryPoint(p,mate[p],axis,old,endpoint);
    if(failure!=EmptyTrajectoryFailure::None)boundaryFailure(status,p,
        failure==EmptyTrajectoryFailure::NormalDisplacement?GeometryBoundaryFailure::EmptyNormal:GeometryBoundaryFailure::EmptyTangential);
}
__global__ void emptyCoordinateKeys(int count,int axis,const Point* points,EmptyCoordinateKey* keys){
    const int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=count)return;
    const auto q=points[p];
    // Coordinate validity is diagnosed by the mesh owner before this policy.
    // Keep sorting's comparison relation well-defined even in a rejected stage.
    keys[p]=finite(q)?EmptyCoordinateKey{boundaryDetail::coordinate(q,(axis+1)%3),
        boundaryDetail::coordinate(q,(axis+2)%3),boundaryDetail::coordinate(q,axis)}:EmptyCoordinateKey{};
}
__global__ void duplicateEmptyCoordinates(int count,const EmptyCoordinateKey* keys,GeometryBoundaryStatus* status){
    const int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>0&&p<count&&sameEmptyCoordinate(keys[p-1],keys[p]))boundaryFailure(status,0,GeometryBoundaryFailure::EmptyDuplicate);
}
}
struct ResidentGeometryBoundary::Implementation {
    TopologyView topology{};int points=0,axis=-1,faces=0,vertices=0;
    Array<PeriodicGeometryFace> periodicFaces;Array<PeriodicGeometryVertex> periodicVertices;
    Array<int> matches,mate;Array<GeometryBounds> bounds;Array<EmptyCoordinateKey> keys;
    Array<unsigned char> sortScratch;std::size_t sortBytes=0;
    Array<GeometryBoundaryStatus> periodicResult,trajectoryResult,uniqueResult;
};
ResidentGeometryBoundary::ResidentGeometryBoundary(const Topology& topology,TopologyView deviceTopology,int pointCount,
    const std::vector<int>& partner,int axis,const std::vector<int>& mate):data_(new Implementation){
    auto& d=*data_;d.topology=deviceTopology;d.points=pointCount;d.axis=axis;
    const int faces=topology.faceOffsets.empty()?0:int(topology.faceOffsets.size()-1);
    if(pointCount<0||partner.size()!=std::size_t(faces)||axis<-1||axis>2||(axis>=0&&mate.size()!=std::size_t(pointCount)))
        throw std::invalid_argument("geometry boundary topology shape mismatch");
    std::vector<PeriodicGeometryFace> periodicFaces;std::vector<PeriodicGeometryVertex> vertices;
    for(int f=0;f<faces;++f)if(partner[f]>=0){
        periodicFaces.push_back({f,partner[f],int(vertices.size())});
        for(int k=topology.faceOffsets[f];k<topology.faceOffsets[f+1];++k)vertices.push_back({topology.facePoints[k],f,partner[f]});
    }
    const auto upload=[](auto& target,const auto& values){
        target.reserve(values.size());if(!values.empty())checked(cudaMemcpy(target.pointer,values.data(),values.size()*sizeof(values[0]),cudaMemcpyHostToDevice));
    };
    upload(d.periodicFaces,periodicFaces);upload(d.periodicVertices,vertices);if(axis>=0)upload(d.mate,mate);
    d.faces=periodicFaces.size();d.vertices=vertices.size();d.matches.reserve(d.vertices);
    if(d.faces)d.bounds.reserve((d.points+127)/128);
    d.periodicResult.reserve(1);d.trajectoryResult.reserve(1);d.uniqueResult.reserve(1);
    if(axis>=0&&d.points){d.keys.reserve(d.points);
        checked(cub::DeviceMergeSort::SortKeys(nullptr,d.sortBytes,d.keys.pointer,d.points,EmptyCoordinateLess{}));
        d.sortScratch.reserve(d.sortBytes);
    }
}
ResidentGeometryBoundary::~ResidentGeometryBoundary()=default;
const GeometryBoundaryStatus* ResidentGeometryBoundary::periodicStatus()const{return data_->periodicResult.pointer;}
const GeometryBoundaryStatus* ResidentGeometryBoundary::emptyTrajectoryStatus()const{return data_->trajectoryResult.pointer;}
const GeometryBoundaryStatus* ResidentGeometryBoundary::emptyUniquenessStatus()const{return data_->uniqueResult.pointer;}
void ResidentGeometryBoundary::periodic(const Point* points,const FaceResult* faces,void* streamHandle){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(streamHandle);
    resetGeometryBoundary<<<1,1,0,stream>>>(d.periodicResult.pointer);checked(cudaGetLastError());
    if(d.faces){
        geometryBounds<<<(d.points+127)/128,128,0,stream>>>(d.points,points,d.bounds.pointer);checked(cudaGetLastError());
        geometryScale<<<1,128,0,stream>>>((d.points+127)/128,d.bounds.pointer,d.periodicResult.pointer);checked(cudaGetLastError());
        matchPeriodicVertices<<<(d.vertices+127)/128,128,0,stream>>>(d.vertices,d.periodicVertices.pointer,d.topology,
            points,faces,d.periodicResult.pointer,d.matches.pointer);checked(cudaGetLastError());
        checkPeriodicFaces<<<(d.faces+127)/128,128,0,stream>>>(d.faces,d.periodicFaces.pointer,d.topology,faces,d.matches.pointer,d.periodicResult.pointer);checked(cudaGetLastError());
    }
    finishGeometryBoundary<<<1,1,0,stream>>>(d.periodicResult.pointer);checked(cudaGetLastError());
}
void ResidentGeometryBoundary::emptyTrajectory(const Point* old,const Point* endpoint,void* streamHandle){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(streamHandle);
    resetGeometryBoundary<<<1,1,0,stream>>>(d.trajectoryResult.pointer);checked(cudaGetLastError());
    if(d.axis>=0&&d.points){checkEmptyTrajectory<<<(d.points+127)/128,128,0,stream>>>(d.points,d.axis,d.mate.pointer,old,endpoint,d.trajectoryResult.pointer);checked(cudaGetLastError());}
    finishGeometryBoundary<<<1,1,0,stream>>>(d.trajectoryResult.pointer);checked(cudaGetLastError());
}
void ResidentGeometryBoundary::emptyUniqueness(const Point* points,void* streamHandle){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(streamHandle);
    resetGeometryBoundary<<<1,1,0,stream>>>(d.uniqueResult.pointer);checked(cudaGetLastError());
    if(d.axis>=0&&d.points){
        emptyCoordinateKeys<<<(d.points+127)/128,128,0,stream>>>(d.points,d.axis,points,d.keys.pointer);checked(cudaGetLastError());
        checked(cub::DeviceMergeSort::SortKeys(d.sortScratch.pointer,d.sortBytes,d.keys.pointer,d.points,EmptyCoordinateLess{},stream));
        duplicateEmptyCoordinates<<<(d.points+127)/128,128,0,stream>>>(d.points,d.keys.pointer,d.uniqueResult.pointer);checked(cudaGetLastError());
    }
    finishGeometryBoundary<<<1,1,0,stream>>>(d.uniqueResult.pointer);checked(cudaGetLastError());
}
