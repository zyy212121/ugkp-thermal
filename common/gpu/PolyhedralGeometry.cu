#include "PolyhedralGeometry.H"
#include "ExtendedGeometryMath.H"
#include "PinnedTransferQueue.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <climits>
#include <cstring>
#include <list>
#include <unordered_map>
#include <cstdint>
#include <stdexcept>
#include <string>
namespace ugkp { namespace gpu { namespace geometry { namespace {
void checked(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(std::string("CUDA polyhedral geometry: ")+cudaGetErrorString(status));}
template<class T> struct Array {
    T* pointer=nullptr;std::size_t capacity=0;
    ~Array(){if(pointer)cudaFree(pointer);}
    Array()=default;Array(const Array&)=delete;Array& operator=(const Array&)=delete;
    void reserve(std::size_t count){if(count<=capacity)return;T* next=nullptr;checked(cudaMalloc(reinterpret_cast<void**>(&next),count*sizeof(T)));
        if(pointer)cudaFree(pointer);pointer=next;capacity=count;}
    void upload(const std::vector<T>& value,PinnedTransferQueue& transfer){reserve(value.size());transfer.upload(pointer,value.data(),value.size()*sizeof(T));}
    void download(std::vector<T>& value,std::size_t count,PinnedTransferQueue& transfer){value.resize(count);transfer.download(value.data(),pointer,count*sizeof(T));}
};
__global__ void facesKernel(int count,TopologyView t,const Point* points,FaceResult* out){
    const int face=blockIdx.x*blockDim.x+threadIdx.x;if(face<count)out[face]=faceGeometry(face,t,points);}
__global__ void rebuildFacesKernel(int count,TopologyView t,const Point* points,FaceResult* out,Point* area){
    const int f=blockIdx.x*blockDim.x+threadIdx.x;if(f<count){out[f]=faceGeometry(f,t,points);area[f]=out[f].area;}}
__global__ void rebuildCellsKernel(int count,TopologyView t,const Point* points,const Point* area,CellResult* out,Point* centres){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<count){out[c]=cellGeometry(c,t,points,area);centres[c]=out[c].centre;}}
__global__ void cellsKernel(int count,TopologyView t,const Point* points,const Point* area,CellResult* out){
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell<count)out[cell]=cellGeometry(cell,t,points,area);}
__global__ void sweepKernel(int count,TopologyView t,const Point* old,const Point* endpoint,SweepResult* out){
    const int face=blockIdx.x*blockDim.x+threadIdx.x;if(face<count)out[face]=faceSweep(face,t,old,endpoint);}
__global__ void surfaceEdgesKernel(int count,TopologyView t,const Point* old,const Point* end,const SurfaceEdge* edges,
    bool empty,int segments,double scale,EdgeResult* out){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)out[i]=surfaceEdge(t,old,end,edges[i],empty,segments,scale);
}
__global__ void boundaryMotionKernel(int count,const Point* areas,const BoundaryMotionFace* faces,
    const int* offsets,const int* incident,const int* sideOffsets,const int* sides,Point* normals,double dt,BoundaryMotionPoint* out){
    const int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=count)return;
    BoundaryMotionPoint r;Point support;
    for(int k=offsets[p];k<offsets[p+1];++k){const auto face=faces[incident[k]];const Point av=areas[face.face],normal=normalized(av);
        const double area=magnitude(av);const Point delta=normal*(face.offset-face.previousOffset)+face.supportVelocity*dt;
        r.displacement+=delta*area;support+=face.supportVelocity*(dt*area);r.direction+=av;r.weight+=area;r.thickness+=face.thickness*area;
    }
    if(r.weight>0){r.displacement=r.displacement/r.weight;support=support/r.weight;r.thickness/=r.weight;}
    int constraints=0;Point* basis=normals?normals+sideOffsets[p]:nullptr;
    if(r.weight>0)for(int k=sideOffsets[p];k<sideOffsets[p+1];++k){Point q=normalized(areas[sides[k]]);
        for(int j=0;j<constraints;++j)q=q-basis[j]*dot(q,basis[j]);
        if(magnitude(q)>1e-10)basis[constraints++]=normalized(q);
    }
    Point recession=r.displacement-support;
    for(int j=0;j<constraints;++j){recession=recession-basis[j]*dot(recession,basis[j]);r.direction=r.direction-basis[j]*dot(r.direction,basis[j]);}
    if(constraints==3){recession={};r.direction={};}
    r.displacement=recession+support;r.direction=normalized(r.direction);out[p]=r;
}
constexpr double maximumDistance=std::numeric_limits<double>::max();
__global__ void wallDistanceKernel(int count,TopologyView t,const Point* points,const Point* centres,
    int wallCount,const int* walls,WallDistanceResult* output){
    const int cell=blockIdx.x;if(cell>=count)return;
    __shared__ double distance[128];__shared__ int invalid[128];
    double best=maximumDistance;int firstInvalid=INT_MAX;
    // A block shares one cell's independent triangle queries. Only the finite
    // minimum is reduced; the first invalid face retains the input list order.
    for(int i=threadIdx.x;i<wallCount;i+=blockDim.x){const int face=walls[i],begin=t.faceOffsets[face],end=t.faceOffsets[face+1];
        for(int j=begin+1;j+1<end;++j){const double triangle=extended::triangleDistance(centres[cell],points[t.facePoints[begin]],
            points[t.facePoints[j]],points[t.facePoints[j+1]]);
            if(!std::isfinite(triangle)){firstInvalid=i;break;}if(triangle<best)best=triangle;}
        if(firstInvalid!=INT_MAX)break;
    }
    distance[threadIdx.x]=best;invalid[threadIdx.x]=firstInvalid;__syncthreads();
    for(int half=64;half;half/=2){if(threadIdx.x<half){
        if(distance[threadIdx.x+half]<distance[threadIdx.x])distance[threadIdx.x]=distance[threadIdx.x+half];
        if(invalid[threadIdx.x+half]<invalid[threadIdx.x])invalid[threadIdx.x]=invalid[threadIdx.x+half];}__syncthreads();}
    if(!threadIdx.x){WallDistanceResult result;result.distance=wallCount?distance[0]:0;
        if(invalid[0]!=INT_MAX)result.invalidFace=walls[invalid[0]];output[cell]=result;}
}
struct MotionControl {double residual=0,tolerance=0;int iterations=0,maximumIterations=0,active=0,errorPoint=INT_MAX;};
__global__ void weightsKernel(int count,const Point* points,const int* offsets,const int* neighbours,const unsigned char* fixed,
    double* weights,double* total,int* errors){
    const int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=count)return;
    total[p]=0;errors[p]=0;if(fixed[p])return;
    for(int k=offsets[p];k<offsets[p+1];++k){const double length=magnitude(points[p]-points[neighbours[k]]);
        if(!(length>0)){errors[p]=1;return;}weights[k]=1/length;total[p]+=weights[k];}
}
__global__ void initializeMotion(int count,const Point* boundary,const unsigned char* fixed,Point* displacement,
    const int* preErrors,int* errors,MotionControl* control,double tolerance,int maximumIterations){
    const int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(!p){*control={};control->tolerance=tolerance;control->maximumIterations=maximumIterations;control->active=count>0&&maximumIterations>0;}
    if(p<count){displacement[p]=fixed[p]?boundary[p]:Point{};errors[p]=preErrors[p];}
}
// A cooperative grid owns the complete recurrence. Every iteration retains
// the Jacobi barrier and original ordered neighbour sum, without host polling.
__global__ void solveMotion(int count,const int* offsets,const int* neighbours,const double* weights,const double* total,
    const unsigned char* fixed,Point* displacement,Point* next,double* partial,int* errors,MotionControl* control){
    auto grid=cooperative_groups::this_grid();const int index=blockIdx.x*blockDim.x+threadIdx.x,stride=blockDim.x*gridDim.x;
    __shared__ double maxima[256];
    while(control->active){
        double local=0;
        for(int p=index;p<count;p+=stride){
            if(fixed[p]){next[p]=displacement[p];continue;}
            if(errors[p]){atomicMin(&control->errorPoint,p);continue;}
            Point sum;for(int k=offsets[p];k<offsets[p+1];++k)sum+=displacement[neighbours[k]]*weights[k];
            next[p]=sum/total[p];if(!finite(next[p])){errors[p]=2;atomicMin(&control->errorPoint,p);continue;}
            const double residual=magnitude(next[p]-displacement[p]);if(local<residual)local=residual;
        }
        maxima[threadIdx.x]=local;__syncthreads();
        for(int half=128;half;half/=2){if(threadIdx.x<half&&maxima[threadIdx.x]<maxima[threadIdx.x+half])maxima[threadIdx.x]=maxima[threadIdx.x+half];__syncthreads();}
        if(!threadIdx.x)partial[blockIdx.x]=maxima[0];grid.sync();
        if(control->errorPoint==INT_MAX)for(int p=index;p<count;p+=stride)displacement[p]=next[p];
        grid.sync();
        if(!index){
            if(control->errorPoint!=INT_MAX)control->active=0;
            else{double residual=0;for(int i=0;i<gridDim.x;++i)if(residual<partial[i])residual=partial[i];
                control->residual=residual;++control->iterations;
                control->active=!(residual<=control->tolerance)&&control->iterations<control->maximumIterations;}
        }
        grid.sync();
    }
}
template<class Kernel> int cooperativeBlocks(Kernel kernel,int requested,int threads){
    int perSm=0,device=0,sms=0;checked(cudaGetDevice(&device));checked(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));
    checked(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm,kernel,threads,0));
    return std::max(1,std::min(requested,perSm*sms));
}
__global__ void publishMotion(int count,const Point* points,const Point* displacement,const Point* boundary,
    const unsigned char* fixed,bool represented,Point* result){
    const int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=count)return;
    result[p]=points[p]+displacement[p];
    if(represented){result[p]=result[p]-points[p];if(fixed[p])result[p]=boundary[p];}
}
}
struct PolyhedralGeometry::Implementation {
    cudaStream_t stream=nullptr;PinnedTransferQueue transfer;Topology cached;
    struct RebuildCache {
        Topology topology;std::vector<Point> points;std::vector<int> walls;RebuildResult result;std::size_t bytes=0;std::uint64_t hash=0;
    };
    // Exact memoization of immutable geometry queries. The bounded host mirror
    // reuses values computed on the GPU; a miss always executes the GPU kernels.
    // No solver options, case identity, tolerance or physical regime form keys.
    using CacheList=std::list<RebuildCache>;
    CacheList rebuildCache;std::unordered_multimap<std::uint64_t,CacheList::iterator> cacheIndex;std::size_t cacheBytes=0;
    static constexpr std::size_t cacheBudget=64u*1024u*1024u;
    static bool sameTopology(const Topology& a,const Topology& b){
        return a.faceOffsets==b.faceOffsets&&a.facePoints==b.facePoints&&a.cellOffsets==b.cellOffsets&&a.cellFaces==b.cellFaces
            &&a.cellSigns==b.cellSigns&&a.vertexOffsets==b.vertexOffsets&&a.vertices==b.vertices;
    }
    static std::uint64_t queryHash(const Topology& t,const std::vector<Point>& points,const std::vector<int>& walls){
        std::uint64_t hash=14695981039346656037ull;
        const auto bytes=[&](const void* data,std::size_t count){const auto* p=static_cast<const unsigned char*>(data);
            for(std::size_t i=0;i<count;++i){hash^=p[i];hash*=1099511628211ull;}};
        for(const auto* field:{&t.faceOffsets,&t.facePoints,&t.cellOffsets,&t.cellFaces,&t.cellSigns,&t.vertexOffsets,&t.vertices,&walls}){
            const auto count=field->size();bytes(&count,sizeof(count));bytes(field->data(),count*sizeof(int));}
        bytes(points.data(),points.size()*sizeof(Point));return hash;
    }
    bool findRebuild(std::uint64_t hash,const Topology& t,const std::vector<Point>& p,const std::vector<int>& walls,RebuildResult& result){
        const auto range=cacheIndex.equal_range(hash);
        for(auto slot=range.first;slot!=range.second;++slot){auto i=slot->second;
            if(i->points.size()!=p.size()||i->walls!=walls||!sameTopology(i->topology,t))continue;
            static_assert(sizeof(Point)==3*sizeof(double),"packed geometry point required");
            if(!p.empty()&&std::memcmp(i->points.data(),p.data(),p.size()*sizeof(Point)))continue;
            result=i->result;rebuildCache.splice(rebuildCache.begin(),rebuildCache,i);return true;
        }return false;
    }
    void rememberRebuild(std::uint64_t hash,const Topology& t,const std::vector<Point>& p,const std::vector<int>& walls,const RebuildResult& result){
        const std::size_t bytes=(t.faceOffsets.size()+t.facePoints.size()+t.cellOffsets.size()+t.cellFaces.size()+t.cellSigns.size()
            +t.vertexOffsets.size()+t.vertices.size()+walls.size())*sizeof(int)+p.size()*sizeof(Point)+result.faces.size()*sizeof(FaceResult)
            +result.cells.size()*sizeof(CellResult)+result.distance.size()*sizeof(WallDistanceResult)+sizeof(RebuildCache)+64;
        if(bytes>cacheBudget)return;
        while(!rebuildCache.empty()&&cacheBytes+bytes>cacheBudget){
            const auto last=std::prev(rebuildCache.end());const auto range=cacheIndex.equal_range(last->hash);
            for(auto i=range.first;i!=range.second;++i)if(i->second==last){cacheIndex.erase(i);break;}
            cacheBytes-=last->bytes;rebuildCache.erase(last);}
        rebuildCache.push_front({t,p,walls,result,bytes,hash});
        try{cacheIndex.emplace(hash,rebuildCache.begin());}catch(...){rebuildCache.pop_front();throw;}
        cacheBytes+=bytes;
    }
    Array<int> faceOffsets,facePoints,cellOffsets,cellFaces,cellSigns,vertexOffsets,vertices;
    Array<SurfaceEdge> edges;Array<EdgeResult> edgeResult;
    Array<BoundaryMotionFace> motionFaces;Array<BoundaryMotionPoint> motionPoints;
    Array<int> motionOffsets,motionIncident,sideOffsets,sideIncident;Array<Point> sideNormals;
    Array<Point> points,endpoint,area,centres;Array<int> walls;Array<WallDistanceResult> distance;Array<FaceResult> faceResult;Array<CellResult> cellResult;Array<SweepResult> sweepResult;
    Implementation(){checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));transfer.bind(stream);}
    ~Implementation(){transfer.discard();if(stream){cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}}
    void topology(const Topology& t){
#define FIELD(name) if(cached.name!=t.name){name.upload(t.name,transfer);cached.name=t.name;}
        FIELD(faceOffsets) FIELD(facePoints)
        // Face-only operations do not invalidate unrelated cached cell CSR.
        if(!t.cellOffsets.empty()){FIELD(cellOffsets) FIELD(cellFaces) FIELD(cellSigns) FIELD(vertexOffsets) FIELD(vertices)}
#undef FIELD
    }
    TopologyView view()const{return {faceOffsets.pointer,facePoints.pointer,cellOffsets.pointer,cellFaces.pointer,cellSigns.pointer,vertexOffsets.pointer,vertices.pointer};}
};
PolyhedralGeometry::PolyhedralGeometry():data_(new Implementation){}
PolyhedralGeometry::~PolyhedralGeometry()=default;
RebuildResult PolyhedralGeometry::rebuild(const Topology& t,const std::vector<Point>& points,const std::vector<int>& walls){
    auto& d=*data_;RebuildResult out;const auto hash=d.queryHash(t,points,walls);if(d.findRebuild(hash,t,points,walls,out))return out;
    PinnedTransferQueue::Operation operation(d.transfer);d.topology(t);d.points.upload(points,d.transfer);d.walls.upload(walls,d.transfer);
    const int nf=t.faceOffsets.empty()?0:int(t.faceOffsets.size()-1),nc=t.cellOffsets.empty()?0:int(t.cellOffsets.size()-1);
    d.faceResult.reserve(nf);d.area.reserve(nf);d.cellResult.reserve(nc);d.centres.reserve(nc);d.distance.reserve(nc);
    if(nf)rebuildFacesKernel<<<(nf+127)/128,128,0,d.stream>>>(nf,d.view(),d.points.pointer,d.faceResult.pointer,d.area.pointer);
    if(nc){rebuildCellsKernel<<<(nc+127)/128,128,0,d.stream>>>(nc,d.view(),d.points.pointer,d.area.pointer,d.cellResult.pointer,d.centres.pointer);
        wallDistanceKernel<<<nc,128,0,d.stream>>>(nc,d.view(),d.points.pointer,d.centres.pointer,walls.size(),d.walls.pointer,d.distance.pointer);}
    d.faceResult.download(out.faces,nf,d.transfer);d.cellResult.download(out.cells,nc,d.transfer);d.distance.download(out.distance,nc,d.transfer);d.transfer.finish();d.rememberRebuild(hash,t,points,walls,out);return out;
}
std::vector<BoundaryMotionPoint> PolyhedralGeometry::boundaryMotion(const Topology& t,int count,const std::vector<Point>& areas,
    const std::vector<BoundaryMotionFace>& faces,const std::vector<unsigned char>& sideFaces,double interval){
    if(count<0||sideFaces.size()!=areas.size()||t.faceOffsets.size()!=areas.size()+1)
        throw std::invalid_argument("boundary motion topology shape mismatch");
    // CSR preserves original surface-face order for every vertex. No atomics
    // change the area-weighted accumulation or Gram-Schmidt projection order.
    std::vector<int> offsets(count+1,0),sideOffsets(count+1,0);
    for(const auto& f:faces)for(int k=t.faceOffsets[f.face];k<t.faceOffsets[f.face+1];++k)++offsets[t.facePoints[k]+1];
    for(std::size_t f=0;f<sideFaces.size();++f)if(sideFaces[f])for(int k=t.faceOffsets[f];k<t.faceOffsets[f+1];++k)++sideOffsets[t.facePoints[k]+1];
    for(int p=1;p<=count;++p){offsets[p]+=offsets[p-1];sideOffsets[p]+=sideOffsets[p-1];}
    std::vector<int> incident(offsets.back()),sideIncident(sideOffsets.back()),cursor=offsets,sideCursor=sideOffsets;
    for(std::size_t i=0;i<faces.size();++i)for(int k=t.faceOffsets[faces[i].face];k<t.faceOffsets[faces[i].face+1];++k)incident[cursor[t.facePoints[k]]++]=i;
    for(std::size_t f=0;f<sideFaces.size();++f)if(sideFaces[f])for(int k=t.faceOffsets[f];k<t.faceOffsets[f+1];++k)sideIncident[sideCursor[t.facePoints[k]]++]=f;
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfer);d.area.upload(areas,d.transfer);d.motionFaces.upload(faces,d.transfer);
    d.motionOffsets.upload(offsets,d.transfer);d.motionIncident.upload(incident,d.transfer);d.sideOffsets.upload(sideOffsets,d.transfer);d.sideIncident.upload(sideIncident,d.transfer);
    d.sideNormals.reserve(sideIncident.size());d.motionPoints.reserve(count);
    if(count)boundaryMotionKernel<<<(count+127)/128,128,0,d.stream>>>(count,d.area.pointer,d.motionFaces.pointer,d.motionOffsets.pointer,d.motionIncident.pointer,
        d.sideOffsets.pointer,d.sideIncident.pointer,d.sideNormals.pointer,interval,d.motionPoints.pointer);
    std::vector<BoundaryMotionPoint> out;d.motionPoints.download(out,count,d.transfer);d.transfer.finish();return out;
}
std::vector<FaceResult> PolyhedralGeometry::faces(const Topology& t,const std::vector<Point>& p){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfer);d.topology(t);d.points.upload(p,d.transfer);
    const int count=t.faceOffsets.empty()?0:int(t.faceOffsets.size()-1);d.faceResult.reserve(count);
    if(count)facesKernel<<<(count+127)/128,128,0,d.stream>>>(count,d.view(),d.points.pointer,d.faceResult.pointer);
    std::vector<FaceResult> result;d.faceResult.download(result,count,d.transfer);d.transfer.finish();return result;
}
std::vector<CellResult> PolyhedralGeometry::cells(const Topology& t,const std::vector<Point>& p,const std::vector<Point>& area){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfer);d.topology(t);d.points.upload(p,d.transfer);d.area.upload(area,d.transfer);
    const int count=t.cellOffsets.empty()?0:int(t.cellOffsets.size()-1);d.cellResult.reserve(count);
    if(count)cellsKernel<<<(count+127)/128,128,0,d.stream>>>(count,d.view(),d.points.pointer,d.area.pointer,d.cellResult.pointer);
    std::vector<CellResult> result;d.cellResult.download(result,count,d.transfer);d.transfer.finish();return result;
}
std::vector<WallDistanceResult> PolyhedralGeometry::wallDistance(const Topology& t,const std::vector<Point>& p,
    const std::vector<Point>& centres,const std::vector<int>& walls){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfer);d.topology(t);d.points.upload(p,d.transfer);
    d.centres.upload(centres,d.transfer);d.walls.upload(walls,d.transfer);const int count=centres.size();d.distance.reserve(count);
    if(count)wallDistanceKernel<<<count,128,0,d.stream>>>(count,d.view(),d.points.pointer,d.centres.pointer,walls.size(),d.walls.pointer,d.distance.pointer);
    std::vector<WallDistanceResult> result;d.distance.download(result,count,d.transfer);d.transfer.finish();return result;
}
std::vector<SweepResult> PolyhedralGeometry::sweep(const Topology& t,const std::vector<Point>& p,const std::vector<Point>& endpoint){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfer);d.topology(t);d.points.upload(p,d.transfer);d.endpoint.upload(endpoint,d.transfer);
    const int count=t.faceOffsets.empty()?0:int(t.faceOffsets.size()-1);d.sweepResult.reserve(count);
    if(count)sweepKernel<<<(count+127)/128,128,0,d.stream>>>(count,d.view(),d.points.pointer,d.endpoint.pointer,d.sweepResult.pointer);
    std::vector<SweepResult> result;d.sweepResult.download(result,count,d.transfer);d.transfer.finish();return result;
}
std::vector<EdgeResult> PolyhedralGeometry::surfaceEdges(const Topology& t,const std::vector<Point>& old,const std::vector<Point>& end,
    const std::vector<SurfaceEdge>& edges,bool empty,int segments,double scale){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfer);d.topology(t);d.points.upload(old,d.transfer);
    d.endpoint.upload(end,d.transfer);d.edges.upload(edges,d.transfer);const int count=edges.size();d.edgeResult.reserve(count);
    if(count)surfaceEdgesKernel<<<(count+63)/64,64,0,d.stream>>>(count,d.view(),d.points.pointer,d.endpoint.pointer,d.edges.pointer,empty,segments,scale,d.edgeResult.pointer);
    std::vector<EdgeResult> out;d.edgeResult.download(out,count,d.transfer);d.transfer.finish();return out;
}
struct HarmonicPointMotion::Implementation {
    int count,blocks,maximumIterations;double tolerance;
    cudaStream_t stream=nullptr;PinnedTransferQueue transfer;
    Array<Point> points,boundary,displacement,next,result;Array<int> offsets,neighbours,preErrors,errors;Array<unsigned char> fixed;
    Array<double> weights,total,partial;Array<MotionControl> control;
    Implementation(int n,double tol,int limit):count(n),blocks(cooperativeBlocks(solveMotion,std::max(1,(n+255)/256),256)),maximumIterations(limit),tolerance(tol){
        checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));transfer.bind(stream);}
    ~Implementation(){transfer.discard();if(stream){cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}}
    void initialize(const std::vector<Point>& p,const std::vector<int>& row,const std::vector<int>& adjacent,const std::vector<unsigned char>& mask){
        PinnedTransferQueue::Operation operation(transfer);points.upload(p,transfer);offsets.upload(row,transfer);neighbours.upload(adjacent,transfer);fixed.upload(mask,transfer);
        for(auto* array:{&boundary,&displacement,&next,&result})array->reserve(count);
        for(auto* array:{&preErrors,&errors})array->reserve(count);
        total.reserve(count);weights.reserve(adjacent.size());partial.reserve(blocks);control.reserve(1);
        if(count)weightsKernel<<<(count+127)/128,128,0,stream>>>(count,points.pointer,offsets.pointer,neighbours.pointer,fixed.pointer,weights.pointer,total.pointer,preErrors.pointer);
        transfer.finish();

    }
};
HarmonicPointMotion::HarmonicPointMotion(const std::vector<Point>& points,const std::vector<int>& offsets,const std::vector<int>& neighbours,
    const std::vector<unsigned char>& fixed,double tolerance,int maximumIterations):data_(new Implementation(points.size(),tolerance,maximumIterations)){
    if(fixed.size()!=points.size()||offsets.size()!=points.size()+1||offsets.front()!=0||std::size_t(offsets.back())!=neighbours.size())
        throw std::invalid_argument("harmonic motion CSR shape mismatch");
    data_->initialize(points,offsets,neighbours,fixed);
}
HarmonicPointMotion::~HarmonicPointMotion()=default;
HarmonicResult HarmonicPointMotion::solve(const std::vector<Point>& boundary,bool represented){
    auto& d=*data_;if(boundary.size()!=std::size_t(d.count))throw std::invalid_argument("harmonic motion boundary shape mismatch");
    PinnedTransferQueue::Operation operation(d.transfer);d.boundary.upload(boundary,d.transfer);
    initializeMotion<<<std::max(1,(d.count+255)/256),256,0,d.stream>>>(d.count,d.boundary.pointer,d.fixed.pointer,d.displacement.pointer,d.preErrors.pointer,d.errors.pointer,
        d.control.pointer,d.tolerance,d.maximumIterations);
    void* arguments[]={&d.count,&d.offsets.pointer,&d.neighbours.pointer,&d.weights.pointer,&d.total.pointer,
        &d.fixed.pointer,&d.displacement.pointer,&d.next.pointer,&d.partial.pointer,&d.errors.pointer,&d.control.pointer};
    checked(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(solveMotion),d.blocks,256,arguments,0,d.stream));
    MotionControl status;d.transfer.download(&status,d.control.pointer,sizeof(status));d.transfer.finish();
    HarmonicResult result;result.residual=status.residual;result.iterations=status.iterations;
    if(status.errorPoint!=INT_MAX){result.errorPoint=status.errorPoint;d.transfer.download(&result.errorKind,d.errors.pointer+status.errorPoint,sizeof(int));}
    else {if(d.count)publishMotion<<<(d.count+255)/256,256,0,d.stream>>>(d.count,d.points.pointer,d.displacement.pointer,d.boundary.pointer,d.fixed.pointer,represented,d.result.pointer);
        d.result.download(result.points,d.count,d.transfer);}
    d.transfer.finish();return result;
}

namespace {
struct QrControl {double largest=0,length=0;int rank=0,active=0,error=0;};
__global__ void qrNorms(int nr,int nc,const double* matrix,double* norm,int* order,int* invalid){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nr)return;
    order[i]=i;invalid[i]=0;double sum=0;
    for(int j=0;j<nc;++j){const double x=matrix[std::size_t(i)*nc+j];
        if(!isfinite(x)){invalid[i]=1;break;}sum+=x*x;}
    norm[i]=sum;if(!isfinite(sum))invalid[i]=1;
}
__global__ void qrInitialize(int nr,int nc,const double* norm,const int* invalid,QrControl* c){
    *c={};for(int i=0;i<nr;++i){if(invalid[i]){c->error=1;return;}
        const double length=sqrt(norm[i]);if(c->largest<length)c->largest=length;}
    c->active=nr>0&&nc>0;
}
__global__ void solveQr(int nr,int nc,double tolerance,double* matrix,double* norm,int* order,double* rhs,double* delta,QrControl* c){
    auto grid=cooperative_groups::this_grid();const int index=blockIdx.x*blockDim.x+threadIdx.x,stride=blockDim.x*gridDim.x;
    __shared__ double projection;
    for(;;){
        // Snapshot collectively before the pivot thread can change active.
        // Reading it directly in a while condition races the next pivot with
        // late-arriving threads and can leave them outside the grid barrier.
        const bool active=c->active;grid.sync();if(!active)break;
        if(!index){const int rank=c->rank;int pivot=rank;
            for(int i=rank+1;i<nr;++i)if(norm[i]>norm[pivot])pivot=i;
            const double length=sqrt(norm[pivot]);
            if(!(length>tolerance*c->largest))c->active=0;
            else{const int p=order[rank];order[rank]=order[pivot];order[pivot]=p;
                const double n=norm[rank];norm[rank]=norm[pivot];norm[pivot]=n;
                const double r=rhs[rank];rhs[rank]=rhs[pivot];rhs[pivot]=r;
                rhs[rank]/=length;c->length=length;if(!isfinite(rhs[rank])){c->error=2;c->active=0;}}
        }
        grid.sync();if(!c->active)break;
        for(int j=index;j<nc;j+=stride){double& x=matrix[std::size_t(order[c->rank])*nc+j];x/=c->length;delta[j]+=x*rhs[c->rank];}
        grid.sync();
        for(int i=c->rank+1+blockIdx.x;i<nr;i+=gridDim.x){
            double* row=matrix+std::size_t(order[i])*nc;const double* pivot=matrix+std::size_t(order[c->rank])*nc;
            for(int pass=0;pass<2;++pass){
                if(!threadIdx.x){double sum=0;for(int j=0;j<nc;++j)sum+=row[j]*pivot[j];projection=sum;}
                __syncthreads();for(int j=threadIdx.x;j<nc;j+=blockDim.x)row[j]-=projection*pivot[j];
                if(!threadIdx.x)rhs[i]-=projection*rhs[c->rank];__syncthreads();
            }
            if(!threadIdx.x){double sum=0;for(int j=0;j<nc;++j)sum+=row[j]*row[j];norm[i]=sum;}
            __syncthreads();
        }
        grid.sync();if(!index){++c->rank;c->active=c->rank<nr&&c->rank<nc;}grid.sync();
    }
}

}
struct DenseMinimumNorm::Implementation {
    cudaStream_t stream=nullptr;PinnedTransferQueue transfer;
    int blocks=1;
    Array<double> matrix,rhs,norm,delta;Array<int> order,invalid;Array<QrControl> control;
    Implementation(){checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));transfer.bind(stream);}
    ~Implementation(){transfer.discard();if(stream){cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}}
    void prepare(int nr,int nc){
        blocks=cooperativeBlocks(solveQr,std::max(1,nr),128);
        matrix.reserve(std::size_t(nr)*nc);rhs.reserve(nr);norm.reserve(nr);delta.reserve(nc);
        order.reserve(nr);invalid.reserve(nr);control.reserve(1);
    }
};
DenseMinimumNorm::DenseMinimumNorm():data_(new Implementation){}
DenseMinimumNorm::~DenseMinimumNorm()=default;
MinimumNormResult DenseMinimumNorm::solve(int nr,int nc,const std::vector<double>& matrix,const std::vector<double>& rhs,double tolerance){
    if(nr<0||nc<0||matrix.size()!=std::size_t(nr)*nc||rhs.size()!=std::size_t(nr))
        throw std::invalid_argument("minimum-norm matrix shape mismatch");
    auto& d=*data_;d.prepare(nr,nc);PinnedTransferQueue::Operation operation(d.transfer);
    d.matrix.upload(matrix,d.transfer);d.rhs.upload(rhs,d.transfer);
    if(nc)checked(cudaMemsetAsync(d.delta.pointer,0,std::size_t(nc)*sizeof(double),d.stream));
    if(nr)qrNorms<<<(nr+127)/128,128,0,d.stream>>>(nr,nc,d.matrix.pointer,d.norm.pointer,d.order.pointer,d.invalid.pointer);
    qrInitialize<<<1,1,0,d.stream>>>(nr,nc,d.norm.pointer,d.invalid.pointer,d.control.pointer);
    void* arguments[]={&nr,&nc,&tolerance,&d.matrix.pointer,&d.norm.pointer,&d.order.pointer,&d.rhs.pointer,&d.delta.pointer,&d.control.pointer};
    checked(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(solveQr),d.blocks,128,arguments,0,d.stream));
    QrControl state;d.transfer.download(&state,d.control.pointer,sizeof(state));d.transfer.finish();
    MinimumNormResult result;result.rank=state.rank;result.initialized=state.error!=1;result.valid=state.error==0;
    if(result.initialized){d.delta.download(result.delta,nc,d.transfer);d.rhs.download(result.rhs,nr,d.transfer);}
    d.transfer.finish();return result;
}

namespace {
__global__ void planarValues(int count,const Point* old,const Point* end,const PlanarConstraint* planar,Polynomial* out){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)out[i]=planarPolynomial(old,end,planar[i]);
}
__global__ void linearizationColumn(int faceCount,int planarCount,TopologyView t,const Point* old,const Point* end,
    const Point* mode,const int* faces,const double* scales,const PlanarConstraint* planar,double* out){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<faceCount)out[i]=sweepDerivative(t,old,end,mode,faces[i])/scales[i];
    else if(i<faceCount+planarCount){const int j=i-faceCount;const auto p=planar[j];
        const auto polynomial=planarPolynomial(old,end,p,mode);
        for(int k=0;k<3;++k)out[faceCount+3*j+k]=polynomial.value[k]/p.scale;}
}
__global__ void accumulateMode(int count,const Point* mode,double coefficient,Point* correction){
    const int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<count)correction[p]+=mode[p]*coefficient;
}
}
struct SweepLinearization::Implementation {
    cudaStream_t stream=nullptr;PinnedTransferQueue transfer;int points=0,faceCount=0,planarCount=0;
    Array<int> offsets,indices,faces;Array<double> scales,column;Array<PlanarConstraint> planar;
    Array<Point> old,end,mode,correction;Array<Polynomial> polynomials;
    Implementation(){checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));transfer.bind(stream);}
    ~Implementation(){transfer.discard();if(stream){cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}}
    TopologyView view()const{return {offsets.pointer,indices.pointer,nullptr,nullptr,nullptr,nullptr,nullptr};}
};
SweepLinearization::SweepLinearization(const Topology& t,const std::vector<Point>& old,const std::vector<int>& faces,
    const std::vector<double>& scales,const std::vector<PlanarConstraint>& planar):data_(new Implementation){
    if(faces.size()!=scales.size())throw std::invalid_argument("sweep face scale shape mismatch");
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfer);d.points=old.size();d.faceCount=faces.size();d.planarCount=planar.size();
    d.offsets.upload(t.faceOffsets,d.transfer);d.indices.upload(t.facePoints,d.transfer);d.old.upload(old,d.transfer);
    d.faces.upload(faces,d.transfer);d.scales.upload(scales,d.transfer);d.planar.upload(planar,d.transfer);
    for(auto* array:{&d.end,&d.mode,&d.correction})array->reserve(d.points);
    d.column.reserve(std::size_t(d.faceCount)+3*std::size_t(d.planarCount));d.polynomials.reserve(d.planarCount);d.transfer.finish();
}
SweepLinearization::~SweepLinearization()=default;
void SweepLinearization::endpoint(const std::vector<Point>& points){
    auto& d=*data_;if(points.size()!=std::size_t(d.points))throw std::invalid_argument("sweep endpoint shape mismatch");
    PinnedTransferQueue::Operation operation(d.transfer);d.end.upload(points,d.transfer);d.transfer.finish();
}
std::vector<Polynomial> SweepLinearization::planarity(const std::vector<Point>& points){
    endpoint(points);auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfer);
    if(d.planarCount)planarValues<<<(d.planarCount+127)/128,128,0,d.stream>>>(d.planarCount,d.old.pointer,d.end.pointer,d.planar.pointer,d.polynomials.pointer);
    std::vector<Polynomial> out;d.polynomials.download(out,d.planarCount,d.transfer);d.transfer.finish();return out;
}
std::vector<double> SweepLinearization::column(const std::vector<Point>& mode){
    auto& d=*data_;if(mode.size()!=std::size_t(d.points))throw std::invalid_argument("sweep mode shape mismatch");
    PinnedTransferQueue::Operation operation(d.transfer);d.mode.upload(mode,d.transfer);const int work=d.faceCount+d.planarCount;
    if(work)linearizationColumn<<<(work+127)/128,128,0,d.stream>>>(d.faceCount,d.planarCount,d.view(),d.old.pointer,d.end.pointer,
        d.mode.pointer,d.faces.pointer,d.scales.pointer,d.planar.pointer,d.column.pointer);
    std::vector<double> out;d.column.download(out,std::size_t(d.faceCount)+3*std::size_t(d.planarCount),d.transfer);d.transfer.finish();return out;
}
void SweepLinearization::resetCorrection(){auto& d=*data_;if(d.points)checked(cudaMemsetAsync(d.correction.pointer,0,std::size_t(d.points)*sizeof(Point),d.stream));}
void SweepLinearization::accumulate(const std::vector<Point>& mode,double coefficient){
    auto& d=*data_;if(mode.size()!=std::size_t(d.points))throw std::invalid_argument("sweep correction mode shape mismatch");
    PinnedTransferQueue::Operation operation(d.transfer);d.mode.upload(mode,d.transfer);
    if(d.points)accumulateMode<<<(d.points+127)/128,128,0,d.stream>>>(d.points,d.mode.pointer,coefficient,d.correction.pointer);
    d.transfer.finish();
}
std::vector<Point> SweepLinearization::correction(){auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfer);
    std::vector<Point> out;d.correction.download(out,d.points,d.transfer);d.transfer.finish();return out;}
}}}
