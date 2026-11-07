// Included inside ugkp::gpu::geometry after the bounds operators.
namespace {
__global__ void resetSurfaceSeams(int count,const SurfaceTopologyEdge* initial,
    SurfaceTopologyEdge* working,unsigned char* removed,SurfaceSeamStatus* status){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count){working[i]=initial[i];removed[i]=0;}
    if(i==0)*status={};
}
__global__ void matchResidentSurfaceSeams(int count,const int* surfaceFaces,
    const SurfaceSeamPatch* patches,const Point* points,const FaceResult* faces,
    SurfaceTopologyEdge* edges,unsigned char* removed,SurfaceEdgeMap* output,SurfaceSeamStatus* status){
    __shared__ int matches[128],first[128];
    const int lane=threadIdx.x;const double tolerance=1e-10*status->scale;
    for(int i=0;i<count;++i){
        if(removed[i]||edges[i].neighbour>=0||patches[i].face<0)continue;
        const auto edge=edges[i];const auto patch=patches[i];
        const Point translation=faces[patch.partner].centre-faces[patch.face].centre;
        int found=0,minimum=count;
        for(int j=lane;j<count;j+=blockDim.x)if(i!=j&&!removed[j]&&edges[j].neighbour<0
            &&surfaceSeamPairMatches(edge,edges[j],points,translation,tolerance)){
            found=found<2?found+1:2;minimum=j<minimum?j:minimum;
        }
        matches[lane]=found;first[lane]=minimum;__syncthreads();
        for(int stride=64;stride;stride/=2){if(lane<stride){
            const int total=matches[lane]+matches[lane+stride];matches[lane]=total>2?2:total;
            first[lane]=first[lane]<first[lane+stride]?first[lane]:first[lane+stride];
        }__syncthreads();}
        if(lane==0){
            if(matches[0]!=1){status->failure=matches[0]?SurfaceSeamFailure::Ambiguous:SurfaceSeamFailure::Unpaired;status->index=i;}
            else{const int j=first[0];edges[i].neighbour=edges[j].owner;
                edges[i].neighbourA=edges[j].a;edges[i].neighbourB=edges[j].b;removed[j]=1;}
        }
        __syncthreads();if(status->failure!=SurfaceSeamFailure::None)return;
    }
    // Compact only a complete valid candidate; consumers use this device count.
    if(lane==0){int kept=0;for(int i=0;i<count;++i)if(!removed[i]){
        const auto e=edges[i];output[kept++]={{e.a,e.b,surfaceFaces[e.owner],
            e.neighbour<0?-1:surfaceFaces[e.neighbour],e.neighbourA,e.neighbourB},e.owner};
    }status->count=kept;}
}
}
struct ResidentSurfaceSeams::Implementation {
    int points=0,edges=0;
    Array<int> surfaceFaces;Array<SurfaceTopologyEdge> initial,working;
    Array<SurfaceSeamPatch> patches;Array<unsigned char> removed;
    Array<SurfaceEdgeMap> output;Array<GeometryBounds> bounds;Array<SurfaceSeamStatus> status;
    template<class T>void install(Array<T>& dst,const std::vector<T>& src){dst.reserve(src.size());
        if(!src.empty())checked(cudaMemcpy(dst.pointer,src.data(),src.size()*sizeof(T),cudaMemcpyHostToDevice));}
};
ResidentSurfaceSeams::ResidentSurfaceSeams(int points,int meshFaces,const std::vector<int>& surfaceFaces,
    const std::vector<SurfaceTopologyEdge>& edges,const std::vector<SurfaceSeamPatch>& patches):data_(new Implementation){
    if(points<0||meshFaces<0||edges.size()!=patches.size()||edges.size()>std::size_t(INT_MAX))throw std::invalid_argument("resident surface seam shape mismatch");
    for(int f:surfaceFaces)if(f<0||f>=meshFaces)throw std::invalid_argument("resident surface face index mismatch");
    for(std::size_t i=0;i<edges.size();++i){const auto e=edges[i];const auto p=patches[i];
        if(e.a<0||e.a>=points||e.b<0||e.b>=points||e.owner<0||std::size_t(e.owner)>=surfaceFaces.size()
            ||e.neighbour< -1||(e.neighbour>=0&&std::size_t(e.neighbour)>=surfaceFaces.size())
            ||(e.neighbour>=0&&(e.neighbourA<0||e.neighbourA>=points||e.neighbourB<0||e.neighbourB>=points))
            ||p.face< -1||p.face>=meshFaces||(p.face>=0&&(p.partner<0||p.partner>=meshFaces)))
            throw std::invalid_argument("resident surface seam addressing mismatch");
    }
    auto& d=*data_;d.points=points;d.edges=edges.size();d.install(d.surfaceFaces,surfaceFaces);d.install(d.initial,edges);d.install(d.patches,patches);
    d.working.reserve(d.edges);d.removed.reserve(d.edges);d.output.reserve(d.edges);d.status.reserve(1);d.bounds.reserve((points+127)/128);
}
ResidentSurfaceSeams::~ResidentSurfaceSeams()=default;
SurfaceSeamDeviceView ResidentSurfaceSeams::view()const{const auto& d=*data_;return {d.edges,d.output.pointer,d.status.pointer};}
void ResidentSurfaceSeams::prepare(const Point* points,const FaceResult* faces,void* streamHandle){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(streamHandle);
    resetSurfaceSeams<<<d.edges?(d.edges+127)/128:1,128,0,stream>>>(d.edges,d.initial.pointer,d.working.pointer,d.removed.pointer,d.status.pointer);checked(cudaGetLastError());
    if(d.edges){geometryBounds<<<(d.points+127)/128,128,0,stream>>>(d.points,points,d.bounds.pointer);checked(cudaGetLastError());
        geometryScale<<<1,128,0,stream>>>((d.points+127)/128,d.bounds.pointer,d.status.pointer);checked(cudaGetLastError());
        matchResidentSurfaceSeams<<<1,128,0,stream>>>(d.edges,d.surfaceFaces.pointer,d.patches.pointer,points,faces,d.working.pointer,d.removed.pointer,d.output.pointer,d.status.pointer);checked(cudaGetLastError());}
}
