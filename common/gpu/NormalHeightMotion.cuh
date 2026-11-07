// Included by PolyhedralGeometry.cu; shares its allocation/error utilities.
#include "NormalHeightMotion.H"
namespace ugkp { namespace gpu { namespace geometry { namespace {
__global__ void normalHeightWalls(int count,TopologyView topology,const int* faces,
    const NormalHeightMotionInput* input,WallNormalHeightGeometry* geometry){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){const auto p=*input;
        geometry[i]=wallNormalHeightGeometry(topology,faces[i],p.reference,p.points,p.displacement);}
}
__global__ void normalHeightPoints(int count,const int* offsets,const int* incident,
    const NormalHeightMotionInput* input,const WallNormalHeightGeometry* geometry,
    MotionPenalty* penalties,double* partial){
    const int p=blockIdx.x*blockDim.x+threadIdx.x;const auto in=*input;
    double minimum=std::numeric_limits<double>::infinity();MotionPenalty penalty;
    if(p<count){if(!in.fixed[p])for(int k=offsets[p];k<offsets[p+1];++k){
        const double height=addWallNormalHeight(penalty,geometry[incident[k]],in.reference[p],in.points[p]);
        if(height<minimum)minimum=height;}
        penalties[p]=penalty;
    }
    __shared__ double values[128];values[threadIdx.x]=minimum;__syncthreads();
    for(int stride=64;stride;stride/=2){if(threadIdx.x<stride&&values[threadIdx.x+stride]<values[threadIdx.x])
        values[threadIdx.x]=values[threadIdx.x+stride];__syncthreads();}
    if(!threadIdx.x)partial[blockIdx.x]=values[0];
}
__global__ void normalHeightControl(int blocks,const double* partial,const NormalHeightMotionInput* input,
    NormalHeightMotionControl* out){
    double minimum=std::numeric_limits<double>::infinity();
    for(int i=threadIdx.x;i<blocks;i+=blockDim.x)if(partial[i]<minimum)minimum=partial[i];
    __shared__ double values[128];values[threadIdx.x]=minimum;__syncthreads();
    for(int stride=64;stride;stride/=2){if(threadIdx.x<stride&&values[threadIdx.x+stride]<values[threadIdx.x])
        values[threadIdx.x]=values[threadIdx.x+stride];__syncthreads();}
    if(!threadIdx.x)*out={values[0],normalHeightMotionTolerance(input->geometricTolerance,values[0])};
}
}
struct ResidentNormalHeightMotion::Implementation {
    int points=0,walls=0,blocks=0;
    Array<int> faceOffsets,facePoints,faces,offsets,incident;
    Array<WallNormalHeightGeometry> geometry;
    Array<MotionPenalty> penalties;Array<double> partial;Array<NormalHeightMotionControl> control;
    TopologyView topology()const{return {faceOffsets.pointer,facePoints.pointer,nullptr,nullptr,nullptr,nullptr,nullptr};}
};
ResidentNormalHeightMotion::ResidentNormalHeightMotion(const Topology& topology,int count,
    const std::vector<int>& owner,const std::vector<int>& wallFaces):data_(new Implementation){
    if(count<0||topology.faceOffsets.size()!=owner.size()+1||topology.vertexOffsets.empty())
        throw std::invalid_argument("normal-height motion topology shape mismatch");
    std::vector<int> faces;for(int face:wallFaces)if(face>=0)faces.push_back(face);
    std::vector<int> offsets(count+1,0);
    for(int f:faces){const int cell=owner[f];
        for(int k=topology.vertexOffsets[cell];k<topology.vertexOffsets[cell+1];++k)++offsets[topology.vertices[k]+1];}
    for(int p=1;p<=count;++p)offsets[p]+=offsets[p-1];
    std::vector<int> incident(offsets.back()),cursor=offsets;
    for(std::size_t i=0;i<faces.size();++i){const int cell=owner[faces[i]];
        for(int k=topology.vertexOffsets[cell];k<topology.vertexOffsets[cell+1];++k)incident[cursor[topology.vertices[k]]++]=i;}
    auto& d=*data_;d.points=count;d.walls=faces.size();d.blocks=count?(count-1)/128+1:0;
    PinnedTransferQueue transfer;PinnedTransferQueue::Operation pending(transfer);
    d.faceOffsets.upload(topology.faceOffsets,transfer);d.facePoints.upload(topology.facePoints,transfer);
    d.faces.upload(faces,transfer);d.offsets.upload(offsets,transfer);d.incident.upload(incident,transfer);
    d.geometry.reserve(d.walls);d.penalties.reserve(count);d.partial.reserve(d.blocks);d.control.reserve(1);transfer.finish();
}
ResidentNormalHeightMotion::~ResidentNormalHeightMotion()=default;
NormalHeightMotionView ResidentNormalHeightMotion::view()const{
    const auto& d=*data_;return {d.penalties.pointer,d.control.pointer};
}
void ResidentNormalHeightMotion::evaluate(const NormalHeightMotionInput* input,void* rawStream){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(rawStream);
    if(d.walls)normalHeightWalls<<<(d.walls-1)/128+1,128,0,stream>>>(d.walls,d.topology(),d.faces.pointer,input,d.geometry.pointer);
    if(d.points)normalHeightPoints<<<d.blocks,128,0,stream>>>(d.points,d.offsets.pointer,d.incident.pointer,input,
        d.geometry.pointer,d.penalties.pointer,d.partial.pointer);
    normalHeightControl<<<1,128,0,stream>>>(d.blocks,d.partial.pointer,input,d.control.pointer);checked(cudaGetLastError());
}
}}}
