// Included by PolyhedralGeometry.cu.
#include "SweepPlanarity.H"
#include <cub/device/device_scan.cuh>
namespace ugkp { namespace gpu { namespace geometry { namespace {
__global__ void classifySweepPlanes(int faces,TopologyView topology,const int* candidateOffset,
    const SweepPlanarityInput* input,PlanarConstraint* candidate,double* tolerance,int* selected){
    const int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=faces)return;
    const int begin=topology.faceOffsets[f],end=topology.faceOffsets[f+1];
    if(end-begin<=3)return;
    const auto points=input->localPoints;
    const int a=topology.facePoints[begin],b=topology.facePoints[begin+1],c=topology.facePoints[begin+2];
    double length=0;
    for(int k=begin+1;k<end;++k){const double distance=magnitude(points[topology.facePoints[k]]-points[a]);length=length>distance?length:distance;}
    const double scale=magnitude(cross(points[b]-points[a],points[c]-points[a]))*length;
    const double uncertainty=64*std::numeric_limits<double>::epsilon()*scale+8*input->coordinateUncertainty*length*length;
    for(int k=begin+3;k<end;++k){
        const int d=topology.facePoints[k],slot=candidateOffset[f]+k-begin-3;
        const double volume=triple(points[b]-points[a],points[c]-points[a],points[d]-points[a]);
        const double absolute=volume<0?-volume:volume;
        selected[slot]=scale>0&&absolute<=uncertainty;
        if(selected[slot]){candidate[slot]={a,b,c,d,scale};
            tolerance[slot]=32*std::numeric_limits<double>::epsilon()+8*input->coordinateUncertainty*length*length/scale;}
    }
}
__global__ void endSweepPlaneFlags(int candidates,int* selected){selected[candidates]=0;}
__global__ void compactSweepPlanes(int candidates,const int* selected,const int* offsets,
    const PlanarConstraint* candidate,const double* inputTolerance,PlanarConstraint* output,double* tolerance,int* count){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(!i)*count=offsets[candidates];
    if(i<candidates&&selected[i]){output[offsets[i]]=candidate[i];tolerance[offsets[i]]=inputTolerance[i];}
}
}
struct ResidentSweepPlanarity::Implementation {
    int faces=0,candidates=0;std::size_t scratchBytes=0;
    Array<int> offsets,vertices,candidateOffsets,selected,prefix,count;
    Array<PlanarConstraint> candidate,planar;Array<double> candidateTolerance,tolerance;Array<unsigned char> scratch;
    TopologyView topology()const{return {offsets.pointer,vertices.pointer,nullptr,nullptr,nullptr,nullptr,nullptr};}
};
ResidentSweepPlanarity::ResidentSweepPlanarity(const Topology& topology):data_(new Implementation){
    if(topology.faceOffsets.empty()||topology.faceOffsets.front()!=0
        ||topology.faceOffsets.back()!=int(topology.facePoints.size()))throw std::invalid_argument("sweep planarity topology shape mismatch");
    auto& d=*data_;d.faces=topology.faceOffsets.size()-1;std::vector<int> offsets(1,0);
    for(int f=0;f<d.faces;++f){const int vertices=topology.faceOffsets[f+1]-topology.faceOffsets[f];
        if(vertices<3)throw std::invalid_argument("sweep planarity requires validated polygon topology");
        const auto next=std::size_t(offsets.back())+std::size_t(vertices-3);
        if(next>=std::size_t(INT_MAX))throw std::length_error("sweep planarity index overflow");offsets.push_back(next);}
    d.candidates=offsets.back();PinnedTransferQueue transfer;PinnedTransferQueue::Operation pending(transfer);
    d.offsets.upload(topology.faceOffsets,transfer);d.vertices.upload(topology.facePoints,transfer);d.candidateOffsets.upload(offsets,transfer);
    d.selected.reserve(std::size_t(d.candidates)+1);d.prefix.reserve(std::size_t(d.candidates)+1);d.count.reserve(1);
    d.candidate.reserve(d.candidates);d.planar.reserve(d.candidates);d.candidateTolerance.reserve(d.candidates);d.tolerance.reserve(d.candidates);
    checked(cub::DeviceScan::ExclusiveSum(nullptr,d.scratchBytes,d.selected.pointer,d.prefix.pointer,d.candidates+1));
    d.scratch.reserve(d.scratchBytes);transfer.finish();
}
ResidentSweepPlanarity::~ResidentSweepPlanarity()=default;
int ResidentSweepPlanarity::capacity()const{return data_->candidates;}
SweepPlanarityView ResidentSweepPlanarity::view()const{const auto& d=*data_;return {d.planar.pointer,d.tolerance.pointer,d.count.pointer};}
void ResidentSweepPlanarity::evaluate(const SweepPlanarityInput* input,void* rawStream){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(rawStream);
    if(d.faces)classifySweepPlanes<<<(d.faces-1)/128+1,128,0,stream>>>(d.faces,d.topology(),d.candidateOffsets.pointer,input,
        d.candidate.pointer,d.candidateTolerance.pointer,d.selected.pointer);
    endSweepPlaneFlags<<<1,1,0,stream>>>(d.candidates,d.selected.pointer);checked(cudaGetLastError());
    auto bytes=d.scratchBytes;checked(cub::DeviceScan::ExclusiveSum(d.scratch.pointer,bytes,d.selected.pointer,d.prefix.pointer,d.candidates+1,stream));
    compactSweepPlanes<<<d.candidates?(d.candidates-1)/128+1:1,128,0,stream>>>(d.candidates,d.selected.pointer,d.prefix.pointer,
        d.candidate.pointer,d.candidateTolerance.pointer,d.planar.pointer,d.tolerance.pointer,d.count.pointer);checked(cudaGetLastError());
}
}}}
