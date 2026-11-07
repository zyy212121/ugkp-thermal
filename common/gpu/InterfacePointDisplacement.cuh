// Included by PolyhedralGeometry.cu; shares geometry storage utilities.
#include "InterfacePointDisplacement.H"
namespace ugkp { namespace gpu { namespace geometry { namespace {
__device__ double interfaceThickness(InterfaceThicknessField field,int i){
    return *reinterpret_cast<const double*>(reinterpret_cast<const char*>(field.first)+std::size_t(i)*field.stride);
}
__global__ void beginInterfaceDisplacement(int* changed){*changed=0;}
__global__ void interfaceDisplacementChanged(int points,int faces,const InterfacePointDisplacementInput* input,int* changed){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;const auto p=*input;
    if(i<points&&!(magnitude(p.solidEndpoint[i]-p.oldSolid[i])==0))atomicExch(changed,1);
    if(i<faces&&!(interfaceThickness(p.filmThickness,i)==interfaceThickness(p.previousFilmThickness,i)))atomicExch(changed,1);
}
__global__ void interfacePointNormals(int points,const int* offsets,const int* incident,
    const InterfacePointDisplacementInput* input,Point* normals){
    const int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=points)return;Point normal;
    for(int k=offsets[p];k<offsets[p+1];++k)normal+=input->solidArea[incident[k]];
    normals[p]=normalized(normal);
}
__global__ void interfacePointDisplacement(int points,const int* solidVertex,const Point* normals,
    const InterfacePointDisplacementInput* input,Point* displacement){
    const int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=points)return;const int s=solidVertex[g];const auto p=*input;
    displacement[g]=s<0?Point{}:p.solidEndpoint[s]+normals[s]*interfaceThickness(p.pointThickness,s)-p.oldGas[g];
}
}
struct ResidentInterfacePointDisplacement::Implementation {
    int solidPoints=0,gasPoints=0,faces=0;
    Array<int> offsets,incident,solidVertex,changed;Array<Point> normals,displacement;
};
ResidentInterfacePointDisplacement::ResidentInterfacePointDisplacement(const Topology& solid,int solidPoints,int gasPoints,
    const std::vector<int>& faces,const std::vector<int>& solidToGas):data_(new Implementation){
    if(solidPoints<0||gasPoints<0||solidToGas.size()!=std::size_t(solidPoints)||solid.faceOffsets.empty())
        throw std::invalid_argument("interface point displacement topology shape mismatch");
    std::vector<int> offsets(solidPoints+1,0),reverse(gasPoints,-1);
    for(int s=0;s<solidPoints;++s){const int g=solidToGas[s];if(g< -1||g>=gasPoints||(g>=0&&reverse[g]>=0))
        throw std::invalid_argument("interface point displacement needs canonical vertex correspondence");
        if(g>=0)reverse[g]=s;}
    for(int f:faces){if(f<0||std::size_t(f+1)>=solid.faceOffsets.size())throw std::invalid_argument("interface face outside topology");
        for(int k=solid.faceOffsets[f];k<solid.faceOffsets[f+1];++k){const int p=solid.facePoints[k];
            if(p<0||p>=solidPoints)throw std::invalid_argument("interface vertex outside topology");++offsets[p+1];}}
    for(int p=1;p<=solidPoints;++p)offsets[p]+=offsets[p-1];
    std::vector<int> incident(offsets.back()),cursor=offsets;
    for(int f:faces)for(int k=solid.faceOffsets[f];k<solid.faceOffsets[f+1];++k)incident[cursor[solid.facePoints[k]]++]=f;
    auto& d=*data_;d.solidPoints=solidPoints;d.gasPoints=gasPoints;d.faces=faces.size();
    PinnedTransferQueue transfer;PinnedTransferQueue::Operation pending(transfer);
    d.offsets.upload(offsets,transfer);d.incident.upload(incident,transfer);d.solidVertex.upload(reverse,transfer);
    d.changed.reserve(1);d.normals.reserve(solidPoints);d.displacement.reserve(gasPoints);transfer.finish();
}
ResidentInterfacePointDisplacement::~ResidentInterfacePointDisplacement()=default;
InterfacePointDisplacementView ResidentInterfacePointDisplacement::view()const{return {data_->displacement.pointer,data_->changed.pointer};}
void ResidentInterfacePointDisplacement::evaluate(const InterfacePointDisplacementInput* input,void* rawStream){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(rawStream);beginInterfaceDisplacement<<<1,1,0,stream>>>(d.changed.pointer);
    const int count=std::max(d.solidPoints,d.faces);
    if(count)interfaceDisplacementChanged<<<(count-1)/128+1,128,0,stream>>>(d.solidPoints,d.faces,input,d.changed.pointer);
    if(d.solidPoints)interfacePointNormals<<<(d.solidPoints-1)/128+1,128,0,stream>>>(d.solidPoints,d.offsets.pointer,d.incident.pointer,input,d.normals.pointer);
    if(d.gasPoints)interfacePointDisplacement<<<(d.gasPoints-1)/128+1,128,0,stream>>>(d.gasPoints,d.solidVertex.pointer,d.normals.pointer,input,d.displacement.pointer);
    checked(cudaGetLastError());
}
}}}
