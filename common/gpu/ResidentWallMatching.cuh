// Included inside ugkp::gpu::geometry, after the common allocation helpers.
namespace {
__global__ void combineWallConstraints(int count,const unsigned char* explicitMask,
    const unsigned char* physicalWallOwner,unsigned char* output){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<count)output[c]=explicitMask[c]||physicalWallOwner[c];
}
}
struct ResidentWallMatching::Implementation {
    int cells=0,faces=0,rows=0;bool reconstruct=false;
    TopologyView topology{};
    Array<int> owner,neighbour,cellOffsets,cellFaces,triangleOffsets,walls,ring,polygonFailure;
    Array<unsigned char> physicalWall,physicalOwner,defaultConstrained,defaultGas,constrained;
    Array<ugkwp::gaswall::WallPolygonTriangle> triangles;
    Array<ugkwp::gaswall::WallPolygonMetric> metrics;
    Array<ugkwp::gaswall::WallMatchingGeometry> result;
    Array<ugkwp::gaswall::WallMatchingStatus> status;
    template<class T> static void install(Array<T>& target,const std::vector<T>& input){
        target.reserve(input.size());if(!input.empty())checked(cudaMemcpy(target.pointer,input.data(),input.size()*sizeof(T),cudaMemcpyHostToDevice));
    }
};
ResidentWallMatching::ResidentWallMatching(const WallMatchingDefinition& definition,TopologyView topology,bool reconstruct):data_(new Implementation){
    auto& d=*data_;d.cells=definition.cells;d.faces=definition.owner.size();d.rows=definition.walls.size();d.reconstruct=reconstruct;d.topology=topology;
    if(d.cells<0||definition.faceOffsets.size()!=std::size_t(d.faces)+1||definition.neighbour.size()!=std::size_t(d.faces)
        ||definition.physicalWall.size()!=std::size_t(d.faces)||(!definition.constrained.empty()&&definition.constrained.size()!=std::size_t(d.cells))
        ||(!definition.gasCell.empty()&&definition.gasCell.size()!=std::size_t(d.cells)))throw std::invalid_argument("resident wall topology shape mismatch");
    std::vector<int> offsets(d.cells+1,0),triangleOffsets(1,0);std::vector<unsigned char> wallOwner(d.cells,0);
    for(int f=0;f<d.faces;++f){const int owner=definition.owner[f],neighbour=definition.neighbour[f];
        const int vertices=definition.faceOffsets[f+1]-definition.faceOffsets[f];
        if(vertices<3||owner<0||owner>=d.cells||neighbour< -1||neighbour>=d.cells||neighbour==owner)throw std::invalid_argument("resident wall topology addressing mismatch");
        ++offsets[owner+1];if(neighbour>=0)++offsets[neighbour+1];
        if(definition.physicalWall[f])wallOwner[owner]=1;
        triangleOffsets.push_back(triangleOffsets.back()+vertices-2);
    }
    for(int f:definition.walls)if(f<0||f>=d.faces)throw std::invalid_argument("resident wall face index mismatch");
    for(int c=0;c<d.cells;++c)offsets[c+1]+=offsets[c];std::vector<int> cursor=offsets,cellFaces(offsets.back());
    // Ascending face traversal reproduces the native geometry builder's CSR.
    for(int f=0;f<d.faces;++f){cellFaces[cursor[definition.owner[f]]++]=f;if(definition.neighbour[f]>=0)cellFaces[cursor[definition.neighbour[f]]++]=f;}
    d.install(d.owner,definition.owner);d.install(d.neighbour,definition.neighbour);d.install(d.cellOffsets,offsets);d.install(d.cellFaces,cellFaces);
    d.install(d.triangleOffsets,triangleOffsets);d.install(d.walls,definition.walls);d.install(d.physicalWall,definition.physicalWall);d.install(d.physicalOwner,wallOwner);
    d.install(d.defaultConstrained,definition.constrained.empty()?std::vector<unsigned char>(d.cells,0):definition.constrained);d.install(d.defaultGas,definition.gasCell);
    d.constrained.reserve(d.cells);d.ring.reserve(definition.faceOffsets.back());d.polygonFailure.reserve(1);
    d.triangles.reserve(triangleOffsets.back());d.metrics.reserve(d.faces);d.result.reserve(d.rows);d.status.reserve(1);
    checked(cudaStreamSynchronize(nullptr)); // Publish completed initialization to arbitrary consumer streams.
}
ResidentWallMatching::~ResidentWallMatching()=default;
WallMatchingDeviceView ResidentWallMatching::view()const{const auto& d=*data_;return {d.rows,d.result.pointer,d.status.pointer};}
void ResidentWallMatching::prepare(const Point* points,const CellResult* cells,const unsigned char* suppliedConstraints,
    const unsigned char* suppliedGas,void* streamHandle){
    using namespace ugkwp::gaswall;auto& d=*data_;const auto stream=static_cast<cudaStream_t>(streamHandle);
    const WallPolygonTopology polygons{d.faces,d.topology.faceOffsets,d.topology.facePoints,d.triangleOffsets.pointer};
    const WallPolygonWorkspace workspace{d.ring.pointer,d.triangles.pointer,d.metrics.pointer,d.polygonFailure.pointer};
    resetResidentWallPolygonStatusKernel<<<1,1,0,stream>>>(d.faces,d.polygonFailure.pointer);checked(cudaGetLastError());
    if(d.faces){rebuildResidentWallPolygonsKernel<<<(d.faces+127)/128,128,0,stream>>>(polygons,points,workspace);checked(cudaGetLastError());}
    if(d.cells){combineWallConstraints<<<(d.cells+127)/128,128,0,stream>>>(d.cells,suppliedConstraints?suppliedConstraints:d.defaultConstrained.pointer,d.physicalOwner.pointer,d.constrained.pointer);checked(cudaGetLastError());}
    resetResidentWallMatchingStatusKernel<<<1,1,0,stream>>>(d.rows,d.polygonFailure.pointer,d.faces,d.status.pointer);checked(cudaGetLastError());
    const WallMatchingTopology topology{polygons,d.cells,d.cellOffsets.pointer,d.cellFaces.pointer,d.owner.pointer,d.neighbour.pointer,
        d.physicalWall.pointer,d.constrained.pointer,suppliedGas?suppliedGas:d.defaultGas.pointer};
    if(d.rows){rebuildResidentWallMatchingKernel<<<(d.rows+127)/128,128,0,stream>>>(d.rows,d.walls.pointer,topology,points,cells,workspace,d.reconstruct,d.result.pointer,d.status.pointer);checked(cudaGetLastError());}
    finishResidentWallMatchingStatusKernel<<<1,1,0,stream>>>(d.rows,d.result.pointer,d.status.pointer);checked(cudaGetLastError());
}
