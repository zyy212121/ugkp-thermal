#include "wall_matching_probe.H"
#include "common/gpu/ResidentWallMatching.H"
#include "common/gasWall/WallGeometryPublication.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
namespace wallMatchingProbe {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> struct Device {
    T* pointer=nullptr;explicit Device(const std::vector<T>& v){if(v.empty())return;check(cudaMalloc(&pointer,v.size()*sizeof(T)));check(cudaMemcpy(pointer,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
    ~Device(){if(pointer)cudaFree(pointer);}
};
template<class T> std::vector<T> download(const T* p,int n){std::vector<T> out(n);if(n)check(cudaMemcpy(out.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return out;}
std::vector<Result> evaluate(const Mesh& m,bool reconstructed){
    using namespace ugkwp::gaswall;using namespace ugkp::gpu::geometry;
    const int nf=m.owner.size(),nc=m.centres.size(),nr=m.walls.size();
    std::vector<unsigned char> constrained=m.constrained;if(constrained.empty())constrained.resize(nc,0);
    for(int f=0;f<nf;++f)if(m.physicalWall[f])constrained[m.owner[f]]=1;
    std::vector<Point> points;for(auto p:m.points)points.push_back({p[0],p[1],p[2]});
    std::vector<CellResult> cells(nc);for(int c=0;c<nc;++c){const auto p=m.centres[c];cells[c].centre={p[0],p[1],p[2]};cells[c].volume=m.volumes[c];}
    Device<Point> dPoints(points);Device<CellResult> dCells(cells);
    Device<int> offsets(m.offsets),indices(m.indices);Device<unsigned char> mask(constrained),gasMask(m.gasCell.empty()?std::vector<unsigned char>(nc,1):m.gasCell);
    WallMatchingDefinition definition{nc,m.offsets,m.owner,m.neighbour,m.walls,m.physicalWall,m.constrained,m.gasCell};
    TopologyView topology{};topology.faceOffsets=offsets.pointer;topology.facePoints=indices.pointer;
    ResidentWallMatching owner(definition,topology,reconstructed);const auto view=owner.view();
    std::vector<WallInput<double,2>> initial(nr);for(auto& input:initial){input.temperature=-1729;input.matchingDistance=-17;}
    Device<WallInput<double,2>> wallInput(initial);Device<int> donor(std::vector<int>(nr,-1));Device<double> weights(std::vector<double>(nr,-1));
    auto previousInput=initial;std::vector<int> previousDonor(nr,-1);std::vector<double> previousWeight(nr,-1);

    cudaStream_t stream;cudaGraph_t graphs[2]{};cudaGraphExec_t executables[2]{};check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    for(int supplied=0;supplied<2;++supplied){
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        owner.prepare(dPoints.pointer,dCells.pointer,supplied?mask.pointer:nullptr,supplied?gasMask.pointer:nullptr,stream);
        if(nr&&!reconstructed)publishCellAverageWallGeometryKernel<<<(nr+127)/128,128,0,stream>>>(nr,view.geometry,view.status,wallInput.pointer,donor.pointer,weights.pointer,nullptr);
        check(cudaGetLastError());check(cudaStreamEndCapture(stream,&graphs[supplied]));
        std::size_t count=0;check(cudaGraphGetNodes(graphs[supplied],nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(graphs[supplied],nodes.data(),&count));
        for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type==cudaGraphNodeTypeHost||type==cudaGraphNodeTypeMemcpy)throw std::runtime_error("wall matching graph contains host communication");}
        check(cudaGraphInstantiateWithFlags(&executables[supplied],graphs[supplied],0));
    }
    std::vector<Result> result(3);
    for(int replay=0;replay<3;++replay){auto active=constrained;if(replay==1)std::fill(active.begin(),active.end(),1);
        if(nc)check(cudaMemcpyAsync(mask.pointer,active.data(),nc,cudaMemcpyHostToDevice,stream));
        check(cudaGraphLaunch(executables[replay==1],stream));check(cudaStreamSynchronize(stream));const auto state=download(view.status,1)[0];
        auto& r=result[replay];r.firstInvalidRow=state.firstInvalidRow;r.invalidFace=state.invalidFace;
        for(const auto& d:download(view.geometry,nr))r.row.push_back({d.wallFace,d.ownerCell,d.donor,d.failureFace,d.origin,d.normal,d.tangent1,d.tangent2,d.matchingPoint,d.firstMomentVector,
            d.ownerDistance,d.matchingDistance,d.maximumOwnerDistance,d.matchingDonorDistance,d.volume,d.geometryVolume,d.firstMoment});
        const auto inputs=download(wallInput.pointer,nr);const auto currentWeights=download(weights.pointer,nr);
        const auto currentDonor=download(donor.pointer,nr);
        for(int row=0;row<nr;++row){const auto& in=inputs[row];
            if(!reconstructed&&state.invalidFace<0&&state.firstInvalidRow==nr){const auto& d=r.row[row];
                if(in.matchingDistance!=d.matchingDistance||in.ownerDistance!=d.ownerDistance||in.quadrature.volume!=d.volume||in.quadrature.firstMoment!=d.firstMoment
                    ||in.temperature!=0||in.quadrature.count!=0||in.quadrature.distance||in.quadrature.volumeWeight||in.diagnostics||currentDonor[row]!=d.donor||currentWeights[row]!=1)
                    throw std::runtime_error("wall input publication differs from native descriptor");
                for(int k=0;k<3;++k)if(in.normal[k]!=d.normal[k])throw std::runtime_error("wall normal publication differs");
                previousInput[row]=in;previousDonor[row]=currentDonor[row];previousWeight[row]=currentWeights[row];
            }else if(in.temperature!=previousInput[row].temperature||in.matchingDistance!=previousInput[row].matchingDistance
                ||currentDonor[row]!=previousDonor[row]||currentWeights[row]!=previousWeight[row])throw std::runtime_error("rejected or reconstructed candidate changed cell-average geometry");
        }

    }
    for(int i=0;i<2;++i){check(cudaGraphExecDestroy(executables[i]));check(cudaGraphDestroy(graphs[i]));}check(cudaStreamDestroy(stream));return result;
}
}
