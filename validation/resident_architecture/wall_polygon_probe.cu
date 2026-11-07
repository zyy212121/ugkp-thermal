#include "wall_polygon_probe.H"
#include "common/gasWall/ResidentWallPolygons.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
namespace wallPolygonProbe {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> struct Device {
    T* pointer=nullptr;explicit Device(const std::vector<T>& values){if(values.empty())return;
        check(cudaMalloc(&pointer,values.size()*sizeof(T)));check(cudaMemcpy(pointer,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice));}
    ~Device(){if(pointer)cudaFree(pointer);}
};
template<class T> std::vector<T> download(const T* data,int n){std::vector<T> out(n);if(n)check(cudaMemcpy(out.data(),data,n*sizeof(T),cudaMemcpyDeviceToHost));return out;}
std::vector<std::vector<Result>> evaluate(const std::vector<std::array<double,3>>& coordinates,const std::vector<int>& offsets,const std::vector<int>& indices){
    using namespace ugkwp::gaswall;using ugkp::gpu::geometry::Point;
    const int faces=offsets.size()-1;std::vector<int> triOffsets(1,0);
    for(int f=0;f<faces;++f)triOffsets.push_back(triOffsets.back()+offsets[f+1]-offsets[f]-2);
    std::vector<Point> points;for(auto p:coordinates)points.push_back({p[0],p[1],p[2]});
    Device<Point> dPoints(points);Device<int> dOffsets(offsets),dIndices(indices),dTriOffsets(triOffsets),dRing(std::vector<int>(indices.size())),dFirst(std::vector<int>(1));
    Device<WallPolygonTriangle> dTriangles(std::vector<WallPolygonTriangle>(triOffsets.back()));Device<WallPolygonMetric> dMetric{std::vector<WallPolygonMetric>(faces)};
    WallPolygonTopology topology{faces,dOffsets.pointer,dIndices.pointer,dTriOffsets.pointer};
    WallPolygonWorkspace workspace{dRing.pointer,dTriangles.pointer,dMetric.pointer,dFirst.pointer};
    cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t executable;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    resetResidentWallPolygonStatusKernel<<<1,1,0,stream>>>(faces,dFirst.pointer);
    if(faces)rebuildResidentWallPolygonsKernel<<<(faces+127)/128,128,0,stream>>>(topology,dPoints.pointer,workspace);
    check(cudaGetLastError());check(cudaStreamEndCapture(stream,&graph));
    std::size_t count=0;check(cudaGraphGetNodes(graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(graph,nodes.data(),&count));
    for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type==cudaGraphNodeTypeHost||type==cudaGraphNodeTypeMemcpy)throw std::runtime_error("polygon graph has host traffic");}
    check(cudaGraphInstantiateWithFlags(&executable,graph,0));std::vector<std::vector<Result>> result(3);
    for(int replay=0;replay<3;++replay){std::vector<Point> sample=points;
        if(replay==1)for(auto& p:sample){const double x=p.x;p.x=-p.y*.5;p.y=x*.5;p.z*=.5;}
        if(replay==1)for(int f=0;f<faces;f+=13){const Point origin=sample[indices[offsets[f]]];for(int k=offsets[f];k<offsets[f+1];++k)sample[indices[k]]=origin;}
        if(!sample.empty())check(cudaMemcpyAsync(dPoints.pointer,sample.data(),sample.size()*sizeof(Point),cudaMemcpyHostToDevice,stream));
        check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
        const auto metrics=download(dMetric.pointer,faces);const auto triangles=download(dTriangles.pointer,triOffsets.back());int first=faces;
        for(int f=0;f<faces;++f){const auto m=metrics[f];Result out;out.normal=m.normal;out.centre=m.centre;out.scale=m.scale;out.valid=m.valid;
            if(!m.valid)first=std::min(first,f);
            for(int k=0;k<m.triangles;++k){const auto t=triangles[triOffsets[f]+k];out.triangles.push_back({t.a,t.b,t.c});}result[replay].push_back(out);}
        if(download(dFirst.pointer,1)[0]!=first)throw std::runtime_error("polygon first failure differs");
    }
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));return result;
}
}
