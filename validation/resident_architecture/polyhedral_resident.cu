// Native resident geometry: change points and consume all results on device
// across captured replays; publish once, then compare to the ordered host math.
#include "common/gpu/PolyhedralGeometry.H"
#include "common/gpu/ExtendedGeometryMath.H"
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>
using namespace ugkp::gpu::geometry;
void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::abort();}}
template<class T> struct Buffer {
    T* p=nullptr;int n=0;
    explicit Buffer(int count):n(count){if(n)check(cudaMalloc(&p,n*sizeof(T)));}
    ~Buffer(){if(p)check(cudaFree(p));}
    void upload(const std::vector<T>& v){assert(int(v.size())==n);if(n)check(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> read(){std::vector<T> v(n);if(n)check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
__host__ __device__ Point moved(Point p,int phase,double scale){
    return {p.x+phase*.00390625*scale,p.y+phase*.001953125*p.z,p.z*(1+phase*.0009765625)};
}
__global__ void coordinates(int n,const Point* base,Point* old,Point* end,const int* phase,double scale){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){old[i]=moved(base[i],*phase,scale);end[i]=moved(base[i],*phase+1,scale);}
}
__global__ void archive(PolyhedralGeometryDeviceView v,const int* phase,FaceResult* faces,CellResult* cells,
    WallDistanceResult* distance,SweepResult* sweeps){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<v.faces){faces[*phase*v.faces+i]=v.face[i];sweeps[*phase*v.faces+i]=v.sweptFaces[i];}
    if(i<v.cells){cells[*phase*v.cells+i]=v.cell[i];distance[*phase*v.cells+i]=v.distance[i];}
}
__global__ void advance(int* phase){++*phase;}
void close(double a,double b){assert(std::isfinite(a)&&std::isfinite(b));assert(std::abs(a-b)<=2e-13*std::max({1.,std::abs(a),std::abs(b)}));}
void close(Point a,Point b){close(a.x,b.x);close(a.y,b.y);close(a.z,b.z);}
void fixture(int count,bool tetra,double scale,bool skew,int wallMode){
    Topology t;t.faceOffsets={0};t.cellOffsets={0};t.vertexOffsets={0};std::vector<Point> base;
    const std::vector<Point> shape=tetra?std::vector<Point>{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}
        :std::vector<Point>{{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}};
    const std::vector<std::vector<int>> faces=tetra?std::vector<std::vector<int>>{{0,2,1},{0,1,3},{1,2,3},{2,0,3}}
        :std::vector<std::vector<int>>{{0,3,2,1},{4,5,6,7},{0,4,7,3},{1,2,6,5},{0,1,5,4},{3,7,6,2}};
    for(int c=0;c<count;++c){const int first=base.size();
        for(auto p:shape){p.x+=2*c;base.push_back({scale*(p.x+(skew?.125*p.y:0)),scale*(p.y+(skew?.0625*p.z:0)),scale*p.z});t.vertices.push_back(base.size()-1);}
        t.vertexOffsets.push_back(t.vertices.size());
        for(const auto& row:faces){t.cellFaces.push_back(t.faceOffsets.size()-1);t.cellSigns.push_back(1);
            for(int p:row)t.facePoints.push_back(first+p);t.faceOffsets.push_back(t.facePoints.size());}
        t.cellOffsets.push_back(t.cellFaces.size());
    }
    const int nf=t.faceOffsets.size()-1,np=base.size(),replays=8;
    std::vector<int> walls;if(wallMode)for(int f=nf-1;f>=0;--f)if(wallMode==1||f%3)walls.push_back(f);
    ResidentPolyhedralGeometry geometry(t,walls);const auto view=geometry.view();
    Buffer<Point> points(np),old(np),end(np);points.upload(base);Buffer<int> phase(1);phase.upload({0});
    Buffer<FaceResult> savedFaces(nf*replays);Buffer<CellResult> savedCells(count*replays);
    Buffer<WallDistanceResult> savedDistance(count*replays);Buffer<SweepResult> savedSweeps(nf*replays);
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    coordinates<<<std::max(1,(np+127)/128),128,0,stream>>>(np,points.p,old.p,end.p,phase.p,scale);
    geometry.rebuild(old.p,stream);geometry.sweep(old.p,end.p,stream);
    archive<<<std::max(1,(nf+127)/128),128,0,stream>>>(view,phase.p,savedFaces.p,savedCells.p,savedDistance.p,savedSweeps.p);
    advance<<<1,1,0,stream>>>(phase.p);check(cudaGetLastError());
    cudaGraph_t graph;check(cudaStreamEndCapture(stream,&graph));
    std::size_t nodes=0;check(cudaGraphGetNodes(graph,nullptr,&nodes));std::vector<cudaGraphNode_t> list(nodes);
    check(cudaGraphGetNodes(graph,list.data(),&nodes));for(auto node:list){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));assert(type==cudaGraphNodeTypeKernel);}
    cudaGraphExec_t executable;check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    for(int i=0;i<replays;++i)check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
    const auto actualFaces=savedFaces.read();const auto actualCells=savedCells.read();
    const auto actualDistance=savedDistance.read();const auto actualSweeps=savedSweeps.read();assert(phase.read()[0]==replays);
    TopologyView host{t.faceOffsets.data(),t.facePoints.data(),t.cellOffsets.data(),t.cellFaces.data(),t.cellSigns.data(),t.vertexOffsets.data(),t.vertices.data()};
    for(int iteration=0;iteration<replays;++iteration){std::vector<Point> before,after,areas,nextAreas;
        for(auto p:base){before.push_back(moved(p,iteration,scale));after.push_back(moved(p,iteration+1,scale));}
        for(int f=0;f<nf;++f){const auto expected=faceGeometry(f,host,before.data()),got=actualFaces[iteration*nf+f];
            assert(expected.invalid==got.invalid);close(expected.area,got.area);close(expected.centre,got.centre);areas.push_back(expected.area);
            nextAreas.push_back(faceGeometry(f,host,after.data()).area);
            const auto sweep=faceSweep(f,host,before.data(),after.data()),gotSweep=actualSweeps[iteration*nf+f];
            assert(sweep.invalid==gotSweep.invalid);close(sweep.area,gotSweep.area);close(sweep.volume,gotSweep.volume);
        }
        for(int c=0;c<count;++c){const auto expected=cellGeometry(c,host,before.data(),areas.data()),got=actualCells[iteration*count+c];
            assert(expected.invalid==got.invalid);close(expected.centre,got.centre);close(expected.volume,got.volume);close(expected.closure,got.closure);close(expected.areaSum,got.areaSum);
            const auto next=cellGeometry(c,host,after.data(),nextAreas.data());double sweep=0;
            for(int k=t.cellOffsets[c];k<t.cellOffsets[c+1];++k)sweep+=t.cellSigns[k]*actualSweeps[iteration*nf+t.cellFaces[k]].volume;
            assert(std::abs(next.volume-got.volume-sweep)<=2e-12*std::max(next.volume,got.volume));
            double distance=walls.empty()?0:std::numeric_limits<double>::max();
            for(int f:walls)for(int k=t.faceOffsets[f]+1;k+1<t.faceOffsets[f+1];++k)
                distance=std::min(distance,extended::triangleDistance(expected.centre,before[t.facePoints[t.faceOffsets[f]]],before[t.facePoints[k]],before[t.facePoints[k+1]]));
            assert(actualDistance[iteration*count+c].invalidFace<0);close(distance,actualDistance[iteration*count+c].distance);
        }
    }
    // The host adapter uses the same resident owner but remains an explicit
    // publication boundary. Reconfigure its immutable topology and wall list.
    PolyhedralGeometry adapter;for(const auto& selected:std::vector<std::vector<int>>{walls,{},walls}){
        const auto result=adapter.rebuild(t,base,selected);assert(int(result.cells.size())==count);
        for(int c=0;c<count;++c)close(result.cells[c].volume,actualCells[c].volume);
    }
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
}
int main(){int configurations=0;
    for(int count:{0,1,7,128})for(bool tetra:{false,true})for(double scale:{.001,1.,1024.})for(bool skew:{false,true})for(int walls:{0,1,2}){
        fixture(count,tetra,scale,skew,walls);++configurations;
    }
    std::printf("PASS %d resident geometry configurations, %d captured replays; face/cell/wall/sweep host comparison and GCL; no transfer nodes\n",configurations,8*configurations);
    return 0;
}
