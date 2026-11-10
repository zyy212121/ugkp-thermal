// Compile as C++ for the host or with nvcc for the actual CUDA operators.
// The state adapter supplies a six-plane cell; classification and face-walking
// are included unchanged from production. This is an operator regression, not
// a substitute for a full application/case run.
#include "BoundaryInjectionSampling.H"
#include "thermal/ParticleTrackingGeometry.H"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define TEST_HD __host__ __device__
#else
#define TEST_HD
#define __device__
#define __forceinline__ inline
#endif
#define GPU_OPERATOR_REAL double
#define GPU_OPERATOR_TIME double
#define GPU_OPERATOR_R(x) x
#define GPU_OPERATOR_TINY(x) x
#define GPU_OPERATOR_THERMAL 0
#define GPU_PERIODIC_FACE_KIND 5
#define GPU_PERIODIC_COORDINATE(hit,shift,eps,normal) hit-shift+eps*normal
#define GPU_WALL_COORDINATE(hit,eps,normal) hit-eps*normal
using namespace ugkp::gpu::geometry;
using ugkwp::BoundaryInjectionTriangle;
struct Particle {double x,y,z;int outside,status;};
struct DeviceState {
    int particleCapacity=1,nCells=1,nFaces=6,nInternalFaces=0,maxFaceWalkHops=64;
    int cellPlaneStart[1]{0},cellPlaneCount[1]{6},pStatus[1]{1},pCellId[1]{0};
    int cellFaceKind[6]{3,3,3,3,3,3},cellFaceNeighbor[6]{-1,-1,-1,-1,-1,-1};
    int cellFaceId[6]{0,1,2,3,4,5},facePeriodicPair[6]{-1,-1,-1,-1,-1,-1};
    double cellLength[1]{1},planeNx[6]{},planeNy[6]{},planeNz[6]{},planeD[6]{};
    double cellFaceRestitution[6]{1,1,1,1,1,1};
    double facePeriodicDx[6]{},facePeriodicDy[6]{},facePeriodicDz[6]{};
    double px[1]{},py[1]{},pz[1]{},pux[1]{1},puy[1]{},puz[1]{};
    double puxOld[1]{1},puyOld[1]{},puzOld[1]{};
};
template<class T> __device__ T clampMin(T a,T b){return a>b?a:b;}
template<class T> __device__ T clampRange(T a,T lo,T hi){return a<lo?lo:(a>hi?hi:a);}
__device__ bool isPeriodicFace(const DeviceState& s,int f){
    return f>=s.nInternalFaces&&f<s.nFaces
        &&s.facePeriodicPair[f]>=s.nInternalFaces&&s.facePeriodicPair[f]<s.nFaces;
}
#include "operators/pointInsideCell.cuh"
#include "GpuParticleTransport.cuh"

struct Fixture {
    DeviceState cell;
    BoundaryInjectionTriangle triangles[4]{};
    int count=0;
    Point centre{},shift{1e-6,0,0};
};
__device__ Particle runOne(const Fixture& fixture,int mode,int ordinal) {
    DeviceState s=fixture.cell;
    Point p=fixture.centre+fixture.shift;
    if(mode!=2)ugkwp::sampleBoundaryInjection(fixture.triangles,0,fixture.count,
        ugkwp::injectionPositionSeed(123,17,.125,ordinal),p.x,p.y,p.z);
    s.px[0]=p.x;s.py[0]=p.y;s.pz[0]=p.z;
    const int outside=!pointInsideCell(s,0,p.x,p.y,p.z);
    trackOneParticleLocalFaceWalk<false>(s,0,.001);
    return {s.px[0],s.py[0],s.pz[0],outside,s.pStatus[0]};
}
#ifdef __CUDACC__
__global__ void track(Fixture fixture,int mode,Particle* output,int n){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)output[i]=runOne(fixture,mode,i);
}
#endif

int main(){
    constexpr int n=100000;
    // The alternating normal displacements reproduce the warped unit inlet.
    const Point points[8]={{-.01,0,0},{.01,0,1},{-.01,1,1},{.01,1,0},
        {1,0,0},{1,0,1},{1,1,1},{1,1,0}};
    const int offsets[7]={0,4,8,12,16,20,24};
    const int ids[24]={0,1,2,3, 4,7,6,5, 0,4,5,1, 3,2,6,7, 0,3,7,4, 1,5,6,2};
    TopologyView topology{};topology.faceOffsets=offsets;topology.facePoints=ids;
    Fixture projected;
    for(int f=0;f<6;++f){
        const auto geometry=faceGeometry(f,topology,points);
        const auto plane=ugkp::thermal::particleTrackingPlane(0,0,geometry.area.x,
            geometry.area.y,geometry.area.z,magnitude(geometry.area),
            geometry.centre.x,geometry.centre.y,geometry.centre.z,2.2204460492503131e-16);
        projected.cell.planeNx[f]=plane.x;projected.cell.planeNy[f]=plane.y;
        projected.cell.planeNz[f]=plane.z;projected.cell.planeD[f]=plane.d;
    }
    const auto inlet=faceGeometry(0,topology,points);
    projected.centre=inlet.centre;
    projected.shift=inlet.area*(-1e-6/(magnitude(inlet.area)+2.2204460492503131e-16));
    int scratch[4];
    projected.count=ugkwp::buildBoundaryInjectionTriangles(points,ids,4,scratch,
        projected.triangles,inlet.centre,inlet.area,projected.shift);
    assert(projected.count==2);
    Fixture legacy=projected;
    // The previous ear-clipping diagonal and physical 3D triangle areas.
    double cumulative=0;
    const int oldEars[2][3]={{3,0,1},{3,1,2}};
    for(int k=0;k<2;++k){
        const Point a=points[oldEars[k][0]],ab=points[oldEars[k][1]]-a,
            ac=points[oldEars[k][2]]-a,p=a+projected.shift;
        cumulative+=.5*magnitude(cross(ab,ac));
        legacy.triangles[k]={{p.x,p.y,p.z},{ab.x,ab.y,ab.z},{ac.x,ac.y,ac.z},cumulative};
    }
    for(int mode=0;mode<3;++mode){
        const auto& fixture=mode==0?legacy:projected;
        std::vector<Particle> output(n);
#ifdef __CUDACC__
        Particle* device=nullptr;
        assert(cudaMalloc(&device,n*sizeof(Particle))==cudaSuccess);
        track<<<(n+127)/128,128>>>(fixture,mode,device,n);
        assert(cudaDeviceSynchronize()==cudaSuccess);
        assert(cudaMemcpy(output.data(),device,n*sizeof(Particle),cudaMemcpyDeviceToHost)==cudaSuccess);
        cudaFree(device);
#else
        for(int i=0;i<n;++i)output[i]=runOne(fixture,mode,i);
#endif
        int outside=0,deleted=0;double maxExpectedError=0;
        for(int i=0;i<n;++i){
            const auto& p=output[i];outside+=p.outside;deleted+=!p.status;
            if(mode!=0){
                Point expected=fixture.centre+fixture.shift;
                if(mode==1)assert(ugkwp::sampleBoundaryInjection(fixture.triangles,0,
                    fixture.count,ugkwp::injectionPositionSeed(123,17,.125,i),
                    expected.x,expected.y,expected.z));
                expected.x+=.001;
                maxExpectedError=std::max(maxExpectedError,
                    std::max(std::abs(expected.x-p.x),
                    std::max(std::abs(expected.y-p.y),std::abs(expected.z-p.z))));
                assert(p.status&&p.x>fixture.centre.x&&p.y>=0&&p.y<=1&&p.z>=0&&p.z<=1);
            }
        }
        if(mode==0)assert(outside>1000&&deleted>1000);
        else assert(outside==0&&deleted==0&&maxExpectedError<1e-14);
        printf("PASS actual_face_walk mode=%s n=%d initial_outside=%d deleted_dt_0.001=%d position_max_error=%.3g\n",
            mode==0?"legacy_3D":mode==1?"projected_Sf_Cf":"face_centre",n,outside,deleted,maxExpectedError);
    }
}