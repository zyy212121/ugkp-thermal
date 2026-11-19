#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <new>
#include <stdexcept>
#include "gpu/SharedGasDeviceStorage.cuh"
#include "thermal/ParticleSurfaceState.H"
#include "gpu/CoupledSurfaceTransfer.cuh"
#include "gpu/CoupledSurfacePipeline.cuh"

using namespace chmt;
static void checked(cudaError_t code){
    if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));
}
template<class T> static T* managed(std::size_t n){
    T* result=nullptr;
    checked(cudaMallocManaged(&result,n*sizeof(T)));
    for(std::size_t i=0;i<n;++i)new(result+i) T{};
    return result;
}
int main(){
    constexpr int largest=257;
    const auto records=managed<CoupledSurfaceRecord>(largest);
    const auto faces=managed<int>(largest);
    const auto surface=managed<CoupledSurfaceDownload>(2*largest);
    const auto walls=managed<CoupledWallDownload>(2*largest);
    const auto failure=managed<CoupledRecordFailure>(1);
    const auto diagnostic=managed<ugkp::diagnostics::DeviceContext>(1);
    const auto frame=managed<std::size_t>(1);
    *frame=1;
    for(int i=0;i<largest;++i){
        faces[i]=largest-i;
        records[i].primary.kind=ExchangeKind::GasSolid;
        records[i].primary.solidCell=0;
        records[i].radiationEnergy=i+1;
    }
    for(const int count:{0,4,128,129,257}){
        const auto run=[&](bool withWall){
            *diagnostic={};
            enqueueCoupledSurfaceArchive(records,count,faces,{},{surface,withWall?walls:nullptr,frame},
                failure,1,1,diagnostic,128,nullptr);
            checked(cudaDeviceSynchronize());
        };
        surface[0].radiationEnergy=-11;
        run(false);
        assert(!ugkp::diagnostics::hasFatal(diagnostic) && surface[0].radiationEnergy==-11);
        for(int i=0;i<count;++i)assert(surface[count+i].radiationEnergy==i+1);
        if(!count)continue;
        records[0].primary.solidCell=-1;
        records[count-1].failed=1;
        run(true);
        assert(diagnostic->firstFatal.sourceSiteId==2631 && diagnostic->firstFatal.entityIndex==faces[count-1]);
        records[count-1].failed=0;
        run(false);
        assert(diagnostic->firstFatal.sourceSiteId==2633 && diagnostic->firstFatal.entityIndex==faces[0]);
        records[0].primary.solidCell=0;
    }
    checked(cudaFree(frame));checked(cudaFree(diagnostic));checked(cudaFree(failure));checked(cudaFree(walls));
    checked(cudaFree(surface));checked(cudaFree(faces));checked(cudaFree(records));
    std::puts("Five archive sizes passed, including both CUDA paths, frame offset and error priority");
}
