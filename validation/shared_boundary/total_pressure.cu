#include "gasBoundary/Resident.cuh"
#include "gasBoundary/Evaluate.H"
#include <cassert>
#include <cstdio>
using namespace ugkp::boundary;
__global__ void sample(View v, Primitive* q) {
    const int i=threadIdx.x;
    if(i>=v.count)return;
    const auto& f=v.faces[i];double value[Components];
    for(int c=0;c<Components;++c)value[c]=evaluate(v,f.function[c],.5,f.value[c]);
    assert(totalState(f,value,{1,-60,-80,0,100000,300},287.05,.6,.8,0,0,q[i]));
}
int main() {
    Definition d;d.faces.resize(12);
    d.functions={{FunctionKind::Table,Bounds::Clamp,0,2,0},{FunctionKind::Table,Bounds::Clamp,2,2,0}};
    d.terms={{0,-40},{1,-120},{0,30},{1,90}};
    for(int i=0;i<12;++i) {
        auto& f=d.faces[i];const int mode=i/4,variant=i%4;
        f.uMode=4;f.pMode=f.tMode=2;f.temperatureGamma=1.4;
        f.compressiblePressure=mode!=2;f.pressureGamma=mode==1?1:1.4;
        f.value[Pressure]=100000;f.value[Temperature]=300;
        f.value[Ux]=variant==0?0:variant==2?-68:-80;
        f.value[Uy]=variant==0?0:variant==2?76:60;
        if(variant==3){f.function[Ux]=0;f.function[Uy]=1;}
    }
    Resident r;std::string error;assert(r.configure(d,nullptr,error));
    Primitive* device;assert(cudaMalloc(&device,12*sizeof(Primitive))==cudaSuccess);
    sample<<<1,32>>>(r.view(),device);
    Primitive h[12];assert(cudaMemcpy(h,device,sizeof(h),cudaMemcpyDeviceToHost)==cudaSuccess);
    for(int i=0;i<12;++i) {
        const auto& q=h[i];
        // Independent native OpenFOAM comparison is performed by the Python driver.
        printf("DEVICE %d %.17g %.17g %.17g %.17g %.17g\n",i,q.ux,q.uy,q.uz,q.p,q.T);
    }
    assert(std::abs(h[1].p-94219.31412188)<1e-7);
    assert(std::abs(h[1].T-290.04653246)<1e-8);
    cudaFree(device);r.clear();
}
