#include "BoundaryInjectionSampling.H"
#include <cassert>
#include <cstdio>
#include <algorithm>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
using ugkp::gpu::geometry::Point;
using ugkwp::BoundaryInjectionTriangle;
struct Test {Point p[8];int ids[8],scratch[8],n,count;BoundaryInjectionTriangle t[8];};
#ifdef __CUDACC__
__global__ void build(Test* cases,int n){int i=threadIdx.x;if(i<n){auto& c=cases[i];c.count=ugkwp::buildBoundaryInjectionTriangles(c.p,c.ids,c.n,c.scratch,c.t,{0,0,.125});}}
#endif
int main(){
    Test cases[6]{};
    const Point concave[6]={{0,0,0},{3,0,0},{3,1,0},{1,1,0},{1,3,0},{0,3,0}};
    for(int j=0;j<5;++j){auto& c=cases[j];c.n=6;for(int i=0;i<6;++i){c.p[i]=concave[i];c.ids[i]=j==1?5-i:i;}}
    // A changed concavity requires rebuilding, rather than fixed fan diagonals.
    cases[2].p[3]={.3,.4,0};
    // Actual 3D wedge-like radial plane: anisotropic embedding, not XY area.
    for(int i=0;i<6;++i){auto& p=cases[3].p[i];p.z=2*p.y;p.x*=.1;}
    cases[4].n=7;cases[4].p[6]={0,1.5,0};cases[4].ids[6]=6;
    cases[5].n=4;
    cases[5].p[0]={1,-.1,0};cases[5].p[1]={3,-.3,0};
    cases[5].p[2]={3,.3,0};cases[5].p[3]={1,.1,0};
    for(int i=0;i<4;++i)cases[5].ids[i]=i;
    for(auto& c:cases){c.count=ugkwp::buildBoundaryInjectionTriangles(c.p,c.ids,c.n,c.scratch,c.t,{0,0,.125});assert(c.count>0);}
    assert(std::abs(cases[0].t[cases[0].count-1].cumulativeArea-5)<1e-12);
    assert(std::abs(cases[1].t[cases[1].count-1].cumulativeArea-5)<1e-12);
    assert(std::abs(cases[4].t[cases[4].count-1].cumulativeArea-5)<1e-12);
    assert(std::abs(cases[2].t[cases[2].count-1].cumulativeArea-3.7)<1e-12);
    assert(std::abs(cases[3].t[cases[3].count-1].cumulativeArea-.5*std::sqrt(5.))<1e-12);
    for(int j: {0,1,3,4}){
        auto& c=cases[j];double meanX=0,meanY=0;int vertical=0;
        for(int i=0;i<200000;++i){double x=0,y=0,z=0;
            assert(ugkwp::sampleBoundaryInjection(c.t,0,c.count,ugkwp::injectionPositionSeed(123,17,.125,i),x,y,z));
            if(j==3){x*=10;assert(std::abs(z-.125-2*y)<1e-12);}else assert(z==.125);
            assert(x>=0&&y>=0&&x<=3&&y<=3&&(x<=1||y<=1));
            vertical+=y>1;meanX+=x;meanY+=y;
        }
        assert(std::abs(meanX/200000-1.1)<.01&&std::abs(meanY/200000-1.1)<.01);
        assert(std::abs(vertical/200000.-.4)<.005);
    }
    // A true wedge inlet trapezoid has width proportional to radius.
    // Area-uniform sampling must not become uniform in radial coordinate.
    const auto& wedge=cases[5];
    assert(std::abs(wedge.t[wedge.count-1].cumulativeArea-.8)<1e-12);
    double radialMean=0;int outer=0;
    for(int i=0;i<200000;++i){double r=0,y=0,z=0;
        assert(ugkwp::sampleBoundaryInjection(wedge.t,0,wedge.count,
            ugkwp::injectionPositionSeed(123,19,.125,i),r,y,z));
        assert(r>=1&&r<=3&&std::abs(y)<=.1*r&&z==.125);
        radialMean+=r;outer+=r>2;
    }
    radialMean/=200000;
    assert(std::abs(radialMean-13./6)<.01);
    assert(std::abs(double(outer)/200000-5./8)<.005);
    assert(std::abs(radialMean-2)>.1);
    printf("PASS wedge area=.8 radial_mean=%.9f outer_fraction=%.9f\n",radialMean,double(outer)/200000);
    Point line[3]={{0,0,0},{1,0,0},{2,0,0}};int ids[3]={0,1,2},scratch[3];BoundaryInjectionTriangle t[3];
    assert(!ugkwp::buildBoundaryInjectionTriangles(line,ids,3,scratch,t,{}));
#ifdef __CUDACC__
    Test* d;assert(cudaMalloc(&d,sizeof(cases))==cudaSuccess);assert(cudaMemcpy(d,cases,sizeof(cases),cudaMemcpyHostToDevice)==cudaSuccess);
    build<<<1,32>>>(d,6);assert(cudaDeviceSynchronize()==cudaSuccess);Test actual[6];assert(cudaMemcpy(actual,d,sizeof(actual),cudaMemcpyDeviceToHost)==cudaSuccess);
    for(int j=0;j<6;++j){assert(actual[j].count==cases[j].count);for(int i=0;i<actual[j].count;++i){
        assert(std::abs(actual[j].t[i].cumulativeArea-cases[j].t[i].cumulativeArea)<1e-12);
        for(int k=0;k<3;++k){assert(actual[j].t[i].a[k]==cases[j].t[i].a[k]);assert(actual[j].t[i].ab[k]==cases[j].t[i].ab[k]);assert(actual[j].t[i].ac[k]==cases[j].t[i].ac[k]);}
    }}
    cudaFree(d);puts("PASS CUDA builder agrees with CPU");
#endif
    puts("PASS concave/moving/reversed/collinear/degenerate/3D area and 800000 uniform samples");
}
