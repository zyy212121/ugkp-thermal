// Scheduled 2A geometry test. Calls the real production intersection helpers.
#include "GpuPrecisionTypes.H"
#include "GpuOperatorContract.cuh"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>
#define __device__
#define __forceinline__ inline
#define GPU_OPERATOR_REAL GpuReal
#define GPU_OPERATOR_R GPU_R
#define GPU_OPERATOR_TINY(x) GPU_TINY(x)
struct DeviceState {
    int* cellPlaneStart=nullptr;
    int* cellPlaneCount=nullptr;
    GpuGeometry *cellLength=nullptr,*planeNx=nullptr,*planeNy=nullptr,*planeNz=nullptr,*planeD=nullptr;
};
#include "operators/clampMin.cuh"
#include "operators/clampRange.cuh"
#include "operators/pointInsideCell.cuh"
int main() {
    static_assert(std::is_same<GpuReal,float>::value,"exercise float bulk");
    static_assert(std::is_same<GpuGeometry,double>::value,"geometry stays double");
    static_assert(std::is_same<decltype(segmentPlaneIntersectionFraction(0.,1.,0.)),double>::value,"fraction stays double");
    // The original strict binary64 threshold: below and exactly at it retain
    // the endpoint fallback. The neighboring larger double uses interpolation.
    const double threshold=1e-300;
    assert(segmentPlaneIntersectionFraction(0.,threshold,.25*threshold)==1.);
    assert(segmentPlaneIntersectionFraction(-1e-310,1e-310,0.)==1.);
    const double above=std::nextafter(threshold,std::numeric_limits<double>::infinity());
    const double nearThreshold=segmentPlaneIntersectionFraction(0.,above,.25*threshold);
    assert(std::abs(nearThreshold-(.25*threshold)/above)<1e-15);
    assert(segmentPlaneIntersectionFraction(-1e-290,1e-290,0.)==.5);
    assert(segmentPlaneIntersectionFraction(1.,0.,.5)==1.);
    assert(segmentPlaneIntersectionFraction(1.,2.,0.)==0.);
    assert(segmentPlaneIntersectionFraction(-2.,-1.,0.)==1.);
    int start=0,count=1;double length=1e-12,nx=1,ny=0,nz=0,offset=0;
    DeviceState s{&start,&count,&length,&nx,&ny,&nz,&offset};
    const double tol=cellClassificationTolerance(s,0);
    const double x0=tol-4e-36,x1=tol+4e-36;
    double hit=-1;
    const int plane=firstSegmentIntersection(s,0,x0,0.,0.,x1,0.,0.,hit);
    assert(plane==0&&std::isfinite(hit)&&hit>0&&hit<1);
    assert(std::abs(hit-(tol-x0)/(x1-x0))<1e-15);
    assert(pointInsideCell(s,0,x0,0.,0.)&&!pointInsideCell(s,0,x1,0.,0.));
    std::cout<<"PASS double geometry threshold="<<threshold<<" tiny segment hit="<<hit<<'\n';
}
