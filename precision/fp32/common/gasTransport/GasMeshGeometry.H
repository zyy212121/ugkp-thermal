#pragma once
#include "../GpuPrecisionTypes.H"
#include "../gpu/PolyhedralGeometryMath.H"
#ifdef __CUDACC__
#define UGKP_GAS_MESH_HD __host__ __device__
#else
#define UGKP_GAS_MESH_HD
#endif
namespace ugkwp {
struct GasFaceGeometry {
    ugkp::gpu::geometry::Point periodicShift{};
    GpuReal area=0,delta=0,weight=GpuReal(.5);
};
// Keep global coordinates and periodic maps in their geometry-owner precision.
// Publish one FP32 set of local metrics for ordinary transport.
UGKP_GAS_MESH_HD inline GasFaceGeometry gasFaceGeometry(
    ugkp::gpu::geometry::Point areaVector,ugkp::gpu::geometry::Point faceCentre,
    ugkp::gpu::geometry::Point ownerCentre,bool neighbour,
    ugkp::gpu::geometry::Point neighbourCentre,bool periodic,
    ugkp::gpu::geometry::Point pairedFaceCentre)
{
    using namespace ugkp::gpu::geometry;
    GasFaceGeometry result;
    const GpuReal sx=GpuReal(areaVector.x),sy=GpuReal(areaVector.y),sz=GpuReal(areaVector.z);
    result.area=::sqrtf(sx*sx+sy*sy+sz*sz);
    Point chord=faceCentre-ownerCentre;
    if(neighbour)chord=neighbourCentre-ownerCentre;
    if(periodic){result.periodicShift=faceCentre-pairedFaceCentre;
        chord=neighbourCentre+result.periodicShift-ownerCentre;}
    const GpuReal dx=GpuReal(chord.x),dy=GpuReal(chord.y),dz=GpuReal(chord.z);
    const GpuReal nx=sx/result.area,ny=sy/result.area,nz=sz/result.area;
    const GpuReal projection=dx*nx+dy*ny+dz*nz;
    const GpuReal axial=projection<0?-projection:projection;
    const GpuReal skew=GpuReal(.05)*::sqrtf(dx*dx+dy*dy+dz*dz);
    result.delta=GpuReal(1)/(axial>skew?axial:skew);
    if(neighbour||periodic){
        const Point ownerOffset=faceCentre-ownerCentre;
        const GpuReal a=GpuReal(ownerOffset.x)*nx+GpuReal(ownerOffset.y)*ny+GpuReal(ownerOffset.z)*nz;
        const GpuReal own=a<0?-a:a;
        const Point mappedNeighbour=ownerCentre+chord;
        const Point neighbourOffset=mappedNeighbour-faceCentre;
        const GpuReal b=GpuReal(neighbourOffset.x)*nx+GpuReal(neighbourOffset.y)*ny+GpuReal(neighbourOffset.z)*nz;
        const GpuReal other=b<0?-b:b;
        result.weight=(own+other)>0?other/(own+other):GpuReal(.5);
    }
    return result;
}
}
#undef UGKP_GAS_MESH_HD
