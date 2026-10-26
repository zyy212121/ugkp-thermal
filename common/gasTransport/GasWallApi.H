#pragma once
#include <cstdint>
#include "../gasWall/WallResolutionDefaults.H"
namespace ugkwpGpu {
struct BoundaryLayerConfigV1
{
    std::uint32_t version=1,speciesCount=0,model=1,nodes=ugkwp::gaswall::defaultWallNodes,maxIterations=60;
    std::uint32_t wallCount=0,quadratureCount=0,matchingCount=0,workspaceSlots=0,reservedAudit=0;
    double relativeTolerance=1e-8,absoluteTolerance=1e-10,stretch=ugkwp::gaswall::defaultWallStretch;
    double turbulentPrandtl=.9,turbulentSchmidt=.7;
};


static constexpr std::uint32_t boundaryLayerDiagnosticScalars=25;
}
extern "C" int ugkwpGpuResidentStrictConfigureBoundaryLayerV1
(void* handle,const ugkwpGpu::BoundaryLayerConfigV1* config,const int* faces,
 const int* quadratureOffsets,const int* matchingOffsets,const int* matchingCells,
 const double* geometry,const double* quadratureDistance,const double* quadratureWeight,
 const double* matchingWeight,const double* speciesFlux);
extern "C" int ugkwpGpuResidentStrictDownloadBoundaryLayerV1
(void* handle,std::uint32_t wallCount,std::uint32_t speciesCount,double* diagnostics);

// Startup-only opt-in; downloaded rows are face-major, node-major y,U[3],T,k,omega,Y[Ns].
extern "C" int ugkwpGpuResidentStrictEnableBoundaryLayerProfilesV1(void* handle);
extern "C" int ugkwpGpuResidentStrictDownloadBoundaryLayerProfilesV1
(void* handle,std::uint32_t walls,std::uint32_t nodes,std::uint32_t species,double* profiles,double* stageTime);
