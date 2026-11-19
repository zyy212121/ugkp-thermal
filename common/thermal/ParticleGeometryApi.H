#pragma once
#include "AttachedParticleGeometry.H"
#include "ParticleTrackingGeometry.H"
#include <cuda_runtime_api.h>
#include "../gpu/ExecutionGraph.H"
namespace ugkp { namespace thermal {


struct ParticleTrackingBoundaryView {
    int nFaces=0;
    const int* kind=nullptr;
    const double* restitution=nullptr;
    const double* tangential=nullptr;
};


int refreshParticleTrackingGeometry(void* owner,const ParticleTrackingBoundaryView& boundary,cudaStream_t stream);


int relocateMobileParticles(void* owner,cudaStream_t stream);

int refreshAcceptedParticleGeometryMoments(void* owner,cudaStream_t gasStream);



gpu::ExecutionGraphNode appendParticleGeometryMoments(void* owner,gpu::ExecutionGraph graph,
    gpu::ExecutionGraphNode dependency,cudaStream_t stream);




int mapAttachedParticles(void* owner,const ParticleSurfaceGeometry& previous,
    const ParticleSurfaceGeometry& endpoint,const int* oldToNewFace,cudaStream_t stream);
}}
