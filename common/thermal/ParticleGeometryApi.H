#pragma once
#include "AttachedParticleGeometry.H"
#include <cuda_runtime_api.h>
namespace ugkp { namespace thermal {
// Validated topology and both point arrays must live until stream completes.
// No field readback or per-particle host work. Only attached active particles
// are mapped; lab-frame velocities and all contact/thermal histories remain
// resident and unchanged. Wall motion forces belong to the contact model.
int mapAttachedParticles(void* owner,const ParticleSurfaceGeometry& previous,
    const ParticleSurfaceGeometry& endpoint,const int* oldToNewFace,cudaStream_t stream);
}}
