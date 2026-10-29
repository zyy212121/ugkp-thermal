#pragma once
#include "AttachedParticleGeometry.H"
#include "ParticleTrackingGeometry.H"
#include <cuda_runtime_api.h>
namespace ugkp { namespace thermal {
// Boundary topology and coefficients are selected by the application once.
// These face-indexed arrays are borrowed, not copied or owned by this API.
struct ParticleTrackingBoundaryView {
    int nFaces=0;
    const int* kind=nullptr;
    const double* restitution=nullptr;
    const double* tangential=nullptr;
};
// Rebuild tracking planes/neighbours from the owner's resident gas mesh after
// geometry publication; no per-step upload of cell-face plane arrays.
int refreshParticleTrackingGeometry(void* owner,const ParticleTrackingBoundaryView& boundary,cudaStream_t stream);
// Validated topology and both point arrays must live until stream completes.
// No field readback or per-particle host work. Only attached active particles
// are mapped; lab-frame velocities and all contact/thermal histories remain
// resident and unchanged. Wall motion forces belong to the contact model.
int mapAttachedParticles(void* owner,const ParticleSurfaceGeometry& previous,
    const ParticleSurfaceGeometry& endpoint,const int* oldToNewFace,cudaStream_t stream);
}}
