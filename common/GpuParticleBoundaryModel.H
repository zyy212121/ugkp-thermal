#ifndef UGKP_GPU_PARTICLE_BOUNDARY_MODEL_H
#define UGKP_GPU_PARTICLE_BOUNDARY_MODEL_H
namespace Foam { namespace gpuParticleBoundary {
// Geometric boundaries reflect resolved particle velocity without dissipation.
// Physical walls retain their configured restitution/contact law.
constexpr double geometricReflectionRestitution = 1.0;
} }
#endif
