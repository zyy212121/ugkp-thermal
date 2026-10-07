#ifndef CHMT_PARTICLES_PARTICLEKERNELS_H
#define CHMT_PARTICLES_PARTICLEKERNELS_H
#include "gpu/BackendResources.H"
namespace chmt {
int launchParticleExchange(const StateBuffers&,const StateBuffers&,StateBuffers&,GeometryBuffers&,SurfaceBuffers&,
    const PhysicsConfig&,Real,std::uint64_t,int,int,DeviceStatus*,void*);
}
#endif
