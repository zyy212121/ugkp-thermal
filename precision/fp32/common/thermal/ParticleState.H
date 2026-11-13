#pragma once
#include <cstdint>
#include "../GpuPrecisionTypes.H"
namespace ugkp { namespace thermal {
// Complete physical thermal/contact history, independent of the application
// mesh vector type. Per-step reduction/candidate scratch is deliberately absent.
struct ParticleThermalHistory {
    bool initialized=false;
    int status=1;
    float temperature=0,theta=0;
    float savedVelocity[3]{}; // Exact lab-frame pre-step/deferred rebound velocity.
    float contactMaximumArea=0,contactPeakFraction=0;
    float coldNodeSpecificEnthalpy[8]{};
    float coldRingSolidMass[8]{},coldFrozenArea=0;
    double coldContactAge=0;
    float cold2DNodeSpecificEnthalpy[64]{};
    double cold2DRingContactAge[8]{};
    float cold2DFrozenArea=0;
};
template<class Archive,class History>
void visitParticleThermalHistory(Archive& a,History& h) {
    a(h.initialized);a(h.status);a(h.temperature);a(h.theta);
    a(h.savedVelocity);a(h.contactMaximumArea);a(h.contactPeakFraction);
    a(h.coldNodeSpecificEnthalpy);a(h.coldRingSolidMass);a(h.coldFrozenArea);a(h.coldContactAge);
    a(h.cold2DNodeSpecificEnthalpy);a(h.cold2DRingContactAge);a(h.cold2DFrozenArea);
}
// Global particle positions retain the tracking geometry scalar.
struct ParticleVector {GpuGeometry x=0,y=0,z=0;};
// Velocities are physical state, independent of the geometry vector type.
struct ParticlePhysicalVector {GpuReal x=0,y=0,z=0;};
template<class Vector=ParticleVector,class PhysicalVector=ParticlePhysicalVector>
struct ParticleState {
    PhysicalVector contactRelativeVelocity{};
    GpuReal contactMechanicalEnergy=0;
    std::uint64_t id=0;int cell=-1,face=-1,state=0;
    Vector position{};PhysicalVector velocity{};
    GpuReal mass=0,energy=0,diameter=0,contactArea=0;
    GpuTime contactAge=0,contactDuration=0;
    std::uint64_t rng=0;
    ParticleThermalHistory thermal{};
    // energy is the represented thermal inventory [J]. Once initialized,
    // temperature/profile is the restart state;
    // energy is a derived output, never a second independent thermal update.
};
// Contact mechanical energy is a diagnostic of the deferred lab-frame
// rebound velocity, not an independently configurable inventory. Use the same
// rule before initial output and after a resident snapshot download.
template<class Vector,class PhysicalVector>
#if defined(__CUDACC__)
__host__ __device__
#endif
inline void normalizeParticleContactMechanicalEnergy(ParticleState<Vector,PhysicalVector>& p) {
    p.contactMechanicalEnergy=0;
    if(p.state==2 || p.state==3){
        const auto& u=p.thermal.savedVelocity;
        p.contactMechanicalEnergy=GpuReal(.5f)*p.mass*(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]);
    }
}
template<class ToVector,class ToPhysicalVector=ParticlePhysicalVector,class FromVector,class FromPhysicalVector>
ParticleState<ToVector,ToPhysicalVector> adaptParticleState
(const ParticleState<FromVector,FromPhysicalVector>& from) {
    ParticleState<ToVector,ToPhysicalVector> to;
    to.contactRelativeVelocity={from.contactRelativeVelocity.x,from.contactRelativeVelocity.y,from.contactRelativeVelocity.z};
    to.position={from.position.x,from.position.y,from.position.z};
    to.velocity={from.velocity.x,from.velocity.y,from.velocity.z};
    to.contactMechanicalEnergy=from.contactMechanicalEnergy;
    to.id=from.id;to.cell=from.cell;to.face=from.face;to.state=from.state;
    to.mass=from.mass;to.energy=from.energy;to.diameter=from.diameter;
    to.contactAge=from.contactAge;to.contactDuration=from.contactDuration;to.contactArea=from.contactArea;
    to.rng=from.rng;to.thermal=from.thermal;return to;
}
}}
