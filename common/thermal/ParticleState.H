#pragma once
#include <cstdint>
namespace ugkp { namespace thermal {
// Complete physical thermal/contact history, independent of the application
// mesh vector type. Per-step reduction/candidate scratch is deliberately absent.
struct ParticleThermalHistory {
    bool initialized=false;
    int status=1;
    double temperature=0,theta=0,specificEnthalpyResidual=0;
    double savedVelocity[3]{}; // Exact lab-frame pre-step/deferred rebound velocity.
    double contactMaximumArea=0,contactPeakFraction=0;
    float coldNodeSpecificEnthalpy[8]{};
    double coldRingSolidMass[8]{},coldFrozenArea=0,coldContactAge=0;
    float cold2DNodeSpecificEnthalpy[64]{};
    double cold2DRingContactAge[8]{},cold2DFrozenArea=0;
};
template<class Archive,class History>
void visitParticleThermalHistory(Archive& a,History& h) {
    a(h.initialized);a(h.status);a(h.temperature);a(h.theta);a(h.specificEnthalpyResidual);
    a(h.savedVelocity);a(h.contactMaximumArea);a(h.contactPeakFraction);
    a(h.coldNodeSpecificEnthalpy);a(h.coldRingSolidMass);a(h.coldFrozenArea);a(h.coldContactAge);
    a(h.cold2DNodeSpecificEnthalpy);a(h.cold2DRingContactAge);a(h.cold2DFrozenArea);
}
struct ParticleVector {double x=0,y=0,z=0;};
template<class Vector=ParticleVector>
struct ParticleState {
    Vector contactRelativeVelocity{};
    double contactMechanicalEnergy=0;
    std::uint64_t id=0;int cell=-1,face=-1,state=0;Vector position{},velocity{};
    double mass=0,energy=0,diameter=0,contactAge=0,contactDuration=0,contactArea=0;
    std::uint64_t rng=0;
    ParticleThermalHistory thermal{};
    // energy is the represented thermal inventory [J]. Once initialized,
    // temperature/profile plus specificEnthalpyResidual is the restart state;
    // energy is a derived output, never a second independent thermal update.
};
template<class ToVector,class FromVector>
ParticleState<ToVector> adaptParticleState(const ParticleState<FromVector>& from) {
    ParticleState<ToVector> to;
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
