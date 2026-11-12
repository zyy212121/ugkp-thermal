#pragma once
#include "FoamSolverScheme.H"
#include "thermal/ParticleNumericalControls.H"
namespace ugkp { namespace solverControls {
// Host startup control only: preserve the existing device-create ABI.
inline int readParticleTrackingMaxFaceWalkHops(
    const Foam::Time& time,const Foam::word& region=Foam::word::null) {
    int hops=32;
    startupScheme(time).read("particle.tracking","maxFaceWalkHops",hops,"default",region);
    return hops;
}
inline thermal::ParticleNumericalControls readParticleNumericalControls(
    const Foam::Time& time,const Foam::word& region=Foam::word::null) {
    thermal::ParticleNumericalControls controls;
    const auto& scheme=startupScheme(time);
    scheme.read("particle.finiteContact","maximumIterations",controls.finiteContactMaximumIterations,"default",region);
    scheme.read("particle.packing","iterations",controls.packingIterations,"default",region);
    return controls;
}
}}
