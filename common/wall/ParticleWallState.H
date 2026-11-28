#pragma once

namespace Foam { namespace gpuThermal {
constexpr int coldWallAxialNodeCount = 8;
constexpr int coldWallRadialRingCount = 8;

enum ParticleWallInteractionModel : unsigned char
{
    particleWallInteractionNone = 0,
    particleWallReboundContact = 1,
    particleWallSolidifyingDeposition = 3,
    particleWallColdWall2D = 4
};

enum ParticleWallState : unsigned char
{
    particleWallMobile = 0,
    particleWallDeposited = 1,
    particleWallTransientRebound = 2,
    particleWallTransientDeposit = 3
};
}}
