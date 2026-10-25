#pragma once
namespace ugkwp { namespace gaswall {
// Internal steady-wall resolution; overridable in constant/fluidProperties:
// turbulence/RAS/kOmegaSSTCoeffs/boundaryLayer { nodes 8; stretch 3; }
inline constexpr int defaultWallNodes=8;
inline constexpr double defaultWallStretch=3.0;
}}
