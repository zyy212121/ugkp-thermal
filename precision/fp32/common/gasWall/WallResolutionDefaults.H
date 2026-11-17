#pragma once
#include <cfloat>
namespace ugkwp { namespace gaswall {

inline constexpr int defaultWallNodes=8;
inline constexpr double defaultWallStretch=3.0;
inline constexpr double defaultWallRelativeTolerance=5e-2;
inline constexpr double defaultWallAbsoluteTolerance=8.0*FLT_EPSILON;
}}
