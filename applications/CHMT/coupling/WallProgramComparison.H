#pragma once
#include "core/Types.H"
namespace chmt {
struct WallProgramComparison { Real thermalError=0,massPredictionError=0,geometryError=0;bool regimeMatches=true; };
}
