#pragma once
#include "core/Types.H"

namespace chmt {
struct WallHeatIntegral {
    EnergyReal convection=0,gasTransport=0,particleContact=0;
};
}
