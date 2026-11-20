#pragma once
#include "core/Types.H"

namespace chmt {
struct MaterialFaceTransport {
    PhysicalReal sweptPore[Ns]{},darcyPore[Ns]{},diffusivePore[Ns]{};
    EnergyReal sweptTemperature=0,darcyTemperature=0;
    bool darcyFromLeft=true;
};
}
