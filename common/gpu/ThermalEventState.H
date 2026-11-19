#pragma once
#include <cstdint>
namespace ugkp { namespace gpu {


struct ThermalEventState {
    double time=0,heatInterval=0,radiationInterval=0;
    double completedTime=0,previousTime=0,completedRadiationTime=0,lastRadiationInterval=0,period=0;
    std::uint64_t sequence=0,completedStep=0,radiationFrequency=1;
    bool radiationEnabled=false;
    int stage=0;
    bool radiationDue=false,particlePublished=false,committed=false;
};
}}
