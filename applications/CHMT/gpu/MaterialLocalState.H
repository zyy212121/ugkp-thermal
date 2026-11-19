#pragma once
#include <cstdint>
namespace chmt {
struct MaterialWindowCommit {
    double time=0;
    std::uint64_t sequence=0,acceptedStep=0;
    bool committed=false;
};
struct MaterialLocalProgress {
    std::uint64_t acceptedSteps=0;
    int failurePhase=0,failureCode=0,entity=-1,accepted=0;
    bool complete=false;
};
}
