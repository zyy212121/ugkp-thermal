#ifndef GPU_RESEARCH_HIERARCHY_H
#define GPU_RESEARCH_HIERARCHY_H
#include <cstring>
namespace gpuResearch {
struct Flags { bool cell; bool warp; bool split; bool heavy; };
// Paper names are independent ablations, not production gpuCsrLevel names.
inline bool decode(const char* name, Flags& flags) {
    struct State { const char* name; Flags flags; };
    const State states[] = {
        {"L0", {false,false,false,false}},
        {"L1", {true,false,false,false}},
        {"T1", {true,true,false,false}}, {"E1", {true,true,false,false}},
        {"S1", {true,true,true,false}},
        {"T2", {true,true,false,true}}, {"E2", {true,true,false,true}},
        {"S2", {true,true,true,true}}
    };
    for (const auto& state : states) {
        if (std::strcmp(name, state.name) == 0) { flags=state.flags; return true; }
    }
    return false;
}
}
#endif
