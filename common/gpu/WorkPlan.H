#pragma once
// Host-only launch policy, published once before CUDA owners and graph capture.
// Scratch sizes and graph geometry depend on this immutable startup policy.
namespace ugkp { namespace gpu {
struct WorkPlan {
    int cellBlockThreads=128,faceBlockThreads=128;
    int particleBlockThreads=128,reductionBlockThreads=128;
    int sparseBlockThreads=128,radiationBlockThreads=128;
    int thermalBlockThreads=128,materialBlockThreads=128;
    int coldWallBlockThreads=128;
    bool operator==(const WorkPlan& b) const {
        return cellBlockThreads==b.cellBlockThreads && faceBlockThreads==b.faceBlockThreads
            && particleBlockThreads==b.particleBlockThreads && reductionBlockThreads==b.reductionBlockThreads
            && sparseBlockThreads==b.sparseBlockThreads && radiationBlockThreads==b.radiationBlockThreads
            && thermalBlockThreads==b.thermalBlockThreads && materialBlockThreads==b.materialBlockThreads
            && coldWallBlockThreads==b.coldWallBlockThreads;
    }
};
inline WorkPlan& workPlan() { static WorkPlan value; return value; }
}}
