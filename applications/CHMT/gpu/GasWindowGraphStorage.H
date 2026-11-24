#pragma once
#include <cuda_runtime.h>
namespace chmt {
struct ResidentMaterialDriveInterval;
struct WindowGraph {
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    const ResidentMaterialDriveInterval* storage=nullptr;
    const void* radiationOwner=nullptr;
    const void* implicitOwner=nullptr;
    unsigned long long radiationRevision=0,implicitRevision=0;
    const void* wallHeat=nullptr;
    bool implicitSolve=false,retainsHistory=false;
    void clear(){if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);
        graph=nullptr;executable=nullptr;storage=nullptr;}
    ~WindowGraph(){clear();}
};
}
