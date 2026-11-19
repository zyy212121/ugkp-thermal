#pragma once
#include <cuda_runtime.h>
namespace chmt {
struct ResidentMaterialDriveInterval;
struct WindowGraph {
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    const ResidentMaterialDriveInterval* storage=nullptr;
    unsigned long long radiationRevision=0;
    void clear(){if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);
        graph=nullptr;executable=nullptr;storage=nullptr;}
    ~WindowGraph(){clear();}
};
}
