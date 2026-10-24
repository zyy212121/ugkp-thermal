#pragma once
#include "GasBoundaryLayerHost.H"

namespace ugkwp
{
// Use the production wall preparation for a current-state timestep estimate.
// This is a steady spatial solve; it neither advances inventories nor substeps
// physical time. The caller applies its representative owner omega afterwards.
template<class Policy,class Host,class Time,class Stream>
int prepareGasCourantBoundaryLayer(Host* host,Time dt,Time time,
    int cellGrid,int cellBlock,Stream stream)
{
    if constexpr(GasBoundaryLayerCapability<Host>::value)
    {
        if(!gasBoundaryLayerEnabled(*host))return 0;
        computeGasPrimitiveGradientsKernel<<<cellGrid,cellBlock,0,stream>>>
            (host->deviceState,true);
        const auto error=cudaGetLastError();
        if(error!=cudaSuccess)
        {
            setLastError("wall primitive gradients for gas Courant",error);
            return -1;
        }
        // Application launch callbacks use the host stream. Keep those launches
        // ordered with the preceding recovery/gradients and the following bound.
        const auto previousStream=host->gasCaptureStream;
        host->gasCaptureStream=stream;
        const int status=prepareGasBoundaryLayerStage<Policy>(host,dt,time,true);
        host->gasCaptureStream=previousStream;
        if(status!=0)
        {
            invalidateGasBoundaryLayerPreparation(*host);
            setLastErrorText("boundaryLayer preparation for gas Courant failed");
            return -1;
        }
        return 1;
    }
    return 0;
}
}
