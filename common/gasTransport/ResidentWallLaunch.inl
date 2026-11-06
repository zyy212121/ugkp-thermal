namespace
{
// Define the callback in the same internal namespace as its forward declaration.
template<bool Capture> int launchBoundaryLayer(DeviceState* s)
{
        const bool analytic=s->gasBoundaryLayerModel.config.model==ugkwp::gaswall::BoundaryLayerModel::ConstantTransport;
        const int workers=analytic?s->gasBoundaryLayer.count:s->gasBoundaryLayerModel.workspaceCount;
        if(workers<=0){setLastErrorText("boundaryLayer scratch pool is empty");return 1;}
        const auto error=launchGasBoundaryLayerPool<Capture>(s->deviceState,
            s->gasBoundaryLayerModel,s->gasBoundaryLayer.count,s->gasCaptureStream);
        if(error!=cudaSuccess){setLastError("prepare sparse boundaryLayer profiles",error);return 1;}
        if(s->gasBoundaryLayerModel.conservative.enabled){
#define RECONSTRUCT_WALL_CAPACITY(N) case N: reconstructConservativeWallPool<N><<<workers,32,0,s->gasCaptureStream>>>(s->deviceState);break
            switch(s->gasBoundaryLayerModel.workspaceCapacity){RECONSTRUCT_WALL_CAPACITY(24);RECONSTRUCT_WALL_CAPACITY(48);RECONSTRUCT_WALL_CAPACITY(128);}
#undef RECONSTRUCT_WALL_CAPACITY
            const auto reconstructionError=cudaGetLastError();if(reconstructionError!=cudaSuccess){setLastError("reconstruct conservative wall profiles",reconstructionError);return 1;}
        }
        return 0;
}
} // namespace

template<class Base> struct ResidentWallGasHostPolicy:Base
{
    template<class HostState> static int applyFaceSources(HostState*s,double,double){
        if(!s->gasBoundaryLayerModel.conservative.enabled)return 0;
        if(prepareConservativeProjection(s))return 1;
        int b=s->fixedFaceBlockThreads;assembleConservativeWallFlux<<<(s->nFaces+b-1)/b,b,0,s->gasCaptureStream>>>(s->deviceState);
        auto error=cudaGetLastError();if(error!=cudaSuccess){setLastError("assemble conservative wall flux",error);return 1;}return 0;
    }
    template<class HostState> static int applySstFaceSources(HostState*s,double dt){
        if(!s->gasBoundaryLayerModel.conservative.enabled)return 0;
        int b=s->fixedFaceBlockThreads;
        int cb=s->fixedCellBlockThreads;conservativeKFluxBudget<<<(s->nCells+cb-1)/cb,cb,0,s->gasCaptureStream>>>(s->deviceState,dt);
        applyConservativeKFluxBudget<<<(s->nFaces+b-1)/b,b,0,s->gasCaptureStream>>>(s->deviceState);
        auto error=cudaGetLastError();if(error!=cudaSuccess){setLastError("assemble conservative wall SST flux",error);return 1;}return 0;
    }

    template<class HostState> static int prepareBoundaryLayer(HostState* s,double,double stageTime)
    {
        if(!s->gasBoundaryLayer.enabled)return 0;
        s->gasBoundaryLayerStageTime=stageTime;
        if(s->gasBoundaryLayerModel.conservative.enabled){
            int block=s->fixedCellBlockThreads,grid=(s->nCells+block-1)/block;
            computeGasGradientLimiterKernel<<<grid,block,0,s->gasCaptureStream>>>(s->deviceState);
            computeSstGradientsKernel<<<grid,block,0,s->gasCaptureStream>>>(s->deviceState);
            prepareConservativeWallMatching<<<(s->gasBoundaryLayer.count+63)/64,64,0,s->gasCaptureStream>>>(s->deviceState);
        }
        return s->gasBoundaryLayerLaunch(s);
    }
};
