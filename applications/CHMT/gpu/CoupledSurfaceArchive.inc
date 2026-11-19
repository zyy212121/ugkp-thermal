struct CoupledSurfaceRecord {
    CoupledSurfaceSample sample{};
    GasWallResult rates{};
    WallFaceSample acceptedWall{};
    GasQ raw{};
    ExchangePacket primary{},pore{};
    GasWallMatchingSample matching{};
    EnergyReal radiationEnergy=0,particleHeatAtBegin=0;
    unsigned radiationReceiver=0;
    int failed=0,wallTransportCode=0;
    ugkwp::gaswall::WallStatus wallStatus{};
};
__device__ inline void packCoupledSurfaceRecord(int slot,const CoupledSurfaceRecord* records,int count,
    const int* faces,ugkp::thermal::ParticleWallExchangeView particle,
    CoupledSurfaceDownload* output,CoupledWallDownload* wallOutput,
    CoupledRecordFailure* failure,int solids,int films){
    if(slot>=count)return;
    const auto& in=records[slot];auto& out=output[slot];
    out.bulk=in.sample.primitive;out.gradient=in.sample.gradient;out.traction=in.rates.traction;
    out.primary=in.primary;out.pore=in.pore;out.acceptedWall=in.acceptedWall;
    out.radiationEnergy=in.radiationEnergy;out.radiationReceiver=in.radiationReceiver;
    out.failed=in.failed;
    out.particleWallEnergy=particle.depositedEnergy
        ?particle.depositedEnergy[faces[slot]]+particle.reflectedEnergy[faces[slot]]-in.particleHeatAtBegin:0;
    if(wallOutput){auto& wall=wallOutput[slot];
        wall.matching=in.matching;
        wall.transportCode=in.wallTransportCode;wall.status=in.wallStatus;
    }
    if(in.failed)atomicMin(&failure->wall,slot);
    if(coupledRecordOwnerFailure(out,solids,films))atomicMin(&failure->owner,slot);
}
struct CoupledSurfaceArchiveView {
    CoupledSurfaceDownload* surface;
    CoupledWallDownload* wall;
    const std::size_t* frame;
    __device__ CoupledSurfaceArchiveView current(int count) const {
        if(!frame)return *this;
        const auto offset=*frame*std::size_t(count);
        return {surface+offset,wall?wall+offset:nullptr,nullptr};
    }
};

template<bool oneBlock>
__global__ void packCoupledSurfaceKernel(const CoupledSurfaceRecord* records,int count,
    const int* faces,ugkp::thermal::ParticleWallExchangeView particle,CoupledSurfaceArchiveView storage,
    CoupledRecordFailure* failure,int solids,int films,ugkp::diagnostics::DeviceContext* diagnostics)
{
    __shared__ CoupledRecordFailure local;
    if constexpr(oneBlock){
        if(!threadIdx.x)local={count,count};
        __syncthreads();
        failure=&local;
    }
    const auto frame=storage.current(count);
    const int slot=blockIdx.x*blockDim.x+threadIdx.x;
    packCoupledSurfaceRecord(slot,records,count,faces,particle,frame.surface,frame.wall,failure,solids,films);
    if constexpr(oneBlock){
        __syncthreads();
        if(!threadIdx.x)reportCoupledRecordFailure(local,count,frame.surface,frame.wall,faces,solids,films,diagnostics);
    }
}
__global__ void validateCoupledSurfaceKernel(const CoupledRecordFailure* failure,int count,
    CoupledSurfaceArchiveView storage,const int* faces,int solids,int films,ugkp::diagnostics::DeviceContext* diagnostics)
{
    if(!blockIdx.x&&!threadIdx.x){
        const auto frame=storage.current(count);
        reportCoupledRecordFailure(*failure,count,frame.surface,frame.wall,faces,solids,films,diagnostics);
    }
}
inline void enqueueCoupledSurfaceArchive(const CoupledSurfaceRecord* records,int count,const int* faces,
    ugkp::thermal::ParticleWallExchangeView particle,CoupledSurfaceArchiveView storage,
    CoupledRecordFailure* failure,int solids,int films,ugkp::diagnostics::DeviceContext* diagnostics,
    int threads,cudaStream_t stream)
{
    if(!count)return;
    if(count<=threads){
        packCoupledSurfaceKernel<true><<<1,threads,0,stream>>>(records,count,faces,particle,storage,
            failure,solids,films,diagnostics);
    }else{
        resetCoupledRecordFailure<<<1,1,0,stream>>>(failure,count);
        packCoupledSurfaceKernel<false><<<(count+threads-1)/threads,threads,0,stream>>>(records,count,faces,
            particle,storage,failure,solids,films,diagnostics);
        validateCoupledSurfaceKernel<<<1,1,0,stream>>>(failure,count,storage,faces,solids,films,diagnostics);
    }
}
