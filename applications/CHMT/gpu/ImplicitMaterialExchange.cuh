namespace implicitWindowGpu {
using ugkp::diagnostics::DeviceContext;
using ugkp::diagnostics::EntityKind;

__device__ void fail(DeviceContext* diagnostics,unsigned site,EntityKind kind,int index,Real value=0){
    using namespace ugkp::diagnostics;
    recordFatal(diagnostics,makeRecord(Code::InvalidPhysicalState,site,4,0,kind,index,-1,value));
}

__global__ void accumulateExchange(ResidentMaterialDriveHistory history,const GasWindowProgress* progress,
    std::uint64_t acceptedFrame,const int* cells,int cellCount,const int* offsets,const int* faces,bool particles,
    MaterialWindowCell* exchange,ImplicitWindowRemovalSum* removed,unsigned* receivers,
    const ImplicitWindowReport* report,DeviceContext* diagnostics,WallHeatIntegral* heat){
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=cellCount||(progress&&progress->stop==GasWindowStop::Fatal)||ugkp::diagnostics::hasFatal(diagnostics))return;
    const auto end=progress?progress->microSteps:acceptedFrame;
    if(report->frames==end)return;
    const auto count=progress?progress->batchFrames:std::size_t(1);
    if(end!=report->frames+count||count>history.frames){
        fail(diagnostics,38520,EntityKind::None,-1,Real(count));return;
    }
    const int cell=cells[row];
    auto sum=exchange[cell];
    for(std::size_t frame=0;frame<count;++frame){
        for(int entry=offsets[row];entry<offsets[row+1];++entry){
            const int f=faces[entry];
            const auto& raw=history.surface[frame*history.faces+f];
            auto* mass=removed[f].mass;
            if(addMaterialWindowPacket(sum,mass,raw.primary,cell)
                ||(hasPacketTransfer(raw.pore)&&addMaterialWindowPacket(sum,mass,raw.pore,cell))
                ||(particles&&raw.particleWallEnergy!=0&&addMaterialWindowPacket(sum,mass,materialHistoryPacket(raw,2),cell))
                ||addMaterialWindowRadiation(sum,raw.radiationEnergy,raw.radiationReceiver)){
                fail(diagnostics,38520,EntityKind::Face,f);return;
            }
            receivers[f]=raw.radiationReceiver;
        }
    }
    exchange[cell]=sum;
    if(!heat)return;
    for(int entry=offsets[row];entry<offsets[row+1];++entry){
        const int face=faces[entry];
        WallHeatIntegral batch;
        for(std::size_t frame=0;frame<count;++frame){
            const auto& raw=history.surface[frame*history.faces+face];
            batch.convection-=raw.primary.conductive+raw.pore.conductive;
            batch.gasTransport-=(raw.primary.energy-raw.primary.conductive)+(raw.pore.energy-raw.pore.conductive);
            batch.particleContact+=raw.particleWallEnergy;
        }
        heat[face].convection+=batch.convection;
        heat[face].gasTransport+=batch.gasTransport;
        heat[face].particleContact+=batch.particleContact;
    }
}

__global__ void acceptExchangeBatch(const GasWindowProgress* progress,std::uint64_t acceptedFrame,
    ImplicitWindowReport* report,const DeviceContext* diagnostics){
    if(!blockIdx.x&&!threadIdx.x&&(!progress||progress->stop!=GasWindowStop::Fatal)&&!ugkp::diagnostics::hasFatal(diagnostics))
        report->frames=progress?progress->microSteps:acceptedFrame;
}

void enqueueExchange(MaterialBackend& b,const ResidentMaterialDriveHistory& history,const GasWindowProgress* progress,
    std::uint64_t acceptedFrame,bool particles,cudaStream_t stream,WallHeatIntegral* heat){
    auto& s=b.implicit;const int owners=s.surfaceCells.size();
    const int threads=ugkp::gpu::workPlan().materialBlockThreads;
    if(history.faces!=s.surfaceFaces.size()||(owners&&!history.surface))
        throw std::logic_error("implicit material exchange history layout mismatch");
    if(owners)accumulateExchange<<<(owners+threads-1)/threads,threads,0,stream>>>(history,progress,acceptedFrame,
        s.surfaceCells.data(),owners,s.surfaceOffsets.data(),s.surfaceFaces.data(),particles,
        s.exchange.data(),s.removed.data(),s.receivers.data(),s.report.data(),b.diagnostics.device(),heat);
    acceptExchangeBatch<<<1,1,0,stream>>>(progress,acceptedFrame,s.report.data(),b.diagnostics.device());
}

__global__ void addExternalRadiation(const EnergyReal* radiation,const int* cells,int cellCount,const int* offsets,
    const int* faces,const unsigned* receivers,MaterialWindowCell* exchange,const ImplicitWindowReport* report,DeviceContext* diagnostics){
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=cellCount||report->initialized||ugkp::diagnostics::hasFatal(diagnostics))return;
    const int cell=cells[row];auto sum=exchange[cell];
    for(int e=offsets[row];e<offsets[row+1];++e){
        const int f=faces[e];
        if(addMaterialWindowRadiation(sum,radiation[f],receivers[f])){
            fail(diagnostics,38520,EntityKind::Face,f,radiation[f]);return;
        }
    }
    exchange[cell]=sum;
}

__global__ void publishRemovals(const ImplicitWindowRemovalSum* input,ImplicitWindowRemoval* output,int n){
    const int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=n)return;
    for(int k=0;k<Nc;++k)output[f].mass[k]=input[f].mass[k].value();
}

inline int blocks(int n,int threads){return (n+threads-1)/threads;}
}

bool beginImplicitMaterialExchange(MaterialBackend& b,const HostState& base,const CouplingInterval& window,
    const CpuMaterialControls& controls,std::string& error){
    auto& s=b.implicit;
    const auto dt=window.end-window.begin;
    const PhysicalReal physicalDt=PhysicalReal(dt);
    if(!validCouplingInterval(window)||window.begin!=base.time||!finite(physicalDt)||physicalDt<=0
        ||controls.nonlinearMaxIterations<=0||controls.linearMaxIterations<=0
        ||!finite(controls.nonlinearRelativeTolerance)||controls.nonlinearRelativeTolerance<0
        ||!finite(controls.energyAbsoluteTolerance)||controls.energyAbsoluteTolerance<0
        ||!finite(controls.energyRelativeTolerance)||controls.energyRelativeTolerance<0
        ||!finite(controls.linearEnergyBudgetFraction)||controls.linearEnergyBudgetFraction<=0
        ||b.physics.nReactions<0||b.physics.nReactions>Nr||base.solid.size()!=b.current.size()){
        error="invalid resident implicit material window input";return false;
    }
    const int cells=b.cells,coupled=base.surface.solidCell.size();
    if(base.surface.solidFace.size()!=std::size_t(coupled)){
        error="invalid implicit material exchange face layout";return false;
    }
    if(!s.surfaceReady||s.surfaceOwners!=base.surface.solidCell||s.exchange.size()!=std::size_t(cells)){
        std::vector<std::vector<int>> rows(cells);
        for(int f=0;f<coupled;++f){
            const int c=base.surface.solidCell[f];
            if(c<0||c>=cells){error="implicit material exchange has no solid owner";return false;}
            rows[c].push_back(f);
        }
        std::vector<int> offsets{0},faces,owners;
        for(int c=0;c<cells;++c)if(!rows[c].empty()){
            owners.push_back(c);faces.insert(faces.end(),rows[c].begin(),rows[c].end());
            offsets.push_back(faces.size());
        }
        if(!b.upload(s.surfaceOffsets,offsets)||!b.upload(s.surfaceFaces,faces)||!b.upload(s.surfaceCells,owners))
            return b.failure(error);
        s.surfaceOwners=base.surface.solidCell;s.surfaceReady=true;++s.revision;
    }
    ImplicitWindowControls input;
    input.interval=Real(physicalDt);input.dt=physicalDt;input.iterations=controls.nonlinearMaxIterations;
    input.changeGoal=std::max(controls.nonlinearRelativeTolerance,PhysicalReal(4)*std::numeric_limits<PhysicalReal>::epsilon());
    input.absolute=controls.energyAbsoluteTolerance;input.relative=controls.energyRelativeTolerance;
    input.linearFraction=controls.linearEnergyBudgetFraction;input.linear.maximumIterations=controls.linearMaxIterations;
    if(!b.uploadRaw(s.controls,&input,1)||!b.allocate(s.report,1)||!s.report.zero(b.stream.get())
        ||!b.allocate(s.exchange,cells)||!s.exchange.zero(b.stream.get())
        ||!b.allocate(s.removed,coupled)||!s.removed.zero(b.stream.get())
        ||!b.allocate(s.receivers,coupled)||!s.receivers.zero(b.stream.get()))return b.failure(error);
    b.operation(window.end);
    error.clear();return true;
}

ugkp::gpu::ExecutionGraphNode appendImplicitMaterialExchange(MaterialBackend& b,const ResidentMaterialDriveHistory& history,
    const GasWindowProgress* progress,bool particles,ugkp::gpu::ExecutionGraph graph,ugkp::gpu::ExecutionGraphNode dependency,
    void* executionStream,WallHeatIntegral* heat){
    using ugkp::gpu::executionGraphDetail::appendRecorded;
    const auto stream=static_cast<cudaStream_t>(executionStream);
    return appendRecorded(graph,{dependency},stream,[&]{
        implicitWindowGpu::enqueueExchange(b,history,progress,0,particles,stream,heat);
    });
}

void enqueueImplicitMaterialFrame(MaterialBackend& b,const CoupledSurfaceDownload* surface,std::size_t faces,
    std::uint64_t acceptedFrame,bool particles,void* stream){
    implicitWindowGpu::enqueueExchange(b,{surface,nullptr,nullptr,1,faces,stream},nullptr,acceptedFrame,particles,
        static_cast<cudaStream_t>(stream),nullptr);
}

void orderImplicitWindow(MaterialBackend& b,void* destination){
    using ugkp::gpu::executionGraphDetail::checked;
    auto& s=b.implicit;
    if(b.stream.get()==destination)return;
    if(!s.ready)checked(cudaEventCreateWithFlags(&s.ready,cudaEventDisableTiming));
    checked(cudaEventRecord(s.ready,b.stream.get()));
    checked(cudaStreamWaitEvent(static_cast<cudaStream_t>(destination),s.ready,0));
}

void acceptImplicitWindowExecution(MaterialBackend& b,void* producer){
    using ugkp::gpu::executionGraphDetail::checked;
    auto& s=b.implicit;
    if(b.stream.get()==producer)return;
    if(!s.ready)checked(cudaEventCreateWithFlags(&s.ready,cudaEventDisableTiming));
    checked(cudaEventRecord(s.ready,static_cast<cudaStream_t>(producer)));
    checked(cudaStreamWaitEvent(b.stream.get(),s.ready,0));
}

unsigned long long implicitWindowRevision(const MaterialBackend& b){return b.implicit.revision;}
const ugkp::diagnostics::DeviceContext* implicitWindowDiagnostics(const MaterialBackend& b){return b.diagnostics.device();}

bool collectImplicitWindowRemoval(MaterialBackend& b,std::vector<ImplicitWindowRemoval>& removed,std::string& error){
    using namespace implicitWindowGpu;
    auto& s=b.implicit;
    const int n=s.removed.size(),threads=ugkp::gpu::workPlan().materialBlockThreads;
    if(!b.allocate(s.removals,n))return b.failure(error);
    if(n)publishRemovals<<<blocks(n,threads),threads,0,b.stream.get()>>>(s.removed.data(),s.removals.data(),n);
    return b.read(s.removals,removed)&&b.finish(error);
}
