#pragma once
#include <cstdio>
// Thin transaction/storage policy consumed by common GpuGasAdvance.cuh.
// All reconstruction, flux, RK, thermo, chemistry and CFL kernels remain common.
namespace sharedGasTrialDetail
{
struct Field {double* pointer;std::size_t count;};
template<class State> void appendSstAuditFields(State* s,std::vector<Field>& fields)
{
    if constexpr(ugkwp::GasSstAuditCapability<State>::value)
        if(s->gasSstAudit.enabled)
        {
            auto&a=s->gasSstAudit;const std::size_t n=s->nCells;
            fields.insert(fields.end(),{{a.transportK,n},{a.transportOmega,n},{a.sourceK,n},{a.sourceOmega,n},
                {a.constraintK,n},{a.constraintOmega,n},{a.initialTransportK,n},{a.initialTransportOmega,n},
                {a.initialSourceK,n},{a.initialSourceOmega,n},{a.initialConstraintK,n},{a.initialConstraintOmega,n},{a.volume,n}});
        }
}
inline std::vector<Field> cellFields(DeviceState* s)
{
    const std::size_t n=s->nCells;
    std::vector<Field> fields={{s->rho,n},{s->rhoUx,n},{s->rhoUy,n},{s->rhoUz,n},{s->rhoE,n},
        {s->Ux,n},{s->Uy,n},{s->Uz,n},{s->p,n},{s->Tgas,n},
        {s->gasSpecies.rho,n*ugkwp::compiledGasSpecies},
        {s->gasSpecies.soundSpeed,n},{s->gasSpecies.heatCapacity,n},{s->gasSpecies.gasConstant,n},
        {s->gasFluxPositivityScale,n},{s->gasDiffusionNumber,n},{s->sstSourceNumber,n},
        {s->rhoK,n},{s->rhoOmega,n},{s->k,n},{s->omega,n},{s->nut,n}};
    appendSstAuditFields(s,fields);return fields;
}
inline std::vector<Field> faceFields(DeviceState* s)
{
    const std::size_t n=s->nFaces;
    return {{s->gasBoundaryRho,n},{s->gasBoundaryUx,n},{s->gasBoundaryUy,n},
        {s->gasBoundaryUz,n},{s->gasBoundaryP,n},{s->gasBoundaryT,n}};
}
template<class T>
int deviceCopy(T* to,const T* from,std::size_t count)
{
    if(!count) return 0;
    const auto error=cudaMemcpy(to,from,count*sizeof(T),cudaMemcpyDeviceToDevice);
    if(error!=cudaSuccess){setLastError("shared gas transaction copy",error);return 1;}
    return 0;
}
inline int copyFields(const std::vector<Field>& fields,double* saved,bool restore)
{
    std::size_t offset=0;
    for(const auto& field:fields)
    {
        if(field.pointer && deviceCopy(restore?field.pointer:saved+offset,
            restore?saved+offset:field.pointer,field.count)) return 1;
        offset+=field.count;
    }
    return 0;
}
inline void releaseSnapshot(SharedGasTrialSnapshot& snapshot)
{
    releaseSharedGasPointer(snapshot.cells);releaseSharedGasPointer(snapshot.faces);
    releaseSharedGasPointer(snapshot.cellStatus);releaseSharedGasPointer(snapshot.faceStatus);
    releaseSharedGasPointer(snapshot.chemistryStatus);releaseSharedGasPointer(snapshot.chemistryAudits);
    releaseSharedGasPointer(snapshot.wallRecords);snapshot.wallRecordBytes=0;
    snapshot.valid=false;
}
inline int ensureSnapshot(DeviceState* s,SharedGasTrialSnapshot& snapshot)
{
    if(snapshot.cells) return 0;
    std::size_t scalarCount=0;for(const auto& f:cellFields(s))scalarCount+=f.count;
    int rc=allocateSharedGasZero(snapshot.cells,scalarCount);
    rc|=allocateSharedGasZero(snapshot.faces,std::size_t(s->nFaces)*6);
    rc|=allocateSharedGasZero(snapshot.cellStatus,s->nCells);
    rc|=allocateSharedGasZero(snapshot.faceStatus,s->nFaces);
    if(s->gasSpecies.mode==ugkwp::GasMode::MixtureChemistry)
    {
        rc|=allocateSharedGasZero(snapshot.chemistryStatus,s->nCells);
        rc|=allocateSharedGasZero(snapshot.chemistryAudits,std::size_t(s->nCells)*3);
        if(!s->gasTrial.chemistryBefore)
            rc|=allocateSharedGasZero(s->gasTrial.chemistryBefore,s->nCells);
        if(!s->gasTrial.chemistryAfter)
            rc|=allocateSharedGasZero(s->gasTrial.chemistryAfter,s->nCells);
    }
    if(rc)releaseSnapshot(snapshot);
    return rc;
}
template<class State> int snapshotWallRecords(State* s,SharedGasTrialSnapshot& saved,bool restore)
{
    if constexpr(ugkwp::GasBoundaryLayerCapability<State>::value)
    {
        if(!s->gasBoundaryLayer.enabled)return 0;
        auto&w=s->gasBoundaryLayer;auto&m=s->gasBoundaryLayerModel;
        struct Bytes{void* pointer;std::size_t count;};const std::size_t n=w.count;
        std::vector<Bytes> fields={{w.exchange,n*sizeof(*w.exchange)},{w.sst,n*sizeof(*w.sst)},
            {w.status,n*sizeof(*w.status)},{w.speciesFlux,n*ugkwp::compiledGasSpecies*sizeof(double)},
            {m.input,n*sizeof(*m.input)},{m.output,n*sizeof(*m.output)},{m.status,n*sizeof(*m.status)}};
        std::size_t total=0;for(const auto& f:fields)total+=f.count;
        if(!saved.wallRecords)
        {
            if(restore || allocateSharedGasZero(saved.wallRecords,total))return 1;
            saved.wallRecordBytes=total;
        }
        if(saved.wallRecordBytes!=total)return 1;
        std::size_t offset=0;
        for(const auto& f:fields)
        {
            if(!f.pointer)return 1;
            const auto error=cudaMemcpy(restore?f.pointer:saved.wallRecords+offset,
                restore?saved.wallRecords+offset:f.pointer,f.count,cudaMemcpyDeviceToDevice);
            if(error!=cudaSuccess){setLastError("wall diagnostic transaction copy",error);return 1;}
            offset+=f.count;
        }
        if(restore)
        {
            s->gasBoundaryLayerStageTime=saved.wallStageTime;
            w.preparedTime=saved.wallAuditStart;w.preparedInterval=saved.wallAuditInterval;
            w.preparedFirstStage=false;++w.generation;
        }
        else
        {
            saved.wallStageTime=s->gasBoundaryLayerStageTime;
            saved.wallAuditStart=w.preparedTime;saved.wallAuditInterval=w.preparedInterval;
        }
    }
    return 0;
}
inline int snapshot(DeviceState* s,SharedGasTrialSnapshot& saved,bool restore)
{
    if(restore && !saved.valid) return 0;
    if(!restore){saved.valid=false;if(ensureSnapshot(s,saved))return 1;}
    int rc=copyFields(cellFields(s),saved.cells,restore)
        ||copyFields(faceFields(s),saved.faces,restore);
    rc|=deviceCopy(restore?s->gasSpecies.cellStatus:saved.cellStatus,
        restore?saved.cellStatus:s->gasSpecies.cellStatus,s->nCells);
    rc|=deviceCopy(restore?s->gasSpecies.faceStatus:saved.faceStatus,
        restore?saved.faceStatus:s->gasSpecies.faceStatus,s->nFaces);
    if(s->gasSpecies.chemistryStatus)
    {
        rc|=deviceCopy(restore?s->gasSpecies.chemistryStatus:saved.chemistryStatus,
            restore?saved.chemistryStatus:s->gasSpecies.chemistryStatus,s->nCells);
        ugkwp::ChemistryAudit<double,ugkwp::compiledGasSpecies>* arrays[]={s->gasSpecies.chemistryAudit,s->gasTrial.chemistryBefore,s->gasTrial.chemistryAfter};
        for(int part=0;part<3;++part)
            rc|=deviceCopy(restore?arrays[part]:saved.chemistryAudits+part*s->nCells,
                restore?saved.chemistryAudits+part*s->nCells:arrays[part],s->nCells);
    }
    rc|=snapshotWallRecords(s,saved,restore);
    if(!restore && !rc)saved.valid=true;
    return rc;
}
inline bool retryableTransport(int value)
{
    const auto code=static_cast<ugkwp::GasTransportCode>(value);
    return code==ugkwp::GasTransportCode::InvalidComposition
        ||code==ugkwp::GasTransportCode::InvalidThermodynamics
        ||code==ugkwp::GasTransportCode::NonFiniteState
        ||code==ugkwp::GasTransportCode::NegativeInventory
        ||code==ugkwp::GasTransportCode::SourceStepLimit;
}
// Only entered after the existing status check fails. No successful-stage
// transfer or additional all-face scan is introduced for error reporting.
template<class State> bool reportBoundaryLayerFailure(State* s)
{
    if constexpr(ugkwp::GasBoundaryLayerCapability<State>::value)
    {
        const auto&w=s->gasBoundaryLayer;const auto&m=s->gasBoundaryLayerModel;
        if(!w.enabled || !w.status || !m.faces || !m.status)return false;
        std::vector<int> codes(w.count);
        if(copyToHost(codes.data(),w.status,codes.size(),"read failed wall statuses"))return true;
        for(int slot=0;slot<w.count;++slot)if(codes[slot])
        {
            int face=-1;
            typename std::remove_pointer<decltype(m.status)>::type detail;
            if(copyToHost(&face,m.faces+slot,1,"read failed wall face")
                || copyToHost(&detail,m.status+slot,1,"read failed wall closure status"))return true;
            char message[256];
            std::snprintf(message,sizeof(message),"boundaryLayer closure failed: face=%d transport=%d wallCode=%d node=%d iteration=%d residual=%.17g",
                face,codes[slot],int(detail.code),detail.node,detail.iteration,detail.residual);
            setLastErrorText(message);return true;
        }
    }
    return false;
}
inline int transportStatus(DeviceState* s)
{
    s->gasTrial.retryableFailure=false;
    std::vector<int> cells(s->nCells),faces(s->nFaces);
    if(copyToHost(cells.data(),s->gasSpecies.cellStatus,cells.size(),"read mixture cell diagnostics")
       ||copyToHost(faces.data(),s->gasSpecies.faceStatus,faces.size(),"read mixture face diagnostics"))return 1;
    bool failed=false,retry=true;
    for(const auto& values:{cells,faces})for(int value:values)if(value)
    {failed=true;retry=retry&&retryableTransport(value);}
    if(failed){s->gasTrial.retryableFailure=retry;if(!reportBoundaryLayerFailure(s))setLastErrorText("shared gas trial rejected by physical cell/face validation");return 1;}
    return 0;
}
inline int chemistryStatus(DeviceState* s)
{
    s->gasTrial.retryableFailure=false;
    if(!s->gasSpecies.chemistryStatus)return 0;
    std::vector<ugkwp::ChemistryStatus> values(s->nCells);
    if(copyToHost(values.data(),s->gasSpecies.chemistryStatus,values.size(),"read mixture chemistry diagnostics"))return 1;
    bool failed=false,retry=true;
    for(const auto& value:values)if(!value)
    {
        failed=true;
        retry=retry && value.code!=ugkwp::ChemistryCode::InvalidModel
            &&value.code!=ugkwp::ChemistryCode::InvalidControls
            &&value.code!=ugkwp::ChemistryCode::InvalidComposition;
    }
    if(failed)
    {s->gasTrial.retryableFailure=retry;setLastErrorText("shared gas chemical trial rejected");return 1;}
    return 0;
}
} // namespace sharedGasTrialDetail

inline void releaseSharedGasTrialStorage(SharedGasTrialStorage& storage)
{
    sharedGasTrialDetail::releaseSnapshot(storage.trial);
    sharedGasTrialDetail::releaseSnapshot(storage.interval);
    releaseSharedGasPointer(storage.chemistryBefore);releaseSharedGasPointer(storage.chemistryAfter);
}
struct SharedGasTrialPolicy
{
    static int beginInterval(DeviceState* s)
    {
        if(s->gasModelPoisoned)return 1;
        s->gasTrial.retryableFailure=false;
        const auto error=cudaDeviceSynchronize();
        if(error!=cudaSuccess){setLastError("begin shared gas interval",error);return 1;}
        return sharedGasTrialDetail::snapshot(s,s->gasTrial.interval,false);
    }
    static int begin(DeviceState* s)
    {
        if(s->gasModelPoisoned)return 1;
        s->gasTrial.retryableFailure=false;
        if(sharedGasTrialDetail::snapshot(s,s->gasTrial.trial,false))return 1;
        if(cudaMemset(s->gasSpecies.cellStatus,0,std::size_t(s->nCells)*sizeof(int))!=cudaSuccess
           ||cudaMemset(s->gasSpecies.faceStatus,0,std::size_t(s->nFaces)*sizeof(int))!=cudaSuccess)return 1;
        if(s->gasSpecies.chemistryStatus && cudaMemset(s->gasSpecies.chemistryStatus,0,
            std::size_t(s->nCells)*sizeof(ugkwp::ChemistryStatus))!=cudaSuccess)return 1;
        return 0;
    }
    static int applySources(DeviceState* s,double dt)
    {
        const int block=s->fixedCellBlockThreads;
        return applyGasGravitySource(s,(s->nCells+block-1)/block,block,dt);
    }
    static int validate(DeviceState* s)
    {
        if(s->gasModelPoisoned)return 1;
        const auto error=cudaDeviceSynchronize();
        if(error!=cudaSuccess){setLastError("validate shared gas trial",error);return 1;}
        const int transport=sharedGasTrialDetail::transportStatus(s);
        const bool transportRetry=s->gasTrial.retryableFailure;
        const int chemistry=sharedGasTrialDetail::chemistryStatus(s);
        const bool chemistryRetry=s->gasTrial.retryableFailure;
        s->gasTrial.retryableFailure=(transport || chemistry) && (!transport || transportRetry) && (!chemistry || chemistryRetry);
        return transport || chemistry;
    }
    static int commit(DeviceState* s){return s->gasModelPoisoned?1:0;}
    static int commitInterval(DeviceState* s){return s->gasModelPoisoned?1:0;}
    static void rollback(DeviceState* s)
    {
        if(sharedGasTrialDetail::snapshot(s,s->gasTrial.trial,true))
        {
            s->gasTrial.retryableFailure=false;s->gasModelPoisoned=true;
            setLastErrorText("shared gas trial rollback failed; resident must be destroyed");
        }
    }
    static void rollbackInterval(DeviceState* s)
    {
        if(sharedGasTrialDetail::snapshot(s,s->gasTrial.interval,true))
        {
            s->gasTrial.retryableFailure=false;s->gasModelPoisoned=true;
            setLastErrorText("shared gas interval rollback failed; resident must be destroyed");
        }
    }
    static bool retryable(DeviceState* s){return s->gasTrial.retryableFailure;}
    static const double* stageVolumes(DeviceState* s,bool){return s->V;}
    static double targetMaxCo(DeviceState* s){return s->gasTrial.targetMaxCo;}
    static int validateTimeStep(DeviceState* s,double)
    {
        if(validate(s))return 1;
        std::vector<double> co(s->nCells),diffusion(s->nCells),sst;
        if(copyToHost(co.data(),s->gasFluxPositivityScale,co.size(),"read trial Courant diagnostics")
            ||copyToHost(diffusion.data(),s->gasDiffusionNumber,diffusion.size(),"read trial diffusion diagnostics"))return 1;
        if(s->hostTurbulenceModel==3)
        {
            sst.resize(s->nCells);
            if(copyToHost(sst.data(),s->sstSourceNumber,sst.size(),"read trial SST stability diagnostics"))return 1;
        }
        for(std::size_t i=0;i<co.size();++i)
            if(!std::isfinite(co[i]) || !std::isfinite(diffusion[i])
                ||co[i]>s->gasTrial.targetMaxCo*(1+1e-12) || diffusion[i]>s->gasTrial.targetMaxCo*(1+1e-12)
                ||(!sst.empty() && (!std::isfinite(sst[i]) || sst[i]>s->gasTrial.targetMaxCo*(1+1e-12))))
            {s->gasTrial.retryableFailure=true;setLastErrorText("shared gas post-source timestep bound exceeded");return 1;}
        return 0;
    }
    static int captureChemistryAudit(DeviceState* s,bool afterTransport)
    {
        if(sharedGasTrialDetail::chemistryStatus(s))return 1;
        return sharedGasTrialDetail::deviceCopy(afterTransport?s->gasTrial.chemistryAfter:s->gasTrial.chemistryBefore,
            s->gasSpecies.chemistryAudit,s->nCells);
    }
};
