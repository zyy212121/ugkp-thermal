#pragma once
// Optional diagnostic storage and strategy are selected once during startup.
// The solve kernel captures its real workspace before the worker reuses it.
extern "C" int ugkwpGpuResidentStrictEnableBoundaryLayerProfilesV1(void* handle)
{
    DeviceState* s=asState(handle);
    if(validateState(s,"enable boundaryLayer profiles"))return 1;
    auto& model=s->gasBoundaryLayerModel;
    if(!s->gasBoundaryLayer.enabled ||
       (model.config.model!=ugkwp::gaswall::BoundaryLayerModel::ReactingSst
        &&model.config.model!=ugkwp::gaswall::BoundaryLayerModel::EquilibriumOde))
    {setLastErrorText("node profiles are available for finiteRate/reactingSst and equilibriumOde boundary layers");return 1;}
    const std::size_t count=std::size_t(s->gasBoundaryLayer.count)*model.config.nodes*(7+ugkwp::compiledGasSpecies);
    double* profile=model.profile;
    if(!profile && allocateSharedGasZero(profile,count))
    {model.profile=profile;return 1;}
    auto configured=model;configured.profile=profile;
    const cudaError_t error=cudaMemcpy(
        reinterpret_cast<unsigned char*>(s->deviceState)+offsetof(DeviceState,gasBoundaryLayerModel),
        &configured,sizeof(configured),cudaMemcpyHostToDevice);
    if(error!=cudaSuccess)
    {
        // Preserve allocation ownership for teardown even on a failed publication.
        model.profile=profile;
        setLastError("publish wall profile storage",error);return 1;
    }
    model=configured;
    invalidateSharedGasConfiguration(s);
    s->gasBoundaryLayerLaunch=&launchBoundaryLayer<true>;
    return 0;
}
extern "C" int ugkwpGpuResidentStrictDownloadBoundaryLayerProfilesV1
(void* handle,std::uint32_t walls,std::uint32_t nodes,std::uint32_t species,double* profiles,double* stageTime)
{
    DeviceState* s=asState(handle);
    if(validateState(s,"download boundaryLayer profiles"))return 1;
    const auto& model=s->gasBoundaryLayerModel;
    if(!model.profile || !profiles || !stageTime || walls!=std::uint32_t(s->gasBoundaryLayer.count)
        || nodes!=std::uint32_t(model.config.nodes) || species!=ugkwp::compiledGasSpecies)
    {setLastErrorText("wall node profile storage or dimensions unavailable");return 1;}
    std::vector<ugkwp::GasBoundaryLayerExchange<double>> exchange(walls);
    if(copyToHost(exchange.data(),s->gasBoundaryLayer.exchange,walls,"read wall profile readiness"))return 1;
    if(s->diagnosticOwner->completeReadback(0,walls!=0))return 1;
    for(const auto& wall:exchange)if(!wall.ready)
    {setLastErrorText("wall node profile has not yet been accepted");return 1;}
    const std::size_t count=std::size_t(walls)*nodes*(7+species);
    if(copyToHost(profiles,model.profile,count,"read solved wall profile nodes"))return 1;
    *stageTime=s->gasBoundaryLayerStageTime;
    return s->diagnosticOwner->completeReadback(0,count!=0);
}

extern "C" int ugkwpGpuResidentStrictDownloadBoundaryLayerParticleReservoirsV1
(void* handle,std::uint32_t walls,std::uint32_t nodes,std::uint32_t species,
 double* rows,double* interval,int* available)
{
 DeviceState*s=asState(handle);
 if(validateState(s,"download wall particle reservoirs"))return 1;
 const auto&m=s->gasBoundaryLayerModel;
 if(!rows||!interval||!available||walls!=std::uint32_t(s->gasBoundaryLayer.count)
    ||nodes!=std::uint32_t(m.config.nodes)||species!=ugkwp::compiledGasSpecies)
 {setLastErrorText("wall particle reservoir output dimensions unavailable");return 1;}
 *available=0;if(!m.particleSource)return 0;
 bool active=false;
 if(copyToHost(&active,&s->deviceState->gasBoundaryLayerModel.particleCoupling,1,"read wall particle epoch"))return 1;
 if(s->diagnosticOwner->completeReadback(0,true))return 1;
 if(!active)return 0;
 const std::size_t count=std::size_t(walls)*nodes;
 std::vector<double> state(count*(species+7)),source(count*4),heat(count);
 if(copyToHost(state.data(),m.particleReservoir,state.size(),"read node gas inventories")
  ||copyToHost(source.data(),m.particleSource,source.size(),"read accepted node sources")
  ||copyToHost(heat.data(),m.particleHeat,heat.size(),"read accepted node heat")
  ||copyToHost(interval,&s->deviceState->gasBoundaryLayerModel.particleIntervalDt,1,"read node source dt")
  ||copyToHost(interval+1,&s->deviceState->gasBoundaryLayerModel.particleIntervalTime,1,"read node source time"))return 1;
 if(s->diagnosticOwner->completeReadback(0,true))return 1;
 for(std::size_t i=0;i<count;++i){double*row=rows+i*(species+12);
  std::copy_n(state.data()+i*(species+7),species+7,row);
  std::copy_n(source.data()+i*4,4,row+species+7);row[species+11]=heat[i];
 }
 *available=1;return 0;
}
