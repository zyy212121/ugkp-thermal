#pragma once
// Host adapter for the native resident backend. No transport/thermo/chemistry
// formula is implemented here: model parsing and capability gates are common.
inline ugkwp::GasCapabilityRequest sharedGasCapabilityRequest
(const DeviceState* state,ugkwp::GasMode mode,int wallTreatment)
{
    ugkwp::GasCapabilityRequest request;
    request.mode=mode;request.fluxScheme=state->hostGasFluxScheme;
    request.reconstruction=state->hostGasReconstruction;request.limiter=state->hostGasLimiter;
    request.timeIntegrator=state->hostGasTimeIntegrator;request.turbulenceModel=state->hostTurbulenceModel;
    request.sstWallTreatment=wallTreatment;

    request.particleCoupling=false;
    return request;
}
inline bool sharedGasViewRollbackFailed(DeviceState* state)
{
    if(!syncSharedGasSpecies(state)) return false;
    // The device may still hold addresses from the rejected view. Forbid all
    // further resident operations and retain those allocations until teardown.
    state->gasModelPoisoned=true;
    setLastErrorText("shared gas device view rollback failed; resident must be destroyed");
    return true;
}
inline int discardSharedGasSpecies(DeviceState* state,
    ugkwp::GasSpeciesState<double,ugkwp::compiledGasSpecies>& owner)
{
    if(!releaseSharedGasSpecies(owner))return 0;

    state->gasRejectedView=owner;owner={};state->gasModelPoisoned=true;
    return 1;
}
template<class T>inline int discardSharedGasField(DeviceState* state,T*& owner,T*& rejected)
{
    if(!releaseSharedGasPointer(owner))return 0;
    rejected=owner;owner=nullptr;state->gasModelPoisoned=true;return 1;
}
extern "C" int ugkwpGpuResidentStrictQueryGasModelCapabilitiesV1
(void* handle,ugkwpGpuIpc::GasModelCapabilitiesV1* capabilities)
{
    if(validateState(asState(handle),"gas model query") || !capabilities) return 1;
    std::uint32_t modes=1;
    ugkwp::GasCapabilityRequest request;
    request.mode=ugkwp::GasMode::MixtureFrozen;
    if(ugkwp::validateGasCapabilities(request)) modes|=2;
    request.mode=ugkwp::GasMode::MixtureChemistry;
    if(ugkwp::validateGasCapabilities(request)) modes|=4;
    *capabilities={ugkwpGpuIpc::gasModelApiVersion,ugkwp::compiledGasSpecies,modes,0};
    return 0;
}
extern "C" int ugkwpGpuResidentStrictConfigureGasModelV1
(void* handle,const ugkwpGpuIpc::GasModelConfigureArgsV1* args,
 const char* modelText,const char* mechanismText)
{
    DeviceState* state=asState(handle);
    std::uint64_t bytes=0;
    if(validateState(state,"gas model configuration") || !args || !modelText
       || !ugkwpGpuIpc::gasModelPayloadBytes(*args,ugkwp::compiledGasSpecies,bytes)
       || (args->mechanismBytes && !mechanismText))
    {setLastErrorText("invalid shared gas model payload");return 1;}
    auto accepted=state->gasSpecies;
    const bool replacing=accepted.rho!=nullptr;
    ugkwp::GasSpeciesState<double,ugkwp::compiledGasSpecies> candidate;
    try
    {
        const auto model=ugkwp::parseGasModelProperties(std::string(modelText,args->modelBytes));
        const auto hostThermo=model.thermoView<ugkwp::compiledGasSpecies>();
        const auto request=sharedGasCapabilityRequest(state,model.mode,state->sstWallTreatment);
        const auto capability=ugkwp::validateGasCapabilities(request);
        if(!capability) throw std::runtime_error(capability.message);
        if(model.mode!=ugkwp::GasMode::SingleLegacy
            && ((request.fluxScheme!=1&&request.fluxScheme!=2)||(request.reconstruction!=0&&request.reconstruction!=1)))
        {
            using namespace ugkp::diagnostics;
            if(state->diagnosticOwner){
                state->diagnosticOwner->recordHostFatal(makeRecord(Code::MissingOperator,2607,26,0,
                    EntityKind::None,-1,request.fluxScheme,request.reconstruction));
                state->diagnosticOwner->dispatchPending();
            }
            setLastErrorText("selected mixture operator is not implemented; supported flux create codes are 1 (Rusanov) and 2 (Kurganov/HLL), reconstruction codes 0 (firstOrder) and 1 (MUSCL)");
            return 1;
        }
        ugkwp::GasMechanismConfiguration mechanism;
        if(model.mode==ugkwp::GasMode::MixtureChemistry)
        {
            mechanism=ugkwp::parseGasMechanismProperties<ugkwp::compiledGasSpecies>
                (std::string(mechanismText,args->mechanismBytes),hostThermo,model.phase,model.speciesNames);
        }
        const std::size_t cells=static_cast<std::size_t>(state->nCells),faces=static_cast<std::size_t>(state->nFaces);
        const std::size_t species=ugkwp::compiledGasSpecies;
        int rc=0;
#define ALLOC_SHARED(Name,Count) rc|=allocateSharedGasZero(candidate.Name,(Count))
        if (!replacing)
        {
        ALLOC_SHARED(rho,species*cells); ALLOC_SHARED(initial,species*cells);
        ALLOC_SHARED(flux,species*faces); ALLOC_SHARED(gradX,species*cells);
        ALLOC_SHARED(gradY,species*cells); ALLOC_SHARED(gradZ,species*cells);
        ALLOC_SHARED(limiter,cells); ALLOC_SHARED(boundaryMassFraction,species*faces);
        ALLOC_SHARED(compositionBoundaryFixed,faces); ALLOC_SHARED(positivityScale,species*cells);
        ALLOC_SHARED(soundSpeed,cells); ALLOC_SHARED(heatCapacity,cells);
        ALLOC_SHARED(gasConstant,cells); ALLOC_SHARED(cellStatus,cells); ALLOC_SHARED(faceStatus,faces);
        }
        if(model.mode==ugkwp::GasMode::MixtureChemistry && !accepted.chemistryStatus)
        {ALLOC_SHARED(chemistryStatus,cells);}
#undef ALLOC_SHARED
        // Copy only canonical immutable metadata into separate device allocations.
        candidate.thermo=hostThermo;
        candidate.thermo.diagnostics=state->diagnostics;
        candidate.thermo.species=nullptr;candidate.thermo.coefficients=nullptr;candidate.thermo.elementComposition=nullptr;
        rc|=uploadSharedGasTable(candidate.thermo.species,model.species.data(),model.species.size());
        rc|=uploadSharedGasTable(candidate.thermo.coefficients,model.coefficients.data(),model.coefficients.size());
        rc|=uploadSharedGasTable(candidate.thermo.elementComposition,model.elementComposition.data(),model.elementComposition.size());
        if(model.diffusionModel==ugkwp::GasDiffusionModel::Constant)
            rc|=uploadSharedGasTable(candidate.diffusivity,model.diffusionCoefficients.data(),species);
        candidate.turbulentSchmidt=model.turbulentSchmidt;
        candidate.chemistryControls=model.chemistryControls;
        candidate.thermoControls=model.chemistryControls.thermo;
        if(model.mode==ugkwp::GasMode::MixtureChemistry)
        {
            candidate.mechanism=mechanism.mechanismView<ugkwp::compiledGasSpecies>();
            candidate.mechanism.reactions=nullptr;candidate.mechanism.reactants=nullptr;candidate.mechanism.products=nullptr;
            candidate.mechanism.efficiencies=nullptr;candidate.mechanism.stoichiometricBasis=nullptr;
            rc|=uploadSharedGasTable(candidate.mechanism.reactions,mechanism.reactions.data(),mechanism.reactions.size());
            rc|=uploadSharedGasTable(candidate.mechanism.reactants,mechanism.reactants.data(),mechanism.reactants.size());
            rc|=uploadSharedGasTable(candidate.mechanism.products,mechanism.products.data(),mechanism.products.size());
            rc|=uploadSharedGasTable(candidate.mechanism.efficiencies,mechanism.efficiencies.data(),mechanism.efficiencies.size());
            rc|=uploadSharedGasTable(candidate.mechanism.stoichiometricBasis,mechanism.stoichiometricBasis.data(),mechanism.stoichiometricBasis.size());
        }
        if(rc) {discardSharedGasSpecies(state,candidate);return 1;}
        candidate.mode=model.mode;
        const bool keepChemistry=accepted.chemistryStatus!=nullptr;
        auto exchangePending=[&]()
        {
#define KEEP_SHARED(Name) std::swap(candidate.Name,accepted.Name)
            KEEP_SHARED(rho); KEEP_SHARED(initial); KEEP_SHARED(flux);
            KEEP_SHARED(gradX); KEEP_SHARED(gradY); KEEP_SHARED(gradZ);
            KEEP_SHARED(limiter); KEEP_SHARED(boundaryMassFraction);
            KEEP_SHARED(compositionBoundaryFixed); KEEP_SHARED(positivityScale);
            KEEP_SHARED(soundSpeed); KEEP_SHARED(heatCapacity); KEEP_SHARED(gasConstant);
            KEEP_SHARED(cellStatus); KEEP_SHARED(faceStatus);
            if (keepChemistry)
            { KEEP_SHARED(chemistryStatus); }
#undef KEEP_SHARED
        };
        if (replacing) exchangePending();
        state->gasSpecies=candidate;
        if(syncSharedGasSpecies(state))
        {
            if (replacing) exchangePending();
            state->gasSpecies=accepted;
            if(sharedGasViewRollbackFailed(state)) state->gasRejectedView=candidate;
            else discardSharedGasSpecies(state,candidate);
            return 1;
        }
        invalidateSharedGasConfiguration(state);
        candidate={};
        if(discardSharedGasSpecies(state,accepted))return 1;
        return 0;
    }
    catch(const std::exception& error)
    {discardSharedGasSpecies(state,candidate);setLastErrorText(error.what());return 1;}
}
extern "C" int ugkwpGpuResidentStrictUploadSpeciesV1
(void* handle,const ugkwpGpuIpc::GasSpeciesIdentityV1* identity,const double* values)
{
    DeviceState* state=asState(handle);
    (void)identity;
    if(validateState(state,"species upload") || !values || !state->gasSpecies.rho)
    {setLastErrorText("missing species upload storage or input");return 1;}
    // A failed transfer must not leave a partially populated resident field.
    // Stage the complete input before publishing a new borrowed device view.
    double* staged=nullptr;
    const std::size_t count=std::size_t(state->nCells)*ugkwp::compiledGasSpecies;
    if(allocateSharedGasZero(staged,count)
       ||copyToDevice(staged,values,count,"upload initial species density"))
    {discardSharedGasField(state,staged,state->gasRejectedView.rho);return 1;}
    double* accepted=state->gasSpecies.rho;
    state->gasSpecies.rho=staged;
    if(syncSharedGasSpecies(state))
    {
        state->gasSpecies.rho=accepted;
        if(sharedGasViewRollbackFailed(state)) state->gasRejectedView.rho=staged;
        else discardSharedGasField(state,staged,state->gasRejectedView.rho);
        return 1;
    }
    if(releaseSharedGasPointer(accepted)){
        state->gasRejectedView.rho=accepted;state->gasModelPoisoned=true;return 1;}
    invalidateSharedGasConfiguration(state,false);
    state->gasSpeciesUploaded=true;
    return 0;
}
extern "C" int ugkwpGpuResidentStrictDownloadSpeciesV1
(void* handle,const ugkwpGpuIpc::GasSpeciesIdentityV1* identity,double* values)
{
    DeviceState* state=asState(handle);
    (void)identity;
    if(validateState(state,"species download") || !values || !state->gasSpecies.rho)
    {setLastErrorText("missing species download storage or output");return 1;}
    std::vector<double> staged(std::size_t(state->nCells)*ugkwp::compiledGasSpecies);
    if(copyToHost(staged.data(),state->gasSpecies.rho,staged.size(),"download accepted species density"))return 1;
    std::copy(staged.begin(),staged.end(),values);
    return state->diagnosticOwner->completeReadback(0,!staged.empty());
}
extern "C" int ugkwpGpuResidentStrictUploadSpeciesBoundaryV1
(void* handle,const ugkwpGpuIpc::GasSpeciesIdentityV1* identity,const int* fixed,const double* values)
{
    DeviceState* state=asState(handle);
    (void)identity;
    if(validateState(state,"species boundary upload") || !fixed || !values
       || !state->gasSpecies.compositionBoundaryFixed || !state->gasSpecies.boundaryMassFraction)
    {setLastErrorText("missing species boundary storage or input");return 1;}
    int* stagedFixed=nullptr;
    double* stagedValues=nullptr;
    const std::size_t count=std::size_t(state->nFaces)*ugkwp::compiledGasSpecies;
    if(allocateSharedGasZero(stagedFixed,state->nFaces)
       ||allocateSharedGasZero(stagedValues,count)
       ||copyToDevice(stagedFixed,fixed,state->nFaces,"upload species boundary operators")
       ||copyToDevice(stagedValues,values,count,"upload species boundary composition"))
    {
        discardSharedGasField(state,stagedFixed,state->gasRejectedView.compositionBoundaryFixed);discardSharedGasField(state,stagedValues,state->gasRejectedView.boundaryMassFraction);
        return 1;
    }
    int* acceptedFixed=state->gasSpecies.compositionBoundaryFixed;
    double* acceptedValues=state->gasSpecies.boundaryMassFraction;
    state->gasSpecies.compositionBoundaryFixed=stagedFixed;
    state->gasSpecies.boundaryMassFraction=stagedValues;
    if(syncSharedGasSpecies(state))
    {
        state->gasSpecies.compositionBoundaryFixed=acceptedFixed;
        state->gasSpecies.boundaryMassFraction=acceptedValues;
        if(sharedGasViewRollbackFailed(state))
        {
            state->gasRejectedView.compositionBoundaryFixed=stagedFixed;
            state->gasRejectedView.boundaryMassFraction=stagedValues;
        }
        else
        {discardSharedGasField(state,stagedFixed,state->gasRejectedView.compositionBoundaryFixed);discardSharedGasField(state,stagedValues,state->gasRejectedView.boundaryMassFraction);}
        return 1;
    }
    invalidateSharedGasConfiguration(state,false);
    const int releasedFixed=releaseSharedGasPointer(acceptedFixed),releasedValues=releaseSharedGasPointer(acceptedValues);
    if(releasedFixed||releasedValues){state->gasRejectedView.compositionBoundaryFixed=acceptedFixed;
        state->gasRejectedView.boundaryMassFraction=acceptedValues;state->gasModelPoisoned=true;return 1;}
    return 0;
}
