MieDoRadiationWorkspace::MieDoRadiationWorkspace
(
    std::unique_ptr<MieDoRadiationWorkspaceData> data
) noexcept
:
    data_(std::move(data))
{}

void MieDoRadiationWorkspace::acceptParentSolve() noexcept
{
    data_->intensitiesInitialized=true;
}

MieDoRadiationWorkspace::~MieDoRadiationWorkspace() = default;

MieDoRadiationWorkspace::MieDoRadiationWorkspace
(
    MieDoRadiationWorkspace&& other
) noexcept = default;

MieDoRadiationWorkspace& MieDoRadiationWorkspace::operator=
(
    MieDoRadiationWorkspace&& other
) noexcept = default;

MieDoRadiationSolver::MieDoRadiationSolver
(
    const fvMesh& mesh,
    const dictionary& radiationProperties,
    const MieTable& mieTable,
    const labelList& coupledSolidPatchIds,
    const MieDoControls& controls
)
:
    data_
    (
        std::make_shared<MieDoRadiationSolverData>
        (
            mesh,
            radiationProperties,
            mieTable,
            coupledSolidPatchIds,
            controls
        )
    )
{}

void MieDoRadiationSolver::updateGeometry()
{
    auto next=std::make_shared<MieDoRadiationSolverData>(data_->mesh,
        data_->radiationDictionary,data_->mieTable,data_->coupledPatchIds,data_->controls);
    next->geometryFamily=data_->geometryFamily;data_=std::move(next);
}

const fvMesh& MieDoRadiationSolver::mesh() const noexcept
{
    return data_->mesh;
}

const MieTable& MieDoRadiationSolver::mieTable() const noexcept
{
    return data_->mieTable;
}

const labelList& MieDoRadiationSolver::coupledSolidPatchIds() const noexcept
{
    return data_->coupledPatchIds;
}

const MieDoControls& MieDoRadiationSolver::controls() const noexcept
{
    return data_->controls;
}

void MieDoRadiationSolver::validateSnapshot
(
    const MieDoSnapshot& snapshot
) const
{
    if (!data_)
    {
        solveFailure("MieDoRadiationSolver has no immutable implementation state");
    }
    data_->validateSnapshot(snapshot);
}

MieDoRadiationWorkspace MieDoRadiationSolver::makeWorkspace() const
{
    if (!data_)
    {
        solveFailure("MieDoRadiationSolver has no immutable implementation state");
    }
    return MieDoRadiationWorkspace
    (
        std::unique_ptr<MieDoRadiationWorkspaceData>
        (
            new MieDoRadiationWorkspaceData(*this, data_)
        )
    );
}


namespace {
std::uint64_t warmLayout(const MieDoRadiationSolverData& data) {
    std::uint64_t hash=1469598103934665603ULL;
    const auto append=[&](const auto& value){const auto* p=reinterpret_cast<const unsigned char*>(&value);
        for(std::size_t i=0;i<sizeof(value);++i){hash^=p[i];hash*=1099511628211ULL;}};

    append(data.mesh.nCells());
    for(const auto v:data.mesh.owner())append(v);
    for(const auto v:data.mesh.neighbour())append(v);
    for(const auto& patch:data.mesh.boundary()){append(patch.start());append(patch.size());}
    for(const auto& direction:data.directions)for(int i=0;i<3;++i){const double v=direction[i];append(v);}
    for(const auto weight:data.weights){const double v=weight;append(v);}
    return hash;
}
}
void MieDoRadiationSolver::freezeWarmStart(MieDoRadiationWorkspace& workspace,
    ugkp::gpu::RadiationWarmStart& output,void* nextWriter) const {
    auto& scratch=*workspace.data_;
    if(scratch.intensitiesInitialized)scratch.prepare(data_);
    output.describe(data_->mesh.nCells(),data_->directions.size(),warmLayout(*data_),scratch.intensitiesInitialized);
    if(scratch.intensitiesInitialized)scratch.gpu->freezeIntensities(output,nextWriter);
}
bool MieDoRadiationSolver::writeWarmStart(MieDoRadiationWorkspace& workspace,const std::string& path) const {
    ugkp::gpu::RadiationWarmStart output;
    freezeWarmStart(workspace,output);
    std::string error;
    if(output.write(path,error))return true;
    ugkp::diagnostics::hostWarning(ugkp::diagnostics::Code::RestartMetadata,
        {42002,__FILE__,__func__,__LINE__,error+"; next solve can initialize its angular field"});
    return false;
}
bool MieDoRadiationSolver::readWarmStart(MieDoRadiationWorkspace& workspace,const std::string& path) const {
    auto& scratch=*workspace.data_;
    scratch.intensitiesInitialized=false;
    ugkp::gpu::ReadWarmStart saved;std::string error;
    if(!ugkp::gpu::readWarmStartFile(path,data_->mesh.nCells(),data_->directions.size(),warmLayout(*data_),saved,error)){
        ugkp::diagnostics::hostWarning(ugkp::diagnostics::Code::RestartMetadata,
            {42002,__FILE__,__func__,__LINE__,error+"; starting angular iteration cold"});
        return false;
    }
    if(saved.metadataChanged)ugkp::diagnostics::hostWarning(ugkp::diagnostics::Code::RestartMetadata,
        {42002,__FILE__,__func__,__LINE__,"DOM metadata changed; readable angular guess uses the saved field width at "+path});
    if(saved.initialized){
        scratch.prepare(data_);
        scratch.gpu->uploadIntensities(saved.values);
        scratch.intensitiesInitialized=true;
    }
    return true;
}

namespace {
ugkp::gpu::RadiationSnapshot residentRadiationInput(const MieDoRadiationSolverData& data,
    const MieDoSnapshot& snapshot) {
    const auto& mesh=data.mesh;const label nCells=mesh.nCells();
    ugkp::gpu::RadiationSnapshot input;
    input.resident=snapshot.resident;
    input.thermalCoupling=snapshot.thermalCoupling;
    if(!input.resident.enabled){
        input.epsilon.assign(snapshot.epsilonS.begin(),snapshot.epsilonS.begin()+nCells);
        input.temperature.assign(snapshot.particleTemperatureK.begin(),snapshot.particleTemperatureK.begin()+nCells);
        input.diameter.assign(snapshot.representativeDiameterM.begin(),snapshot.representativeDiameterM.begin()+nCells);
    }
    forAll(mesh.boundary(),patchI)forAll(mesh.boundary()[patchI],faceI){
        input.wallTemperature.push_back(data.boundaryTemperature(snapshot,patchI,faceI));
        input.emissivity.push_back(data.boundaryEmissivity(snapshot,patchI,faceI));
        input.residentWallTemperature.push_back(data.isCoupledPatch[patchI]||!data.patchControls[patchI].hasSpecifiedTemperature);
        input.contactEmissivity.push_back(data.isCoupledPatch[patchI]);
    }
    return input;
}
}

MieDoResult MieDoRadiationSolver::solve(const MieDoSnapshot& snapshot) const
{
    MieDoRadiationWorkspace workspace = makeWorkspace();
    auto result=solve(snapshot, workspace);result.resident={};return result;
}

MieDoResult MieDoRadiationSolver::solve
(
    const MieDoSnapshot& snapshot,
    MieDoRadiationWorkspace& workspace,
    bool exportFields
) const
{
    if (!data_)
    {
        solveFailure("MieDoRadiationSolver has no immutable implementation state");
    }
    if (!workspace.data_)
    {
        solveFailure("Mie DO workspace has no storage");
    }
    const MieDoRadiationSolverData& data = *data_;
    data.validateSnapshot(snapshot);
    MieDoRadiationWorkspaceData& scratch = *workspace.data_;
    scratch.prepare(data_);
    const fvMesh& mesh=data.mesh;
    const label nCells=mesh.nCells();
    const auto input=residentRadiationInput(data,snapshot);
    scratch.gpu->setThermalIterationHook(snapshot.thermalIterationHook, snapshot.thermalCoupling.graphOwner, snapshot.thermalCoupling.graphRevision);
    const bool warm=scratch.intensitiesInitialized;scratch.intensitiesInitialized=false;
    ugkp::gpu::RadiationResult output;
    try {output=scratch.gpu->solve(input,warm,exportFields);}
    catch(const ugkp::gpu::RadiationError& error){solveFailure(error.what(),error.iteration,error.value,error.cell,error.patch,error.face);}
    catch(const std::exception& error){solveFailure(std::string("CUDA Mie DO solve failed: ")+error.what());}
    if(exportFields&&hostDiagnosticWarningsEnabled()){
        const auto effectiveControls=radiationControls(data);
        for(label cellI=0;cellI<nCells;++cellI){const auto& phase=output.phaseMetrics[cellI];
            if(phase.rowError>effectiveControls.phaseTolerance||phase.columnError>effectiveControls.phaseTolerance)
                WarningInFunction<<"DO phase balance iteration budget exhausted; time="<<mesh.time().timeName()<<"; cell="<<cellI
                    <<"; iterations="<<phase.iterations<<"; rowResidual="<<phase.rowError<<"; columnResidual="<<phase.columnError
                    <<"; limit="<<effectiveControls.phaseTolerance<<"; retaining computed phase probabilities"<<endl;
        }
        for(std::size_t direction=0;direction<output.directionWarnings.size();++direction){const auto& warning=output.directionWarnings[direction];
            const char* reason=warning.stopReason==4?"non-finite arithmetic":warning.stopReason==5?"finite-precision stagnation":warning.stopReason==3?"Arnoldi breakdown without true-residual convergence":"iteration budget exhausted";
            if(warning.linearIteration>0||warning.stopReason)WarningInFunction<<"DO FGMRES stopped: "<<reason<<"; time="<<mesh.time().timeName()
                <<"; direction="<<direction<<"; sourceIteration="<<warning.sourceIteration<<"; linearIterations="<<warning.linearIteration
                <<"; residual="<<warning.residual<<"; limit="<<warning.limit
                <<"; relativeResidual="<<warning.linearRelativeResidual<<"; relativeLimit="<<warning.linearRelativeLimit<<"; retaining computed intensity"<<endl;
        }
        if(output.maximumRelativeChange>effectiveControls.sourceTolerance)
            WarningInFunction<<"DO source iteration budget exhausted; time="<<mesh.time().timeName()<<"; iterations="<<output.sourceIterations
                <<"; residual="<<output.maximumRelativeChange<<"; limit="<<effectiveControls.sourceTolerance<<"; retaining computed intensities"<<endl;
    }
    scratch.intensitiesInitialized=true;return publishedRadiationResult(mesh,scratch,output);
}

ugkp::gpu::DiscreteOrdinateRadiation& MieDoRadiationSolver::prepareParent
(const MieDoSnapshot& snapshot,MieDoRadiationWorkspace& workspace) const {
    if(!data_||!workspace.data_)solveFailure("parent radiation workspace is unavailable");
    if(!snapshot.resident.enabled)solveFailure("parent radiation requires resident input fields");
    data_->validateSnapshot(snapshot);
    auto& scratch=*workspace.data_;scratch.prepare(data_);
    scratch.gpu->setThermalIterationHook(snapshot.thermalIterationHook,
        snapshot.thermalCoupling.graphOwner,snapshot.thermalCoupling.graphRevision);
    scratch.gpu->prepareParentInputs(residentRadiationInput(*data_,snapshot),scratch.intensitiesInitialized);
    return *scratch.gpu;
}

MieDoResult MieDoRadiationSolver::collect(MieDoRadiationWorkspace& workspace) const {
    if(!workspace.data_||!workspace.data_->gpu)solveFailure("radiation output requested before solve");
    return publishedRadiationResult(data_->mesh,*workspace.data_,workspace.data_->gpu->collect());
}
