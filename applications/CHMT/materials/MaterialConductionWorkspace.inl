struct MaterialConductionWorkspace {
    std::unique_ptr<Foam::fvMesh> mesh;
    std::unique_ptr<Foam::volScalarField> temperature,capacity,conductivity,source;
    ugkp::gpu::ImplicitDiffusion diffusion;
    const Foam::fvMesh* origin=nullptr;
    std::uint64_t topology=0;
    std::vector<Vec3> points;
    Foam::wordList temperatureTypes;
    Foam::string boundaryDefinition;
    static Foam::string boundaryPolicy(const Foam::fvMesh& region){
        Foam::OStringStream output;
        forAll(region.boundaryMesh(),patch){
            output<<region.boundaryMesh()[patch].name()<<Foam::nl;
            region.boundaryMesh()[patch].write(output);
        }
        return output.str();
    }
    bool sameGeometry(const Foam::fvMesh& accepted,const HostMesh& state)const{
        if(!mesh||origin!=&accepted||topology!=state.topologyHash||points.size()!=state.points.size()
            ||mesh->nCells()!=accepted.nCells()||mesh->nFaces()!=accepted.nFaces())return false;
        if(mesh->faces()!=accepted.faces()||mesh->faceOwner()!=accepted.faceOwner()
            ||mesh->faceNeighbour()!=accepted.faceNeighbour()||boundaryDefinition!=boundaryPolicy(accepted))return false;
        for(std::size_t i=0;i<points.size();++i)
            if(points[i].x!=state.points[i].x||points[i].y!=state.points[i].y||points[i].z!=state.points[i].z)return false;
        return true;
    }
    void clearFields(){temperature.reset();capacity.reset();conductivity.reset();source.reset();}
};
bool prepareConductionWorkspace(Foam::fvMesh& acceptedMesh,const HostState& next,
    const PhysicsConfig& p,MaterialConductionWorkspace& workspace,std::string& error){
    const auto n=next.solidMesh.volumes.size();
    if(!validateThermalBoundaries(next.solidMesh,error)){return false;}
    if((acceptedMesh.nCells()!=static_cast<Foam::label>(n)||next.solidMesh.points.size()!=static_cast<std::size_t>(acceptedMesh.nPoints()))){error="CPU material fvMesh topology mismatch";return false;}
    if(!workspace.sameGeometry(acceptedMesh,next.solidMesh)){
        workspace.clearFields();
        workspace.mesh.reset(new Foam::fvMesh(Foam::IOobject(acceptedMesh.name(),acceptedMesh.time().timeName(),acceptedMesh.time(),
            Foam::IOobject::NO_READ,Foam::IOobject::NO_WRITE,false),foamPoints(next.solidMesh.points),
            Foam::faceList(acceptedMesh.faces()),Foam::labelList(acceptedMesh.faceOwner()),Foam::labelList(acceptedMesh.faceNeighbour())));
        auto& mesh=*workspace.mesh;
        Foam::List<Foam::polyPatch*> patches(acceptedMesh.boundaryMesh().size());
        forAll(patches,patch)patches[patch]=acceptedMesh.boundaryMesh()[patch].clone(mesh.boundaryMesh()).ptr();
        mesh.addFvPatches(patches);
        workspace.origin=&acceptedMesh;workspace.topology=next.solidMesh.topologyHash;workspace.points=next.solidMesh.points;
        workspace.boundaryDefinition=MaterialConductionWorkspace::boundaryPolicy(acceptedMesh);
    }
    auto& mesh=*workspace.mesh;
    for(std::size_t c=0;c<n;++c)if(!closeEnough(mesh.V()[c],next.solidMesh.volumes[c],p.tolerances.absoluteGeometry,p.tolerances.relativeGeometry)){
        materialWarning(next,ugkp::diagnostics::Code::ConservationResidual,"OpenFOAM / conservative material volume discrepancy",mesh.V()[c]-next.solidMesh.volumes[c],p.tolerances.absoluteGeometry+p.tolerances.relativeGeometry*absValue(next.solidMesh.volumes[c]));}
    Foam::wordList patchTypes(mesh.boundary().size(),Foam::zeroGradientFvPatchScalarField::typeName);
    forAll(mesh.boundary(),patch){if(mesh.boundary()[patch].coupled()||mesh.boundary()[patch].type()=="empty")patchTypes[patch]=mesh.boundary()[patch].type();
        else{bool hasFixed=false,hasFree=false;forAll(mesh.boundary()[patch],f){const int face=mesh.boundary()[patch].start()+f;

            const bool fixed=next.solidMesh.boundaryKind[face]!=BoundaryKind::Interface&&fixedTemperature(next.solidMesh.thermalBoundary.empty()?nullptr:next.solidMesh.thermalBoundary.data(),face);
            hasFixed=hasFixed||fixed;hasFree=hasFree||!fixed;}if(hasFixed&&hasFree){error="mixed per-face material thermal policy within one native patch";return false;}if(hasFixed)patchTypes[patch]=Foam::fixedValueFvPatchScalarField::typeName;}}
    if(!workspace.temperature||workspace.temperatureTypes!=patchTypes){
        workspace.temperature.reset(new Foam::volScalarField(Foam::IOobject("CHMTMaterialTemperature",mesh.time().timeName(),mesh,Foam::IOobject::NO_READ,Foam::IOobject::NO_WRITE,false),mesh,
            Foam::dimensionedScalar("initial",Foam::dimTemperature,300),patchTypes));
        workspace.temperatureTypes=patchTypes;
    }
    auto& temperature=*workspace.temperature;
    forAll(mesh.boundary(),patch)if(patchTypes[patch]==Foam::fixedValueFvPatchScalarField::typeName)forAll(mesh.boundary()[patch],f){const int face=mesh.boundary()[patch].start()+f;temperature.boundaryFieldRef()[patch][f]=next.solidMesh.boundaryPrimitive[face].temperature;}
    Foam::wordList coefficientTypes(mesh.boundary().size(),Foam::zeroGradientFvPatchScalarField::typeName);
    forAll(mesh.boundary(),patch)if(mesh.boundary()[patch].coupled()||mesh.boundary()[patch].type()=="empty")coefficientTypes[patch]=mesh.boundary()[patch].type();
    const Foam::dimensionSet capacityDimensions=Foam::dimEnergy/Foam::dimVolume/Foam::dimTemperature;
    const Foam::dimensionSet conductivityDimensions=Foam::dimPower/Foam::dimLength/Foam::dimTemperature;
    ugkp::gpu::installGpuImplicitDiffusion();
    mesh.schemes().setFluxRequired(temperature.name());
    if(!workspace.capacity){
        workspace.capacity.reset(new Foam::volScalarField(Foam::IOobject("CHMTMaterialSecant",mesh.time().timeName(),mesh,Foam::IOobject::NO_READ,Foam::IOobject::NO_WRITE,false),mesh,Foam::dimensionedScalar("zero",capacityDimensions,0),coefficientTypes));
        workspace.conductivity.reset(new Foam::volScalarField(Foam::IOobject("CHMTMaterialKappa",mesh.time().timeName(),mesh,Foam::IOobject::NO_READ,Foam::IOobject::NO_WRITE,false),mesh,Foam::dimensionedScalar("zero",conductivityDimensions,0),coefficientTypes));
        workspace.source.reset(new Foam::volScalarField(Foam::IOobject("CHMTMaterialEnergySource",mesh.time().timeName(),mesh,Foam::IOobject::NO_READ,Foam::IOobject::NO_WRITE,false),mesh,Foam::dimensionedScalar("zero",Foam::dimPower/Foam::dimVolume,0)));
    }
    return true;
}
