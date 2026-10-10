#ifndef UGKWP_BOUNDARY_LAYER_INPUT_H
#define UGKWP_BOUNDARY_LAYER_INPUT_H

#include "fvCFD.H"
#include "wallPolyPatch.H"
#include "GpuBackendApi.H"
#include "gasTransport/GasModelIO.H"
#include "gasTransport/GasCapabilities.H"
#include "gasWall/WallGeometryBuilder.H"
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <vector>

namespace Foam
{
// Host-only dictionary, immutable mesh and diagnostic adapter. All wall-profile
// evaluation and physical gas inventories remain in the resident backend.
class BoundaryLayerInput
{
    bool active_=false,prepared_=false;
    ugkwpGpuIpc::BoundaryLayerConfigV1 config_;
    ugkwp::gaswall::WallGeometryOptions<double> geometryOptions_;
    ugkwp::gaswall::WallGeometryCache<double> geometryCache_;
    std::vector<std::string> speciesNames_;
    std::vector<int> faces_,quadratureOffsets_,matchingOffsets_,matchingCells_;
    std::vector<double> geometry_,quadratureDistance_,quadratureWeight_,matchingWeight_,speciesFlux_;

    static scalar coefficient(const dictionary& d,const word& name,const word& alias,scalar fallback)
    {
        if(d.found(name) && d.found(alias))
            FatalErrorInFunction<<"boundaryLayer cannot specify both "<<name<<" and "<<alias<<exit(FatalError);
        return d.lookupOrDefault<scalar>(d.found(alias)?alias:name,fallback);
    }
    static std::uint32_t checkedCount(std::size_t count,const char* name)
    {
        if(count>std::size_t(std::numeric_limits<int>::max()))
            FatalErrorInFunction<<"boundaryLayer "<<name<<" exceeds compact index capacity"<<exit(FatalError);
        return static_cast<std::uint32_t>(count);
    }
    static void check(int result,const char* operation)
    {
        if(result) FatalErrorInFunction<<operation<<": "<<ugkwpGpuResidentStrictLastError()<<exit(FatalError);
    }
    void readSpeciesFlux(const Time& time,const fvMesh& mesh)
    {
        speciesFlux_.assign(speciesNames_.size()*faces_.size(),0.0);
        if(!isFile(time.path()/time.constant()/"gasWallProperties")) return;
        IOdictionary properties(IOobject("gasWallProperties",time.constant(),mesh,
            IOobject::MUST_READ,IOobject::NO_WRITE,false));
        if(properties.lookupOrDefault<label>("schemaVersion",1)!=1)
            FatalErrorInFunction<<"gasWallProperties requires schemaVersion 1"<<exit(FatalError);
        const dictionary& patches=properties.subDict("patches");
        forAllConstIter(dictionary,patches,iter)
        {
            const word name(iter().keyword());
            const label patchId=mesh.boundaryMesh().findPatchID(name);
            if(patchId<0)
                FatalErrorInFunction<<"gasWallProperties names unknown patch "<<name<<exit(FatalError);
            const polyPatch& patch=mesh.boundaryMesh()[patchId];
            if(!isA<wallPolyPatch>(patch))
                FatalErrorInFunction<<"gasWallProperties patch "<<name<<" is not a physical wall"<<exit(FatalError);
            const dictionary& values=patches.subDict(name);
            const scalarList flux(values.lookup("speciesFlux"));
            if(flux.size()!=label(speciesNames_.size()))
                FatalErrorInFunction<<"speciesFlux on patch "<<name<<" requires "<<speciesNames_.size()
                    <<" values in gasModelProperties species order"<<exit(FatalError);
            double mass=0;
            forAll(flux,species)
            {
                if(!std::isfinite(flux[species]))
                    FatalErrorInFunction<<"speciesFlux must be finite on patch "<<name<<exit(FatalError);
                mass+=flux[species];
            }
            if(!std::isfinite(mass) || mass<0)
                FatalErrorInFunction<<"boundaryLayer does not support net wall suction on patch "<<name<<exit(FatalError);
            for(std::size_t slot=0;slot<faces_.size();++slot)
                if(faces_[slot]>=patch.start() && faces_[slot]<patch.start()+patch.size())
                    forAll(flux,species)speciesFlux_[std::size_t(species)*faces_.size()+slot]=flux[species];
        }
    }
public:
    BoundaryLayerInput(const dictionary& turbulence,const ugkwp::GasModelConfiguration& gasModel,
        label turbulenceModel,bool pureGas,scalar turbulentPrandtl)
    {
        const dictionary controls=turbulenceModel==3
            ? turbulence.subDict("RAS").subOrEmptyDict("kOmegaSSTCoeffs") : turbulence;
        active_=controls.lookupOrDefault<word>("wallTreatment","lowRe")=="boundaryLayer";
        if(!active_) return;
        const dictionary& d=controls.subDict("boundaryLayer");
        const std::set<std::string> allowed={"model","nodes","maxIterations","workspaceSlots","budgetAudit",
            "relativeTolerance","rtol","absoluteTolerance","atol","stretch",
            "turbulentPrandtl","Prt","turbulentSchmidt","Sct",
            "geometryTolerance","maxRayCells","quadratureRefinement"};
        forAllConstIter(dictionary,d,iter)
            if(!allowed.count(iter().keyword().c_str()))
                FatalErrorInFunction<<"Unknown boundaryLayer control "<<iter().keyword()<<exit(FatalError);
        const word model=d.lookupOrDefault<word>("model","finiteRate");
        if(model=="constantTransport")config_.model=0;
        else if(model=="finiteRate" || model=="reactingSst")config_.model=1;
        else FatalErrorInFunction<<"Unknown boundaryLayer model '"<<model
            <<"'; valid models are constantTransport and finiteRate (reactingSst alias)"<<exit(FatalError);
        ugkwp::GasCapabilityRequest request;
        request.mode=gasModel.mode;request.turbulenceModel=turbulenceModel;
        request.particleCoupling=!pureGas;request.sstWallTreatment=2;request.boundaryLayerModel=config_.model;
        const auto capability=ugkwp::validateGasCapabilities(request);
        if(!capability)FatalErrorInFunction<<capability.message<<exit(FatalError);
        speciesNames_=gasModel.speciesNames;
        config_.speciesCount=checkedCount(speciesNames_.size(),"species count");
        const label nodes=d.lookupOrDefault<label>("nodes",24);
        const label iterations=d.lookupOrDefault<label>("maxIterations",60);
        const label slots=d.lookupOrDefault<label>("workspaceSlots",0);
        if(nodes<4 || nodes>128 || iterations<1 || slots<0 || slots>4096)
            FatalErrorInFunction<<"boundaryLayer nodes must be 4..128, maxIterations positive, "
                <<"and workspaceSlots 0(auto)..4096"<<exit(FatalError);
        config_.nodes=nodes;config_.maxIterations=iterations;config_.workspaceSlots=slots;
        config_.budgetAudit=d.lookupOrDefault<Switch>("budgetAudit",false)?1:0;
        if(config_.budgetAudit && turbulenceModel!=3)
            FatalErrorInFunction<<"boundaryLayer budgetAudit requires SST"<<exit(FatalError);
        config_.relativeTolerance=coefficient(d,"relativeTolerance","rtol",1e-8);
        config_.absoluteTolerance=coefficient(d,"absoluteTolerance","atol",1e-10);
        config_.stretch=d.lookupOrDefault<scalar>("stretch",2);
        config_.turbulentPrandtl=coefficient(d,"turbulentPrandtl","Prt",turbulentPrandtl);
        config_.turbulentSchmidt=coefficient(d,"turbulentSchmidt","Sct",gasModel.turbulentSchmidt);
        if(!std::isfinite(config_.relativeTolerance) || config_.relativeTolerance<=0
            || !std::isfinite(config_.absoluteTolerance) || config_.absoluteTolerance<0
            || !std::isfinite(config_.stretch) || config_.stretch<1
            || !std::isfinite(config_.turbulentPrandtl) || config_.turbulentPrandtl<=0
            || !std::isfinite(config_.turbulentSchmidt) || config_.turbulentSchmidt<=0)
            FatalErrorInFunction<<"Invalid boundaryLayer tolerance, stretch, Prt or Sct controls"<<exit(FatalError);
        geometryOptions_.relativeTolerance=d.lookupOrDefault<scalar>("geometryTolerance",1e-9);
        geometryOptions_.maxRayCells=d.lookupOrDefault<label>("maxRayCells",32);
        geometryOptions_.quadratureRefinement=d.lookupOrDefault<label>("quadratureRefinement",1);
        // Owner SST source quadrature follows the actual auxiliary profile grid.
        // Frozen/finite-rate laminar walls have no SST volume source to sample.
        if(turbulenceModel==3)
        {
            geometryOptions_.profileNodes=config_.nodes;
            geometryOptions_.profileStretch=config_.stretch;
        }
    }
    bool active() const {return active_;}
    label model() const {return config_.model;}

    void prepare(const Time& time,const fvMesh& mesh,const volScalarField& temperature)
    {
        if(!active_)return;
        if(prepared_)FatalErrorInFunction<<"boundaryLayer geometry must be prepared once per resident"<<exit(FatalError);
        if(mesh.moving() || &temperature.mesh()!=&mesh)
            FatalErrorInFunction<<"gasUGKP boundaryLayer requires its fixed gas mesh"<<exit(FatalError);
        ugkwp::gaswall::WallGeometryMesh<double> geometryMesh;
        geometryMesh.geometryVersion=1;
        for(const point& p:mesh.points())geometryMesh.points.push_back({{p.x(),p.y(),p.z()}});
        for(const vector& c:mesh.cellCentres())geometryMesh.cellCentres.push_back({{c.x(),c.y(),c.z()}});
        for(scalar v:mesh.cellVolumes())geometryMesh.volumes.push_back(v);
        geometryMesh.faceOffsets.push_back(0);
        forAll(mesh.faces(),face)
        {
            for(label p:mesh.faces()[face])geometryMesh.facePoints.push_back(p);
            checkedCount(geometryMesh.facePoints.size(),"face point count");
            geometryMesh.faceOffsets.push_back(geometryMesh.facePoints.size());
            geometryMesh.owner.push_back(mesh.faceOwner()[face]);
            geometryMesh.neighbour.push_back(face<mesh.nInternalFaces()?mesh.faceNeighbour()[face]:-1);
        }
        geometryMesh.physicalWall.assign(mesh.nFaces(),0);
        geometryMesh.gasCell.assign(mesh.nCells(),1);
        forAll(mesh.boundaryMesh(),patchId)
        {
            const polyPatch& patch=mesh.boundaryMesh()[patchId];
            if(!isA<wallPolyPatch>(patch))continue;
            const fvPatchScalarField& wallTemperature=temperature.boundaryField()[patchId];
            if(wallTemperature.type()!="fixedValue")
                FatalErrorInFunction<<"boundaryLayer requires fixedValue wall temperature on physical wall "
                    <<patch.name()<<exit(FatalError);
            forAll(patch,localFace)
            {
                if(!std::isfinite(wallTemperature[localFace]) || wallTemperature[localFace]<=0)
                    FatalErrorInFunction<<"boundaryLayer requires positive finite wall temperature"<<exit(FatalError);
                const label face=patch.start()+localFace;
                geometryMesh.physicalWall[face]=1;faces_.push_back(face);
            }
        }
        if(faces_.empty())FatalErrorInFunction<<"boundaryLayer requires at least one physical wall"<<exit(FatalError);
        ugkwp::gaswall::WallStatus status;
        if(!geometryCache_.rebuild(geometryMesh,faces_,status,geometryOptions_))
            FatalErrorInFunction<<"Cannot certify boundaryLayer owner geometry and unconstrained matching donor; "
                <<"face="<<status.node<<" status="<<label(status.code)<<exit(FatalError);
        quadratureOffsets_.push_back(0);matchingOffsets_.push_back(0);
        for(const auto& descriptor:geometryCache_.descriptors)
        {
            for(int d=0;d<3;++d)geometry_.push_back(descriptor.normal[d]);
            geometry_.push_back(descriptor.ownerDistance);geometry_.push_back(descriptor.matchingDistance);
            geometry_.push_back(descriptor.volume);geometry_.push_back(descriptor.firstMoment);
            quadratureDistance_.insert(quadratureDistance_.end(),descriptor.distance.begin(),descriptor.distance.end());
            quadratureWeight_.insert(quadratureWeight_.end(),descriptor.volumeWeight.begin(),descriptor.volumeWeight.end());
            matchingCells_.insert(matchingCells_.end(),descriptor.matchingCells.begin(),descriptor.matchingCells.end());
            matchingWeight_.insert(matchingWeight_.end(),descriptor.matchingWeights.begin(),descriptor.matchingWeights.end());
            quadratureOffsets_.push_back(checkedCount(quadratureDistance_.size(),"quadrature count"));
            matchingOffsets_.push_back(checkedCount(matchingCells_.size(),"matching count"));
        }
        config_.wallCount=checkedCount(faces_.size(),"wall count");
        config_.quadratureCount=checkedCount(quadratureDistance_.size(),"quadrature count");
        config_.matchingCount=checkedCount(matchingCells_.size(),"matching count");
        readSpeciesFlux(time,mesh);
        prepared_=true;
    }
    void configureResident(void* handle) const
    {
        if(!active_)return;
        if(!prepared_)FatalErrorInFunction<<"boundaryLayer geometry is not prepared"<<exit(FatalError);
        check(ugkwpGpuResidentStrictConfigureBoundaryLayerV1(handle,&config_,faces_.data(),
            quadratureOffsets_.data(),matchingOffsets_.data(),matchingCells_.data(),geometry_.data(),
            quadratureDistance_.data(),quadratureWeight_.data(),matchingWeight_.data(),speciesFlux_.data()),
            "configure resident boundaryLayer");
    }
    void writeDiagnostics(const Time& time,void* handle) const
    {
        if(!active_)return;
        if(!prepared_ || !time.writeTime())
            FatalErrorInFunction<<"boundaryLayer diagnostics require prepared geometry and writeTime()"<<exit(FatalError);
        const std::size_t columns=ugkwpGpuIpc::boundaryLayerDiagnosticScalars+2*config_.speciesCount;
        std::vector<double> diagnostics(faces_.size()*columns);
        check(ugkwpGpuResidentStrictDownloadBoundaryLayerV1(handle,config_.wallCount,config_.speciesCount,
            diagnostics.data()),"download resident boundaryLayer diagnostics");
        const fileName directory=time.timePath()/"uniform";
        mkDir(directory);
        const fileName path=directory/"gasBoundaryLayer.csv";
        std::ofstream output(path.c_str());
        output<<std::setprecision(std::numeric_limits<double>::max_digits10);
        output<<"face,owner,matchingDistance,ownerDistance,heatFluxIntoGas,tractionX,tractionY,tractionZ,"
            <<"wallKFlux,integratedKSource,ownerOmega,residual,iterations,traceDensity,viscosity,stageTime,"
            <<"budgetAuditAvailable,budgetIntervalStart,budgetIntervalDuration,budgetTransportK,budgetTransportOmega,budgetSourceK,budgetSourceOmega,"
            <<"budgetConstraintK,budgetConstraintOmega";
        for(const auto& species:speciesNames_)output<<",speciesFlux_"<<species;
        for(const auto& species:speciesNames_)output<<",reactionIntegral_"<<species;
        output<<'\n';
        // Stage time is supplied by the accepted resident diagnostic record,
        // not substituted with the write endpoint or used for a coarse mismatch.
        for(std::size_t row=0;row<faces_.size();++row)
        {
            for(std::size_t col=0;col<columns;++col)
            {
                if(col)output<<',';
                const bool unavailableBudget=col>=17&&col<25&&diagnostics[row*columns+16]==0;
                if(!unavailableBudget)output<<diagnostics[row*columns+col];
            }
            output<<'\n';
        }
        output.close();
        if(!output)FatalErrorInFunction<<"Cannot write boundaryLayer diagnostics "<<path<<exit(FatalError);
    }
};
}
#endif
