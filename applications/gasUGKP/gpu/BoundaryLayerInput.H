#ifndef UGKWP_BOUNDARY_LAYER_INPUT_H
#define UGKWP_BOUNDARY_LAYER_INPUT_H

#include "fvCFD.H"
#include "wallPolyPatch.H"
#include "GpuBackendApi.H"
#include "gasTransport/GasModelIO.H"
#include "gasWall/WallGeometryBuilder.H"
#include "DiagnosticsFoam.H"
#include "DiagnosticsCatalogue.H"
#include "gasTransport/GasBoundaryLayerProfileOutput.H"
#include <fstream>
#include <iomanip>
#include <limits>
#include <vector>

namespace Foam
{
// Host-only dictionary, immutable mesh and diagnostic adapter. All wall-profile
// evaluation and physical gas inventories remain in the resident backend.
class BoundaryLayerInput
{
    bool active_=false,prepared_=false,hasLocalSst_=false;
    ugkwpGpu::BoundaryLayerConfigV1 config_;
    ugkp::diagnostics::DeviceContext geometryDiagnostics_{};
    ugkwp::gaswall::WallGeometryOptions<double> geometryOptions_;
    ugkwp::gaswall::WallGeometryCache<double> geometryCache_;
    std::vector<std::string> speciesNames_;
    std::vector<int> faces_,quadratureOffsets_,matchingOffsets_,matchingCells_;
    std::vector<double> geometry_,quadratureDistance_,quadratureWeight_,matchingWeight_,speciesFlux_;

    std::vector<double> profileFaceCentres_;
    using ProfileWriter=void (BoundaryLayerInput::*)(const Time&,void*) const;
    ProfileWriter profileWriter_=&BoundaryLayerInput::skipProfiles;
    void skipProfiles(const Time&,void*) const {}
    void writeProfiles(const Time& time,void* handle) const
    {
        std::vector<double> profiles(faces_.size()*config_.nodes*(7+config_.speciesCount));
        double stageTime=0;
        const int result=ugkwpGpuResidentStrictDownloadBoundaryLayerProfilesV1(handle,
            config_.wallCount,config_.nodes,config_.speciesCount,profiles.data(),&stageTime);
        if(result)
        {
            WarningInFunction<<"Cannot download boundaryLayer node profiles: "
                <<ugkwpGpuResidentStrictLastError()<<endl;
            return; // Optional output cannot reject an accepted physical step.
        }
        const fileName path=time.timePath()/"uniform"/"gasBoundaryLayerProfiles.csv";
        std::ofstream output(path.c_str());
        const bool flowWritten=ugkwp::writeGasBoundaryLayerProfiles(output,faces_,profileFaceCentres_,geometry_,
            speciesNames_,config_.nodes,profiles.data(),stageTime,time.value());
        output.close();
        if(!flowWritten || !output)WarningInFunction<<"Cannot write finite boundaryLayer node profiles "<<path<<endl;
        // Equilibrium ODE has no local k/omega: do not create that interface/file.
        if(hasLocalSst_)
        {
            const fileName sstPath=time.timePath()/"uniform"/"gasBoundaryLayerSstProfiles.csv";
            std::ofstream sst(sstPath.c_str());
            const bool written=ugkwp::writeGasBoundaryLayerSstProfiles(sst,true,faces_,
                config_.nodes,speciesNames_.size(),profiles.data(),stageTime,time.value());
            sst.close();
            if(!written || !sst)WarningInFunction<<"Cannot write finite off-wall SST profiles "<<sstPath<<endl;
        }
    }

    static scalar coefficient(const dictionary& d,const word& name,const word& alias,scalar fallback)
    {
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
        const dictionary& patches=properties.subDict("patches");
        forAllConstIter(dictionary,patches,iter)
        {
            const word name(iter().keyword());
            const label patchId=mesh.boundaryMesh().findPatchID(name);
            if(patchId<0)
                FatalErrorInFunction<<"gasWallProperties names unknown patch "<<name<<exit(FatalError);
            const polyPatch& patch=mesh.boundaryMesh()[patchId];
            const dictionary& values=patches.subDict(name);


            const auto onPatch=[&](int face){return face>=patch.start()&&face<patch.start()+patch.size();};
            const auto first=std::find_if(faces_.begin(),faces_.end(),onPatch);
            if(first==faces_.end())continue;
            const scalarList flux(values.lookup("speciesFlux"));
            for(std::size_t species=0;species<speciesNames_.size();++species)
            {
                if(species>=std::size_t(flux.size()))
                    FatalErrorInFunction<<"speciesFlux missing species "<<speciesNames_[species]
                        <<" at index "<<species<<" on patch "<<name<<exit(FatalError);
                for(auto face=first;face!=faces_.end();++face)
                    if(onPatch(*face))speciesFlux_[species*faces_.size()+std::size_t(face-faces_.begin())]=flux[species];
            }
        }
    }
public:
    BoundaryLayerInput(const dictionary& turbulence,const ugkwp::GasModelConfiguration& gasModel,
        label turbulenceModel,bool,scalar turbulentPrandtl)
    {
        const dictionary controls=turbulenceModel==3
            ? turbulence.subDict("RAS").subOrEmptyDict("kOmegaSSTCoeffs") : turbulence;
        active_=controls.lookupOrDefault<word>("wallTreatment","lowRe")=="boundaryLayer";
        if(!active_) return;
        const dictionary& d=controls.subDict("boundaryLayer");
        const word model=d.lookupOrDefault<word>("model","finiteRate");
        if(model=="constantTransport")config_.model=0;
        else if(model=="finiteRate" || model=="reactingSst")config_.model=1;
        else if(model=="equilibriumOde")config_.model=2;
        else FatalErrorInFunction<<"Unknown boundaryLayer model '"<<model
            <<"'; valid models are constantTransport, finiteRate (reactingSst alias), and equilibriumOde"<<exit(FatalError);
        hasLocalSst_=turbulenceModel==3 && config_.model==1;
        // Read once. Subsequent timesteps never inspect the option or dictionary.
        if(d.lookupOrDefault<bool>("writeProfiles",false))profileWriter_=&BoundaryLayerInput::writeProfiles;
        speciesNames_=gasModel.speciesNames;
        config_.speciesCount=checkedCount(speciesNames_.size(),"species count");
        const label nodes=d.lookupOrDefault<label>("nodes",ugkwp::gaswall::defaultWallNodes);
        const label iterations=d.lookupOrDefault<label>("maxIterations",60);
        const label slots=d.lookupOrDefault<label>("workspaceSlots",0);
        config_.nodes=nodes;config_.maxIterations=iterations;config_.workspaceSlots=slots;
        geometryOptions_.reconstructMatching=true;
        geometryOptions_.constructQuadrature=turbulenceModel==3&&config_.model==1;
        geometryOptions_.profileNodes=nodes;
        config_.relativeTolerance=coefficient(d,"relativeTolerance","rtol",1e-8);
        config_.absoluteTolerance=coefficient(d,"absoluteTolerance","atol",1e-10);
        config_.stretch=d.lookupOrDefault<scalar>("stretch",ugkwp::gaswall::defaultWallStretch);
        geometryOptions_.profileStretch=config_.stretch;
        config_.turbulentPrandtl=coefficient(d,"turbulentPrandtl","Prt",turbulentPrandtl);
        config_.turbulentSchmidt=coefficient(d,"turbulentSchmidt","Sct",gasModel.turbulentSchmidt);
        geometryOptions_.relativeTolerance=d.lookupOrDefault<scalar>("geometryTolerance",1e-9);
        geometryOptions_.maxRayCells=d.lookupOrDefault<label>("maxRayCells",32);
        geometryOptions_.quadratureRefinement=d.lookupOrDefault<label>("quadratureRefinement",1);
        // Production consumes physical wall response, never the auxiliary
        // owner-k integral; profile controls do not create geometry points.
    }
    bool active() const {return active_;}
    label model() const {return config_.model;}

    void prepare(const Time& time,const fvMesh& mesh,const volScalarField&)
    {
        if(!active_)return;


        prepared_=false;geometryCache_.clear();
        faces_.clear();geometry_.clear();quadratureOffsets_.clear();matchingOffsets_.clear();
        quadratureDistance_.clear();quadratureWeight_.clear();matchingCells_.clear();matchingWeight_.clear();
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
            forAll(patch,localFace)
            {
                const label face=patch.start()+localFace;
                geometryMesh.physicalWall[face]=1;faces_.push_back(face);
            }
        }
        ugkwp::gaswall::WallStatus status;
        ugkp::diagnostics::reset(geometryDiagnostics_,0,0,time.timeIndex(),time.value());
        geometryOptions_.diagnostics=&geometryDiagnostics_;
        if(!geometryCache_.rebuild(geometryMesh,faces_,status,geometryOptions_))
            FatalErrorInFunction<<"Cannot construct boundaryLayer owner geometry or matching donor; "
                <<"face="<<status.node<<" status="<<label(status.code)<<exit(FatalError);
        ugkp::diagnostics::dispatch(ugkp::diagnostics::takeSnapshot(geometryDiagnostics_,
            ugkp::diagnostics::sourceCatalogue()),ugkp::diagnostics::makeFoamSink());
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
        if(profileWriter_==&BoundaryLayerInput::writeProfiles)
        {
            profileFaceCentres_.clear();profileFaceCentres_.reserve(3*faces_.size());
            for(const int face:faces_)
                for(int d=0;d<3;++d)profileFaceCentres_.push_back(mesh.faceCentres()[face][d]);
        }
        readSpeciesFlux(time,mesh);
        prepared_=true;
    }
    void configureResident(void* handle)
    {
        if(!active_)return;
        if(!prepared_)FatalErrorInFunction<<"boundaryLayer geometry is not prepared"<<exit(FatalError);
        check(ugkwpGpuResidentStrictConfigureBoundaryLayerV1(handle,&config_,faces_.data(),
            quadratureOffsets_.data(),matchingOffsets_.data(),matchingCells_.data(),geometry_.data(),
            quadratureDistance_.data(),quadratureWeight_.data(),matchingWeight_.data(),speciesFlux_.data()),
            "configure resident boundaryLayer");
        if(profileWriter_==&BoundaryLayerInput::writeProfiles)
        {
            const int result=ugkwpGpuResidentStrictEnableBoundaryLayerProfilesV1(handle);
            if(result)
            {
                WarningInFunction<<"Boundary-layer node output unavailable: "
                    <<ugkwpGpuResidentStrictLastError()<<endl;
                profileWriter_=&BoundaryLayerInput::skipProfiles;
            }
        }
    }
    void writeDiagnostics(const Time& time,void* handle) const
    {
        if(!active_)return;
        if(!prepared_)
            FatalErrorInFunction<<"boundaryLayer diagnostics require prepared geometry"<<exit(FatalError);
        const std::size_t columns=ugkwpGpu::boundaryLayerDiagnosticScalars+2*config_.speciesCount;
        std::vector<double> diagnostics(faces_.size()*columns);
        check(ugkwpGpuResidentStrictDownloadBoundaryLayerV1(handle,config_.wallCount,config_.speciesCount,
            diagnostics.data()),"download resident boundaryLayer diagnostics");
        const fileName directory=time.timePath()/"uniform";
        mkDir(directory);
        const fileName path=directory/"gasBoundaryLayer.csv";
        std::ofstream output(path.c_str());
        const bool written=ugkwp::writeGasBoundaryLayerDiagnostics(output,hasLocalSst_,
            faces_.size(),speciesNames_,diagnostics.data());
        output.close();
        if(!written || !output)FatalErrorInFunction<<"Cannot write finite boundaryLayer diagnostics "<<path<<exit(FatalError);
        (this->*profileWriter_)(time,handle);
    }
};
}
#endif
