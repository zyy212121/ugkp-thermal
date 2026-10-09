#ifndef CHMT_GPU_SHAREDGASDEVICESTORAGE_CUH
#define CHMT_GPU_SHAREDGASDEVICESTORAGE_CUH
#include "gpu/Buffer.H"
#include "core/HostState.H"
#include "materials/Thermodynamics.H"
#include "../../../common/gasTransport/GasStateView.H"
#include "../../../common/gasTransport/GasGeometryPolicy.H"
#include "../../../common/gasTransport/GasModelIO.H"
#include "../../../common/gasTransport/GasMechanismIO.H"
#include "../../../common/gasNumerics/GpuSstAlgebra.cuh"
#include <cstring>
#include <climits>
namespace chmt {
struct SharedGasDeviceView:ugkwp::GasStateView<Real,Real,ugkwp::SstCoefficients> {
    ugkwp::GasSpeciesState<Real,Ns> gasSpecies;
};
static_assert(std::is_trivially_copyable<SharedGasDeviceView>::value,"shared gas view is POD");
template<class T>struct GasAllocation {bool allocate(T&,std::size_t,CudaFault&){return true;}};
template<class T>struct GasAllocation<T*> {
    Buffer<T> values;
    bool allocate(T*& pointer,std::size_t n,CudaFault& fault){
        if(!values.resize(n,fault)||!values.zero(nullptr))return false;
        pointer=values.data();return true;
    }
};
// Allocation and marshaling only. No numerical gas, chemistry or RK formula.
class SharedGasDeviceStorage {
    using Time=Real;using SstCoefficients=ugkwp::SstCoefficients;
    CudaFault fault_;
    SharedGasDeviceView view_{};
    Buffer<SharedGasDeviceView> device_;
#define UGKWP_GAS_STATE_FIELD(Type,Name) GasAllocation<Type> owner_##Name;
#include "../../../common/gasTransport/GasStateFields.inc"
#undef UGKWP_GAS_STATE_FIELD
    Buffer<Real> speciesRho_,speciesInitial_,speciesFlux_,speciesGX_,speciesGY_,speciesGZ_,speciesLimiter_,speciesBoundary_,speciesScale_,diffusivity_,sound_,cp_,r_,thermoCoeff_,elements_,basis_;
    Buffer<int> compositionFixed_,cellStatus_,faceStatus_;
    Buffer<ugkwp::SpeciesThermoData<Real>> thermoData_;
    Buffer<ugkwp::GasReactionData<Real>> reactions_;
    Buffer<ugkwp::GasStoichTerm<Real>> reactants_,products_;
    Buffer<ugkwp::GasColliderEfficiency<Real>> efficiencies_;
    Buffer<ugkwp::ChemistryAudit<Real,Ns>> chemistryAudit_;
    Buffer<ugkwp::ChemistryStatus> chemistryStatus_;
    Buffer<Real> geometryOld_,geometryNew_,geometrySweeps_;
    std::array<Buffer<Real>,13> sstAudit_;
    std::uint64_t identity_=0,thermoIdentity_=0,mechanismIdentity_=0;
    std::vector<Real> volume_;
    Tolerances geometryTolerances_;
    int activeSpecies_=-1;
    static std::size_t count(const char* name,std::size_t cells,std::size_t faces,std::size_t addressing){
        if(!std::strcmp(name,"cellFaceId"))return addressing;
        if(!std::strncmp(name,"pressureSchedule",16))return 0;
        if(!std::strncmp(name,"face",4)||!std::strncmp(name,"gasBoundary",11)
            ||!std::strncmp(name,"riemannBoundary",15)||!std::strncmp(name,"gasPhi",6)
            ||!std::strncmp(name,"sstPhi",6)||!std::strncmp(name,"sstBoundary",11)
            ||!std::strcmp(name,"Sfx")||!std::strcmp(name,"Sfy")||!std::strcmp(name,"Sfz")
            ||!std::strcmp(name,"magSf")||!std::strcmp(name,"deltaCoeffs")||!std::strcmp(name,"scheduledInletFaceMask"))return faces;
        return cells;
    }
    template<class T>bool upload(T* destination,const std::vector<T>& source){return source.empty()||fault_.check(cudaMemcpy(destination,source.data(),source.size()*sizeof(T),cudaMemcpyHostToDevice),"upload shared gas field");}
    template<class T>bool allocateSpecies(Buffer<T>& owner,T*& pointer,std::size_t n){if(!owner.resize(n,fault_)||!owner.zero(nullptr))return false;pointer=owner.data();return true;}
public:
    SharedGasDeviceStorage()=default;
    SharedGasDeviceStorage(const SharedGasDeviceStorage&)=delete;
    SharedGasDeviceStorage& operator=(const SharedGasDeviceStorage&)=delete;
    SharedGasDeviceView& hostView(){return view_;}
    const SharedGasDeviceView& hostView()const{return view_;}
    SharedGasDeviceView* deviceView()const{return device_.data();}
    const std::string& error()const{return fault_.message;}
    bool refreshView(){return device_.uploadOne(view_,fault_);}
    bool bindStageGeometry(const HostStageGeometry& stage){
        if(stage.oldVolume.size()!=std::size_t(view_.nCells)||stage.newVolume.size()!=stage.oldVolume.size()||stage.sweptVolume.size()!=std::size_t(view_.nFaces))return false;
        if(!geometryOld_.upload(stage.oldVolume,fault_)||!geometryNew_.upload(stage.newVolume,fault_)||!geometrySweeps_.upload(stage.sweptVolume,fault_))return false;
        view_.gasGeometry.absoluteGeometryTolerance=geometryTolerances_.absoluteGeometry;view_.gasGeometry.relativeGeometryTolerance=geometryTolerances_.relativeGeometry;
        view_.gasGeometry.enabled=true;view_.gasGeometry.oldVolume=geometryOld_.data();view_.gasGeometry.newVolume=geometryNew_.data();view_.gasGeometry.faceSweptVolume=geometrySweeps_.data();view_.gasGeometry.interval=stage.interval;
        return refreshView();
    }
    bool clearGeometry(){view_.gasGeometry={};return refreshView();}
    template<class T>bool read(const T* pointer,std::size_t n,std::vector<T>& output){
        std::vector<T> candidate(n);if(n&&!fault_.check(cudaMemcpy(candidate.data(),pointer,n*sizeof(T),cudaMemcpyDeviceToHost),"download shared gas field"))return false;
        output=std::move(candidate);return true;
    }
    template<class T>bool write(T* pointer,const std::vector<T>& values){return upload(pointer,values);}
    bool clearTrialStatus(){for(auto& buffer:sstAudit_)if(!buffer.zero(nullptr))return false;return cellStatus_.zero(nullptr)&&faceStatus_.zero(nullptr)&&chemistryStatus_.zero(nullptr)&&chemistryAudit_.zero(nullptr);}
    bool checkStatus(std::string& error){
        if(!fault_.check(cudaDeviceSynchronize(),"synchronize shared gas trial")){error=fault_.message;return false;}
        std::vector<int> cells,faces;
        if(!cellStatus_.download(cells,nullptr)||!faceStatus_.download(faces,nullptr)){error=fault_.message;return false;}
        for(std::size_t c=0;c<cells.size();++c)if(cells[c]){error="shared gas cell "+std::to_string(c)+" status "+std::to_string(cells[c]);return false;}
        for(std::size_t f=0;f<faces.size();++f)if(faces[f]){error="shared gas face "+std::to_string(f)+" status "+std::to_string(faces[f]);return false;}
        std::vector<ugkwp::ChemistryStatus> chemical;
        if(!chemistryStatus_.download(chemical,nullptr)){error=fault_.message;return false;}
        for(std::size_t c=0;c<chemical.size();++c)if(!chemical[c]){error="shared chemistry cell "+std::to_string(c)+" status "+std::to_string(int(chemical[c].code));return false;}
        error.clear();return true;
    }
    bool configure(const ModelConfig& model,const ugkwp::GasModelConfiguration& gas,
        const ugkwp::GasMechanismConfiguration& mechanism,const HostState& state,std::string& error){
        if(identity_!=0){error="gas model configuration is immutable after allocation";return false;}
        geometryTolerances_=model.physics.tolerances;
        const std::size_t nc=state.gas.size(),nf=state.gasMesh.owner.size();
        if(!nc||nc>INT_MAX||nf>INT_MAX||state.gasMesh.volumes.size()!=nc){error="invalid shared gas dimensions";return false;}
        const auto& mesh=state.gasMesh;
        if(mesh.cellCentres.size()!=nc||mesh.cellFaceOffsets.size()!=nc+1||mesh.cellFaceOffsets.front()!=0
            ||mesh.cellFaceOffsets.back()!=int(mesh.cellFaces.size())||mesh.neighbour.size()!=nf||mesh.periodicPartner.size()!=nf
            ||mesh.areaVectors.size()!=nf||mesh.faceCentres.size()!=nf||mesh.boundaryKind.size()!=nf||mesh.boundaryPrimitive.size()!=nf){error="incomplete common gas mesh addressing";return false;}
        for(std::size_t c=0;c<nc;++c)if(!finite(mesh.volumes[c])||mesh.volumes[c]<=0||!finite(mesh.cellCentres[c])||mesh.cellFaceOffsets[c]>mesh.cellFaceOffsets[c+1]){error="invalid common gas cell geometry";return false;}
        for(std::size_t f=0;f<nf;++f)if(mesh.owner[f]<0||std::size_t(mesh.owner[f])>=nc||mesh.neighbour[f]<-1
            ||(mesh.neighbour[f]>=0&&std::size_t(mesh.neighbour[f])>=nc)||!finite(mesh.areaVectors[f])||mag(mesh.areaVectors[f])<=0||!finite(mesh.faceCentres[f])){error="invalid common gas face geometry";return false;}
        for(int f:mesh.cellFaces)if(f<0||std::size_t(f)>=nf){error="invalid common gas cell-face addressing";return false;}

        const bool mixture=gas.mode!=ugkwp::GasMode::SingleLegacy;
        activeSpecies_=model.physics.singleGasSpecies;
        if(!mixture&&(activeSpecies_<0||activeSpecies_>=Ns)){error="single gas active material species missing";return false;}
        if(gas.speciesNames.size()!=Ns||gas.speciesOrderHash!=model.physics.speciesFingerprint||gas.thermoHash!=model.physics.gasThermoFingerprint){error="shared gas model identity mismatch";return false;}
        view_.nCells=int(nc);view_.nFaces=int(nf);view_.nInternalFaces=0;
        while(view_.nInternalFaces<int(nf)&&state.gasMesh.neighbour[view_.nInternalFaces]>=0)++view_.nInternalFaces;
        for(std::size_t f=view_.nInternalFaces;f<nf;++f)if(state.gasMesh.neighbour[f]>=0){error="common gas mesh requires contiguous internal faces";return false;}
#define UGKWP_GAS_STATE_FIELD(Type,Name) if(!owner_##Name.allocate(view_.Name,count(#Name,nc,nf,state.gasMesh.cellFaces.size()),fault_)){error=fault_.message;return false;}
#include "../../../common/gasTransport/GasStateFields.inc"
#undef UGKWP_GAS_STATE_FIELD
        auto& sp=view_.gasSpecies;sp.mode=gas.mode;sp.thermoControls=gas.chemistryControls.thermo;sp.chemistryControls=gas.chemistryControls;sp.turbulentSchmidt=gas.turbulentSchmidt;
#define ALLOC(Owner,Field,N) if(!allocateSpecies(Owner,sp.Field,(N))){error=fault_.message;return false;}
        if(mixture){
        ALLOC(speciesRho_,rho,Ns*nc);ALLOC(speciesInitial_,initial,Ns*nc);ALLOC(speciesFlux_,flux,Ns*nf);
        ALLOC(speciesGX_,gradX,Ns*nc);ALLOC(speciesGY_,gradY,Ns*nc);ALLOC(speciesGZ_,gradZ,Ns*nc);
        ALLOC(speciesLimiter_,limiter,nc);ALLOC(speciesBoundary_,boundaryMassFraction,Ns*nf);ALLOC(speciesScale_,positivityScale,Ns*nc);
        ALLOC(compositionFixed_,compositionBoundaryFixed,nf);ALLOC(sound_,soundSpeed,nc);ALLOC(cp_,heatCapacity,nc);ALLOC(r_,gasConstant,nc);
        }
        ALLOC(cellStatus_,cellStatus,nc);ALLOC(faceStatus_,faceStatus,nf);
        if(gas.mode==ugkwp::GasMode::MixtureChemistry){ALLOC(chemistryAudit_,chemistryAudit,nc);ALLOC(chemistryStatus_,chemistryStatus,nc);}
#undef ALLOC
        if(mixture){
        if(!thermoData_.upload(gas.species,fault_)||!thermoCoeff_.upload(gas.coefficients,fault_)||!elements_.upload(gas.elementComposition,fault_)){error=fault_.message;return false;}
        sp.thermo=gas.thermoView<Ns>();sp.thermo.species=thermoData_.data();sp.thermo.coefficients=thermoCoeff_.data();sp.thermo.elementComposition=elements_.data();
        if(gas.diffusionModel==ugkwp::GasDiffusionModel::Constant){if(!diffusivity_.upload(gas.diffusionCoefficients,fault_)){error=fault_.message;return false;}sp.diffusivity=diffusivity_.data();}
        if(gas.mode==ugkwp::GasMode::MixtureChemistry){
            if(!reactions_.upload(mechanism.reactions,fault_)||!reactants_.upload(mechanism.reactants,fault_)||!products_.upload(mechanism.products,fault_)
                ||!efficiencies_.upload(mechanism.efficiencies,fault_)||!basis_.upload(mechanism.stoichiometricBasis,fault_)){error=fault_.message;return false;}
            sp.mechanism=mechanism.mechanismView<Ns>();sp.mechanism.reactions=reactions_.data();sp.mechanism.reactants=reactants_.data();sp.mechanism.products=products_.data();sp.mechanism.efficiencies=efficiencies_.data();sp.mechanism.stoichiometricBasis=basis_.data();
        }
        }else{
            const auto& t=model.physics.species[activeSpecies_];
            view_.Rgas=t.R;view_.gasCp=t.cp0;view_.gammaGas=t.cp0/(t.cp0-t.R);
        }
        identity_=gas.speciesOrderHash;thermoIdentity_=gas.thermoHash;mechanismIdentity_=mechanism.mechanismHash;
        view_.rhoMin=1e-30;view_.TgasMin=1e-12;view_.gasMu=model.physics.gasViscosity;
        view_.gasPrClamped=1;
        view_.gasThermalConductivity=model.physics.gasConductivity;
        view_.maxDiffusionNumber=.25;view_.turbulentPrandtl=model.physics.sst.turbulentPrandtl;
        view_.sstCoefficients=ugkwp::defaultSstCoefficients();
        const auto& st=model.physics.sst;auto& coeff=view_.sstCoefficients;
        coeff.a1=st.a1;coeff.betaStar=st.betaStar;coeff.beta1=st.beta1;coeff.beta2=st.beta2;
        coeff.gamma1=st.gamma1;coeff.gamma2=st.gamma2;coeff.alphaK1=st.sigmaK1;coeff.alphaK2=st.sigmaK2;
        coeff.alphaOmega1=st.sigmaOmega1;coeff.alphaOmega2=st.sigmaOmega2;coeff.c1=st.productionLimit;
        view_.sstKMin=maxValue(st.minimumK,1e-12);view_.sstOmegaMin=maxValue(st.minimumOmega,1e-12);view_.sstMaxSourceNumber=.25;
        view_.turbulenceModel=model.physics.enableSst?3:0;view_.sstConfigured=model.physics.enableSst;view_.sstWallTreatment=0;
        if(model.physics.enableSst){
            auto& audit=view_.gasSstAudit;audit.enabled=true;
            Real** pointers[]={&audit.transportK,&audit.transportOmega,&audit.sourceK,&audit.sourceOmega,&audit.constraintK,&audit.constraintOmega,
                &audit.initialTransportK,&audit.initialTransportOmega,&audit.initialSourceK,&audit.initialSourceOmega,&audit.initialConstraintK,&audit.initialConstraintOmega,&audit.volume};
            for(std::size_t i=0;i<sstAudit_.size();++i)if(!allocateSpecies(sstAudit_[i],*pointers[i],nc)){error=fault_.message;return false;}
        }
        view_.sstWallKappa=.41;view_.sstWallE=9.8;view_.sstWallCmu=.09;view_.lesDeltaCoeff=1;view_.waleCw=.325;view_.smagorinskyCs=.17;
        if(!uploadGeometry(state.gasMesh)||!uploadState(state,error)||!refreshView()){if(error.empty())error=fault_.message;return false;}
        error.clear();return true;
    }
    bool uploadGeometry(const HostMesh& mesh){
        const std::size_t nc=mesh.volumes.size(),nf=mesh.owner.size();
        if(nc!=std::size_t(view_.nCells)||nf!=std::size_t(view_.nFaces))return false;
        volume_=mesh.volumes;std::vector<Real> cx(nc),cy(nc),cz(nc),length(nc),distance(nc,1);
        std::vector<int> start(nc),counts(nc);
        for(std::size_t c=0;c<nc;++c){cx[c]=mesh.cellCentres[c].x;cy[c]=mesh.cellCentres[c].y;cz[c]=mesh.cellCentres[c].z;length[c]=std::cbrt(mesh.volumes[c]);start[c]=mesh.cellFaceOffsets[c];counts[c]=mesh.cellFaceOffsets[c+1]-start[c];if(mesh.wallDistance.size()==nc)distance[c]=mesh.wallDistance[c];}
        std::vector<int> neighbours=mesh.neighbour;
        for(std::size_t f=0;f<nf;++f)if(mesh.boundaryKind[f]==BoundaryKind::Periodic){
            const int pair=mesh.periodicPartner[f];if(pair<0||std::size_t(pair)>=nf)return false;
            neighbours[f]=mesh.owner[pair];
        }
        if(!upload(view_.Cx,cx)||!upload(view_.Cy,cy)||!upload(view_.Cz,cz)||!upload(view_.V,mesh.volumes)||!upload(view_.cellLength,length)||!upload(view_.sstWallDistance,distance)
            ||!upload(view_.cellPlaneStart,start)||!upload(view_.cellPlaneCount,counts)||!upload(view_.cellFaceId,mesh.cellFaces)||!upload(view_.faceOwner,mesh.owner)||!upload(view_.faceNeighbour,neighbours)||!upload(view_.facePeriodicPair,mesh.periodicPartner))return false;
        std::vector<Real> sx(nf),sy(nf),sz(nf),area(nf),fx(nf),fy(nf),fz(nf),dx(nf),dy(nf),dz(nf),delta(nf),weight(nf,.5),rho(nf),pressure(nf),temp(nf),ux(nf),uy(nf),uz(nf),Y(Ns*nf);
        std::vector<int> kind(nf),fixU(nf),fixT(nf),fixR(nf),fixP(nf),fixedY(nf),kMode(nf),omegaMode(nf);
        std::vector<Real> kBoundary(nf),omegaBoundary(nf);
        for(std::size_t f=0;f<nf;++f){auto a=mesh.areaVectors[f];sx[f]=a.x;sy[f]=a.y;sz[f]=a.z;area[f]=mag(a);fx[f]=mesh.faceCentres[f].x;fy[f]=mesh.faceCentres[f].y;fz[f]=mesh.faceCentres[f].z;
            const int o=mesh.owner[f],n=mesh.neighbour[f];Vec3 chord=mesh.faceCentres[f]-mesh.cellCentres[o];
            if(n>=0)chord=mesh.cellCentres[n]-mesh.cellCentres[o];
            const auto b=mesh.boundaryKind[f];kind[f]=b==BoundaryKind::Internal?0:b==BoundaryKind::NoSlip||b==BoundaryKind::Interface?2:b==BoundaryKind::Slip?1:b==BoundaryKind::Periodic?5:b==BoundaryKind::Empty?4:0;
            if(b==BoundaryKind::Periodic){const int pair=mesh.periodicPartner[f];const Vec3 shift=mesh.faceCentres[f]-mesh.faceCentres[pair];dx[f]=shift.x;dy[f]=shift.y;dz[f]=shift.z;chord=mesh.cellCentres[mesh.owner[pair]]+shift-mesh.cellCentres[o];}
            delta[f]=1/maxValue(absValue(dot(chord,a/area[f])),.05*mag(chord));
            if(n>=0||b==BoundaryKind::Periodic){
                const Real own=absValue(dot(mesh.faceCentres[f]-mesh.cellCentres[o],a/area[f]));
                const Vec3 mappedNeighbour=mesh.cellCentres[o]+chord;
                const Real other=absValue(dot(mappedNeighbour-mesh.faceCentres[f],a/area[f]));
                weight[f]=(own+other)>0?other/(own+other):.5;
            }
            const auto& w=mesh.boundaryPrimitive[f];rho[f]=w.rho;pressure[f]=w.pressure;temp[f]=w.temperature;ux[f]=w.velocity.x;uy[f]=w.velocity.y;uz[f]=w.velocity.z;
            const bool inlet=b==BoundaryKind::Inlet;fixU[f]=inlet||b==BoundaryKind::NoSlip;fixT[f]=fixedTemperature(mesh.thermalBoundary.empty()?nullptr:mesh.thermalBoundary.data(),int(f));fixR[f]=inlet;fixP[f]=inlet||b==BoundaryKind::Outlet;fixedY[f]=inlet;
            for(int s=0;s<Ns;++s)Y[s*nf+f]=w.Y[s];
            kMode[f]=omegaMode[f]=inlet?1:0;
            if(mesh.boundarySst.size()==nf){kBoundary[f]=mesh.boundarySst[f].k;omegaBoundary[f]=mesh.boundarySst[f].omega;}
        }
#define PUT(Name,Source) if(!upload(view_.Name,Source))return false
        PUT(Sfx,sx);PUT(Sfy,sy);PUT(Sfz,sz);PUT(magSf,area);PUT(faceCx,fx);PUT(faceCy,fy);PUT(faceCz,fz);PUT(facePeriodicDx,dx);PUT(facePeriodicDy,dy);PUT(facePeriodicDz,dz);PUT(deltaCoeffs,delta);PUT(faceWeight,weight);
        PUT(gasBoundaryKind,kind);PUT(riemannBoundaryKind,kind);
        PUT(sstBoundaryK,kBoundary);PUT(sstBoundaryOmega,omegaBoundary);PUT(sstBoundaryKMode,kMode);PUT(sstBoundaryOmegaMode,omegaMode);
#define BOUNDARY(Name,Source) PUT(gasBoundary##Name,Source);PUT(riemannBoundary##Name,Source)
        BOUNDARY(Rho,rho);BOUNDARY(P,pressure);BOUNDARY(T,temp);BOUNDARY(Ux,ux);BOUNDARY(Uy,uy);BOUNDARY(Uz,uz);BOUNDARY(UFix,fixU);BOUNDARY(TFix,fixT);BOUNDARY(RhoFix,fixR);BOUNDARY(PFix,fixP);
#undef BOUNDARY
#undef PUT
        return view_.gasSpecies.mode==ugkwp::GasMode::SingleLegacy
            ||(upload(view_.gasSpecies.boundaryMassFraction,Y)&&upload(view_.gasSpecies.compositionBoundaryFixed,fixedY));
    }
    bool uploadState(const HostState& state,std::string& error){
        if(state.gas.size()!=std::size_t(view_.nCells)||state.gasMesh.volumes!=volume_){error="shared gas upload geometry/state mismatch";return false;}
        const std::size_t n=state.gas.size();std::vector<Real> rho(n),px(n),py(n),pz(n),energy(n),species(Ns*n),k(n),omega(n);
        for(std::size_t c=0;c<n;++c){const auto& q=state.gas[c];const Real v=volume_[c];if(!finite(v)||v<=0||!finite(q.mass)||q.mass<=0||!finite(q.energy)||!finite(q.momentum)){error="invalid integrated gas upload";return false;}
            if(view_.gasSpecies.mode==ugkwp::GasMode::SingleLegacy)
                for(int s=0;s<Ns;++s)if(q.species[s]!=(s==activeSpecies_?q.mass:0)){error="legacy gas upload contains another species";return false;}
            Real sum=0;for(int s=0;s<Ns;++s){if(!finite(q.species[s])||q.species[s]<0){error="invalid species upload";return false;}species[s*n+c]=q.species[s]/v;sum+=q.species[s];}
            if(!closeEnough(sum,q.mass,0,view_.gasSpecies.densityClosureTolerance)){error="species upload does not close";return false;}
            rho[c]=q.mass/v;px[c]=q.momentum.x/v;py[c]=q.momentum.y/v;pz[c]=q.momentum.z/v;energy[c]=q.energy/v;
            if(!finite(rho[c])||!finite(px[c])||!finite(py[c])||!finite(pz[c])||!finite(energy[c])){error="density conversion overflow";return false;}
            for(int s=0;s<Ns;++s)if(!finite(species[s*n+c])){error="species density conversion overflow";return false;}
            if(view_.sstConfigured){
                if(state.sst.size()!=n||!finite(state.sst[c].rhoK)||state.sst[c].rhoK<0
                    ||!finite(state.sst[c].rhoOmega)||state.sst[c].rhoOmega<=0){error="invalid SST integral upload";return false;}
                k[c]=state.sst[c].rhoK/v;omega[c]=state.sst[c].rhoOmega/v;
            }
        }
        if(!upload(view_.rho,rho)||!upload(view_.rhoUx,px)||!upload(view_.rhoUy,py)||!upload(view_.rhoUz,pz)||!upload(view_.rhoE,energy)||(view_.gasSpecies.mode!=ugkwp::GasMode::SingleLegacy&&!upload(view_.gasSpecies.rho,species))||!upload(view_.rhoK,k)||!upload(view_.rhoOmega,omega)||!clearTrialStatus()){error=fault_.message;return false;}
        error.clear();return true;
    }
    bool downloadState(HostState& state,std::string& error){
        std::vector<Real> rho,px,py,pz,energy,species,k,omega;const std::size_t n=view_.nCells;
        if(!read(view_.rho,n,rho)||!read(view_.rhoUx,n,px)||!read(view_.rhoUy,n,py)||!read(view_.rhoUz,n,pz)||!read(view_.rhoE,n,energy)||(view_.gasSpecies.mode!=ugkwp::GasMode::SingleLegacy&&!read(view_.gasSpecies.rho,Ns*n,species))){error=fault_.message;return false;}
        auto candidate=state.gas;candidate.resize(n);
        for(std::size_t c=0;c<n;++c){auto& q=candidate[c];q.mass=rho[c]*volume_[c];q.momentum={px[c]*volume_[c],py[c]*volume_[c],pz[c]*volume_[c]};q.energy=energy[c]*volume_[c];
            for(int s=0;s<Ns;++s)q.species[s]=view_.gasSpecies.mode==ugkwp::GasMode::SingleLegacy?(s==activeSpecies_?q.mass:0):species[s*n+c]*volume_[c];
            if(!finite(q.mass)||q.mass<=0||!finite(q.energy)||!finite(q.momentum)){error="invalid shared gas downloaded candidate";return false;}}
        auto sst=state.sst;
        if(view_.sstConfigured){
            if(!read(view_.rhoK,n,k)||!read(view_.rhoOmega,n,omega)){error=fault_.message;return false;}
            sst.resize(n);for(std::size_t c=0;c<n;++c){sst[c]={k[c]*volume_[c],omega[c]*volume_[c]};
                if(!finite(sst[c].rhoK)||sst[c].rhoK<0||!finite(sst[c].rhoOmega)||sst[c].rhoOmega<=0){error="invalid SST integral download";return false;}}
        }
        state.gas=std::move(candidate);state.sst=std::move(sst);error.clear();return true;
    }
};
}
#endif
