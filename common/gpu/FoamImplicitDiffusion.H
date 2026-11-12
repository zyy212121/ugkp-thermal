#ifndef UGKP_GPU_FOAM_IMPLICIT_DIFFUSION_H
#define UGKP_GPU_FOAM_IMPLICIT_DIFFUSION_H
#include "ImplicitDiffusion.H"
#include "FoamSparseSolve.H"
#include "FoamThermalFieldOperators.H"
#include "gaussLaplacianScheme.H"
#include "fvMesh.H"
#include "fvSchemes.H"
#include "fvmLaplacian.H"
#include "surfaceInterpolate.H"
#include "fixedGradientFvPatchFields.H"
#include <type_traits>
namespace ugkp { namespace gpu {
// A scoped request follows the native runtime-selected Gauss/interpolation/
// snGrad path. It exists only while constructing one equation, never in a
// matrix-address cache; a nested or failed construction restores its parent.
struct FoamTransientAssembly {
    DiffusionTransientInput input;
    bool consumed=false;
    ImplicitDiffusion* resident=nullptr;
    const double* deviceConductivity=nullptr;
    const double* deviceBoundaryConductivity=nullptr;
    void* conductivityStream=nullptr;
    const Foam::volScalarField* interpolatedConductivity=nullptr;
    const Foam::surfaceScalarField* conductivityWeights=nullptr;
    const Foam::surfaceScalarField* conductivityBoundaryValues=nullptr;
    bool harmonicConductivity=false;
    static FoamTransientAssembly*& active(){static thread_local FoamTransientAssembly* request=nullptr;return request;}
    struct Scope {
        FoamTransientAssembly* previous;
        explicit Scope(FoamTransientAssembly& request):previous(active()){active()=&request;}
        ~Scope(){active()=previous;}
    };
};
// Retain OpenFOAM's selected interpolation, snGrad and boundary policies. The
// cell/face matrix assembly and correction divergence execute on CUDA.
class FoamImplicitDiffusion : public Foam::fv::gaussLaplacianScheme<Foam::scalar,Foam::scalar> {
    using Base=Foam::fv::gaussLaplacianScheme<Foam::scalar,Foam::scalar>;
public:
    FoamImplicitDiffusion(const Foam::fvMesh& mesh,Foam::Istream& input):Base(mesh,input){}
    Foam::tmp<Foam::fvMatrix<Foam::scalar>> fvmLaplacian(const Foam::volScalarField& gamma,const Foam::volScalarField& temperature) override {
        auto* request=FoamTransientAssembly::active();
        // Dispatch by the actual selected interpolation implementation. Other
        // registered schemes continue to execute their own unchanged policy.
        if(request&&request->resident&&request->deviceConductivity
            &&dynamic_cast<const FoamThermalLinear*>(&this->tinterpGammaScheme_())){
            const auto weights=this->tinterpGammaScheme_().weights(gamma);
            auto placeholder=Foam::surfaceScalarField::New("interpolate("+gamma.name()+')',gamma.mesh(),gamma.dimensions());
            request->interpolatedConductivity=&gamma;request->conductivityWeights=&weights();
            auto result=fvmLaplacian(placeholder(),temperature);
            request->interpolatedConductivity=nullptr;request->conductivityWeights=nullptr;
            return result;
        }
        return Foam::fv::laplacianScheme<Foam::scalar,Foam::scalar>::fvmLaplacian(gamma,temperature);
    }
    Foam::tmp<Foam::fvMatrix<Foam::scalar>> fvmLaplacian(const Foam::surfaceScalarField& gamma,const Foam::volScalarField& temperature) override {
        using namespace Foam;
        const auto& mesh=this->mesh();
        const auto delta=this->tsnGradScheme_().deltaCoeffs(temperature);
        const bool corrected=this->tsnGradScheme_().corrected();
        auto* transient=FoamTransientAssembly::active();
        bool deviceCorrection=false;
        if(transient&&transient->resident&&corrected
            &&dynamic_cast<const FoamThermalCorrectedGradient*>(&this->tsnGradScheme_())){
            const auto selected=fv::gradScheme<scalar>::New(mesh,ugkp::solverControls::thermalDiscretization(mesh).selection("gradSchemes","grad("+temperature.name()+')'));
            const auto* gauss=dynamic_cast<const FoamThermalGaussGradient*>(&selected());
            deviceCorrection=gauss&&gauss->linearInterpolation();
            forAll(temperature.boundaryField(),p){const auto& field=temperature.boundaryField()[p];
                if(field.coupled())deviceCorrection=deviceCorrection&&(field.type()=="cyclic"||field.type()=="fixedJump"||field.type()=="uniformJump");
                else{const word& kind=field.type();deviceCorrection=deviceCorrection&&(kind=="fixedValue"||kind=="fixedGradient"
                    ||kind=="zeroGradient"||kind=="mixed"||kind=="empty"||kind=="wedge"||kind=="symmetry"||kind=="symmetryPlane");}}
        }
        bool deviceBoundary=transient&&transient->resident&&((!corrected
            &&(this->tsnGradScheme_().type()=="uncorrected"||this->tsnGradScheme_().type()=="orthogonal"))||deviceCorrection);
        forAll(temperature.boundaryField(),p){const auto& field=temperature.boundaryField()[p];
            if(field.coupled())deviceBoundary=deviceBoundary&&(field.type()=="cyclic"||field.type()=="fixedJump"||field.type()=="uniformJump");
            else{const word& kind=field.type();deviceBoundary=deviceBoundary&&(kind=="fixedValue"||kind=="fixedGradient"
                ||kind=="zeroGradient"||kind=="mixed"||kind=="empty"||kind=="wedge"||kind=="symmetry"||kind=="symmetryPlane");}}
        tmp<surfaceScalarField> correction;
        if(corrected&&!deviceCorrection)correction=this->tsnGradScheme_().correction(temperature);
        const bool deviceInterpolation=transient&&transient->interpolatedConductivity;
        DiffusionInput input;input.cells=mesh.nCells();input.internalFaces=mesh.nInternalFaces();input.corrected=corrected;
        input.volume.assign(mesh.V().begin(),mesh.V().end());
        forAll(mesh.neighbour(),f){
            input.owner.push_back(mesh.owner()[f]);input.neighbour.push_back(mesh.neighbour()[f]);
            input.gamma.push_back(deviceInterpolation?0:gamma[f]);input.area.push_back(mesh.magSf()[f]);input.delta.push_back(delta()[f]);
            input.correction.push_back(corrected&&!deviceCorrection?correction()[f]:0);
        }
        forAll(temperature.boundaryField(),p){
            const auto& field=temperature.boundaryField()[p];const auto& cells=mesh.boundary()[p].faceCells();
            const scalarField internal(field.coupled()?field.gradientInternalCoeffs(delta().boundaryField()[p]):field.gradientInternalCoeffs());
            const scalarField boundary(field.coupled()?field.gradientBoundaryCoeffs(delta().boundaryField()[p]):field.gradientBoundaryCoeffs());
            forAll(field,f){
                input.owner.push_back(cells[f]);input.gamma.push_back(deviceInterpolation?0:gamma.boundaryField()[p][f]);input.area.push_back(mesh.magSf().boundaryField()[p][f]);
                input.delta.push_back(delta().boundaryField()[p][f]);input.correction.push_back(corrected&&!deviceCorrection?correction().boundaryField()[p][f]:0);
                input.gradientInternal.push_back(internal[f]);input.gradientBoundary.push_back(boundary[f]);
            }
        }
        if(deviceBoundary){
            input.deviceTemperature=transient->resident->deviceSolution();
            const auto& weights=mesh.weights();const auto& directions=mesh.nonOrthCorrectionVectors();
            forAll(mesh.neighbour(),f){input.gradientWeights.push_back(weights[f]);input.faceArea.push_back(thermalVector(mesh.Sf()[f]));
                input.correctionDirection.push_back(thermalVector(directions[f]));}
            forAll(temperature.boundaryField(),p){const auto& field=temperature.boundaryField()[p];
                scalarField vi,vb,ni,nb,jump;const labelUList* adjacent=nullptr;
                vector tx(1,0,0),ty(0,1,0),tz(0,0,1);
                if(field.coupled()){
                    const auto& cyclic=refCast<const cyclicFvPatchScalarField>(field);
                    adjacent=&cyclic.cyclicPatch().neighbFvPatch().faceCells();
                    tx=cyclic.transform().transform(tx);ty=cyclic.transform().transform(ty);tz=cyclic.transform().transform(tz);
                    if(isA<jumpCyclicFvPatchScalarField>(field)){jump=refCast<const jumpCyclicFvPatchScalarField>(field).jump();
                        if(!cyclic.cyclicPatch().owner())jump*=-1;}
                }else{vi=field.valueInternalCoeffs(weights.boundaryField()[p]);vb=field.valueBoundaryCoeffs(weights.boundaryField()[p]);
                    ni=field.gradientInternalCoeffs();nb=field.gradientBoundaryCoeffs();}
                forAll(field,f){input.gradientWeights.push_back(weights.boundaryField()[p][f]);
                    input.faceArea.push_back(thermalVector(mesh.Sf().boundaryField()[p][f]));
                    input.correctionDirection.push_back(thermalVector(directions.boundaryField()[p][f]));
                    input.gradientNeighbour.push_back(adjacent?(*adjacent)[f]:-1);input.jump.push_back(jump.size()?jump[f]:0);
                    input.valueInternal.push_back(adjacent?0:vi[f]);input.valueBoundary.push_back(adjacent?0:vb[f]);
                    input.normalInternal.push_back(adjacent?0:ni[f]);input.normalBoundary.push_back(adjacent?0:nb[f]);
                    input.boundaryValueDelta.push_back(mesh.boundary()[p].deltaCoeffs()[f]);
                    const auto& geometry=mesh.boundary()[p];
                    input.neighbourTransform.push_back(thermalVector(adjacent?tx:thermalGradientBoundaryValue(geometry,f,tx)));
                    input.neighbourTransform.push_back(thermalVector(adjacent?ty:thermalGradientBoundaryValue(geometry,f,ty)));
                    input.neighbourTransform.push_back(thermalVector(adjacent?tz:thermalGradientBoundaryValue(geometry,f,tz)));}
            }
        }
        static thread_local ImplicitDiffusion backend;DiffusionMatrix coefficients;
        if(transient&&transient->interpolatedConductivity){
            const auto& field=*transient->interpolatedConductivity;const auto& weights=*transient->conductivityWeights;
            input.deviceConductivity=transient->deviceConductivity;input.deviceBoundaryConductivity=transient->deviceBoundaryConductivity;input.conductivityStream=transient->conductivityStream;
            input.harmonicConductivity=transient->harmonicConductivity;
            input.conductivityWeights.assign(weights.primitiveField().begin(),weights.primitiveField().end());
            forAll(field.boundaryField(),p){const auto& patch=field.boundaryField()[p];scalarField adjacent,jump;const labelUList* adjacentCells=nullptr;
                if(patch.coupled()){
                    if(isA<cyclicFvPatchScalarField>(patch)){const auto& cyclic=refCast<const cyclicFvPatchScalarField>(patch);
                        adjacentCells=&cyclic.cyclicPatch().neighbFvPatch().faceCells();
                        if(isA<jumpCyclicFvPatchScalarField>(patch)){jump=refCast<const jumpCyclicFvPatchScalarField>(patch).jump();
                            if(!cyclic.cyclicPatch().owner())jump*=-1;}}
                    else adjacent=patch.patchNeighbourField();
                }
                forAll(patch,f){input.conductivityWeights.push_back(weights.boundaryField()[p][f]);
                    input.conductivityBoundary.push_back(transient->conductivityBoundaryValues?transient->conductivityBoundaryValues->boundaryField()[p][f]:patch[f]);
                    input.conductivityNeighbour.push_back(adjacent.size()?adjacent[f]:0);
                    input.conductivityCoupled.push_back(patch.coupled()?1:patch.type()=="zeroGradient"?2:0);
                    input.conductivityAdjacentCells.push_back(adjacentCells?(*adjacentCells)[f]:-1);input.conductivityJump.push_back(jump.size()?jump[f]:0);}}
        }
        if(transient&&transient->resident){
            transient->resident->assembleResident(input,&transient->input);transient->consumed=true;
            tmp<fvScalarMatrix> result(new fvScalarMatrix(temperature,
                delta().dimensions()*gamma.dimensions()*mesh.magSf().dimensions()*temperature.dimensions()));
            auto& matrix=result.ref();matrix.diag()=0;if(mesh.nInternalFaces())matrix.upper()=0;
            // Native remote interfaces consume boundary coefficients on the
            // host; local cyclic/jump coefficients stay entirely on device.
            bool remote=false;
            forAll(temperature.boundaryField(),p)
                remote=remote||(temperature.boundaryField()[p].coupled()
                    &&!isA<cyclicFvPatchScalarField>(temperature.boundaryField()[p]));
            if(remote){
                std::vector<double> diagonal,boundary;
                transient->resident->downloadBoundaryCoefficients(diagonal,boundary);
                int b=0;forAll(temperature.boundaryField(),p)forAll(temperature.boundaryField()[p],f){
                    matrix.internalCoeffs()[p][f]=diagonal[b];matrix.boundaryCoeffs()[p][f]=boundary[b++];}
            }
            return result;
        }
        backend.assemble(input,coefficients,transient?&transient->input:nullptr);
        if(transient)transient->consumed=true;
        tmp<fvScalarMatrix> result(new fvScalarMatrix(temperature,delta().dimensions()*gamma.dimensions()*mesh.magSf().dimensions()*temperature.dimensions()));
        auto& matrix=result.ref();
        forAll(matrix.diag(),c){matrix.diag()[c]=coefficients.diagonal[c];matrix.source()[c]=coefficients.source[c];}
        auto& upper=matrix.upper();forAll(upper,f)upper[f]=coefficients.upper[f];
        int b=0;
        forAll(temperature.boundaryField(),p)forAll(temperature.boundaryField()[p],f){matrix.internalCoeffs()[p][f]=coefficients.boundaryDiagonal[b];matrix.boundaryCoeffs()[p][f]=coefficients.boundarySource[b++];}
        if(corrected&&mesh.schemes().fluxRequired(temperature.name())){
            matrix.faceFluxCorrectionPtr()=new surfaceScalarField(surfaceScalarField::New("gpuDiffusionFluxCorrection",mesh,gamma.dimensions()*mesh.magSf().dimensions()*correction().dimensions()));
            auto& flux=*matrix.faceFluxCorrectionPtr();forAll(flux,f)flux[f]=coefficients.fluxCorrection[f];
            int f=input.internalFaces;forAll(flux.boundaryField(),p)forAll(flux.boundaryField()[p],j)flux.boundaryFieldRef()[p][j]=coefficients.fluxCorrection[f++];
        }
        return result;
    }
};
inline void installGpuImplicitDiffusion(){
    installGpuThermalFields();
    // Replace the common Gauss implementation in these two applications;
    // dictionary selection and existing interpolation/snGrad choices survive.
    // There is no retry or CPU solver dispatch on a CUDA failure.
    using Scheme=Foam::fv::laplacianScheme<Foam::scalar,Foam::scalar>;
    static const bool installed=[](){
        Scheme::IstreamConstructorTablePtr_->set("Gauss",+[](const Foam::fvMesh& mesh,Foam::Istream& input)->Foam::tmp<Scheme>{return Foam::tmp<Scheme>(new FoamImplicitDiffusion(mesh,input));});
        return true;
    }();
    (void)installed;
}
inline void formGpuImplicitEquation(Foam::fvScalarMatrix&,const Foam::volScalarField&,
    const Foam::volScalarField&,double,bool);
// Both volume and face conductivity overloads preserve the native selected
// interpolation and nonorthogonal correction. Only the unnecessary intermediate
// matrix download/re-upload is removed; residual/flux consumers retain fvMatrix.
template<class Conductivity> inline Foam::tmp<Foam::fvScalarMatrix> gpuImplicitThermalEquation(
    const Conductivity& conductivity,const Foam::volScalarField& temperature,
    const Foam::volScalarField& capacity,const Foam::volScalarField& rhs,
    double dt,bool sourceFromCapacity=false){
    std::vector<double> c(capacity.primitiveField().begin(),capacity.primitiveField().end());
    std::vector<double> b(rhs.primitiveField().begin(),rhs.primitiveField().end());
    FoamTransientAssembly request;request.input={&c,&b,dt,sourceFromCapacity};
    FoamTransientAssembly::Scope scope(request);
    auto selected=Foam::fv::laplacianScheme<Foam::scalar,Foam::scalar>::New(temperature.mesh(),
        ugkp::solverControls::thermalDiscretization(temperature.mesh()).selection("laplacianSchemes","laplacian("+conductivity.name()+','+temperature.name()+')'));
    auto result=selected.ref().fvmLaplacian(conductivity,temperature);
    // A separately registered non-Gauss scheme owns its native assembly.
    // Preserve that explicit dictionary selection and its existing completion.
    if(!request.consumed)formGpuImplicitEquation(result.ref(),capacity,rhs,dt,sourceFromCapacity);
    return result;
}
inline void formGpuImplicitEquation(Foam::fvScalarMatrix& matrix,const Foam::volScalarField& capacity,
    const Foam::volScalarField& rhs,double dt,bool sourceFromCapacity=false){
    using namespace Foam;
    DiffusionMatrix factors;
    factors.diagonal.assign(matrix.diag().begin(),matrix.diag().end());factors.source.assign(matrix.source().begin(),matrix.source().end());
    if(matrix.lduAddr().upperAddr().size())factors.upper.assign(matrix.upper().begin(),matrix.upper().end());
    if(matrix.asymmetric())factors.lower.assign(matrix.lower().begin(),matrix.lower().end());
    forAll(matrix.internalCoeffs(),p)forAll(matrix.internalCoeffs()[p],f){
        factors.boundaryDiagonal.push_back(matrix.internalCoeffs()[p][f]);factors.boundarySource.push_back(matrix.boundaryCoeffs()[p][f]);}
    auto* correction=matrix.faceFluxCorrectionPtr();
    if(correction){
        factors.fluxCorrection.assign(correction->primitiveField().begin(),correction->primitiveField().end());
        forAll(correction->boundaryField(),p)forAll(correction->boundaryField()[p],f)factors.fluxCorrection.push_back(correction->boundaryField()[p][f]);
    }
    const auto& nativeVolume=matrix.psi().mesh().V();
    std::vector<double> volume(nativeVolume.begin(),nativeVolume.end());
    std::vector<double> c(capacity.primitiveField().begin(),capacity.primitiveField().end()),b(rhs.primitiveField().begin(),rhs.primitiveField().end());
    static thread_local ImplicitDiffusion backend;backend.transient(volume,c,b,dt,sourceFromCapacity,factors);
    forAll(matrix.diag(),i){matrix.diag()[i]=factors.diagonal[i];matrix.source()[i]=factors.source[i];}
    if(factors.upper.size())forAll(matrix.upper(),f)matrix.upper()[f]=factors.upper[f];
    if(factors.lower.size())forAll(matrix.lower(),f)matrix.lower()[f]=factors.lower[f];
    int face=0;forAll(matrix.internalCoeffs(),p)forAll(matrix.internalCoeffs()[p],f){matrix.internalCoeffs()[p][f]=factors.boundaryDiagonal[face];matrix.boundaryCoeffs()[p][f]=factors.boundarySource[face++];}
    if(correction){int f=0;forAll(*correction,i)(*correction)[i]=factors.fluxCorrection[f++];
        forAll(correction->boundaryField(),p)forAll(correction->boundaryField()[p],i)correction->boundaryFieldRef()[p][i]=factors.fluxCorrection[f++];}
}

// A native policy adapter around one resident equation owner. Native boundary
// evaluation and explicitly selected schemes are retained. No assembled cell or
// face coefficient is downloaded just to feed the CUDA sparse solver again.
class FoamResidentThermalEquation {
    Foam::tmp<Foam::fvScalarMatrix> equation_;
    ImplicitDiffusion& resident_;
    SparseConjugateGradient& sparse_;
public:
    FoamResidentThermalEquation(const Foam::tmp<Foam::fvScalarMatrix>& equation,
        ImplicitDiffusion& resident,SparseConjugateGradient& sparse,bool retainSolution=false)
        :equation_(equation),resident_(resident),sparse_(sparse){
        if(!retainSolution)resident_.uploadSolution(equation_().psi().primitiveField().begin(),equation_().psi().size());
    }
    Foam::solverPerformance solve(const Foam::dictionary& controls,bool publish=true){
        resident_.snapshotSolution();
        const auto values=resident_.deviceValues();
        const auto result=solveFoamScalarMatrix(equation_.ref(),controls,sparse_,&values,resident_.deviceSolution());
        if(publish)publishSolution();return result;
    }
    bool configureResidentLocalSolve(const Foam::dictionary& dictionary){
        if(!resident_.residentIteration())return false;
        ConjugateGradientControls controls;
        controls.maximumIterations=dictionary.lookupOrDefault<Foam::label>("maxIter",1000);
        controls.minimumIterations=dictionary.lookupOrDefault<Foam::label>("minIter",0);
        controls.tolerance=dictionary.lookupOrDefault<Foam::scalar>("tolerance",1e-6);
        controls.relativeTolerance=dictionary.lookupOrDefault<Foam::scalar>("relTol",0);
        controls.normalizationSmall=Foam::solverPerformance::small_;
        controls.singularSmall=Foam::solverPerformance::vsmall_;
        return withFoamScalarMatrix(equation_.ref(),[&](const SparseLduView& matrix,const SparseCommunication& communication){
            if(communication.addInterface||communication.globalSum)return false;
            sparse_.configureResidentLocalLdu(matrix,resident_.deviceValues(),controls);return true;
        });
    }
    void publishSolution(){
        resident_.refreshBoundaryTemperature();
        auto& temperature=const_cast<Foam::volScalarField&>(equation_().psi());
        resident_.downloadSolution(temperature.primitiveFieldRef().begin(),temperature.size());
        std::vector<double> boundary,normal;
        resident_.downloadBoundaryState(boundary,normal);
        int offset=0;forAll(temperature.boundaryField(),p){
            if(!normal.empty()&&temperature.boundaryField()[p].type()=="fixedGradient"){
                auto& gradient=Foam::refCast<Foam::fixedGradientFvPatchScalarField>(temperature.boundaryFieldRef()[p]).gradient();
                forAll(gradient,f)gradient[f]=normal[offset+f];}
            offset+=temperature.boundaryField()[p].size();}
        temperature.correctBoundaryConditions();
    }
    const double* residentResidual(){
        const auto values=resident_.deviceValues();
        withFoamScalarMatrix(equation_.ref(),[&](const SparseLduView& matrix,const SparseCommunication& communication){
            sparse_.residualResidentLdu(matrix,values,resident_.deviceSolution(),resident_.deviceResidual(),communication);
            return 0;
        });
        return resident_.deviceResidual();
    }
    Foam::tmp<Foam::scalarField> residual(){
        residentResidual();
        Foam::tmp<Foam::scalarField> result(new Foam::scalarField(equation_().psi().size()));
        resident_.downloadResidual(result.ref().begin(),result().size());return result;
    }
    std::vector<double> fluxValues(bool boundaryOnly,bool download=true){
        using namespace Foam;const auto& temperature=equation_().psi();
        std::vector<double> neighbour,jumps;std::vector<int> coupled,adjacentCells;
        forAll(temperature.boundaryField(),p){const auto& patch=temperature.boundaryField()[p];
            scalarField adjacent,jump;const labelUList* cells=nullptr;
            if(patch.coupled()){
                if(isA<cyclicFvPatchScalarField>(patch)){const auto& cyclic=refCast<const cyclicFvPatchScalarField>(patch);
                    cells=&cyclic.cyclicPatch().neighbFvPatch().faceCells();
                    if(isA<jumpCyclicFvPatchScalarField>(patch)){jump=refCast<const jumpCyclicFvPatchScalarField>(patch).jump();
                        if(!cyclic.cyclicPatch().owner())jump*=-1;}
                }else adjacent=patch.patchNeighbourField();
            }
            forAll(patch,f){neighbour.push_back(adjacent.size()?adjacent[f]:0);coupled.push_back(patch.coupled()?1:0);
                adjacentCells.push_back(cells?(*cells)[f]:-1);jumps.push_back(jump.size()?jump[f]:0);}}
        std::vector<double> values;resident_.flux(neighbour,coupled,adjacentCells,jumps,values,boundaryOnly,download);return values;
    }
    Foam::tmp<Foam::surfaceScalarField> flux(){
        using namespace Foam;const auto& temperature=equation_().psi();const auto values=fluxValues(false);
        auto result=surfaceScalarField::New("flux("+temperature.name()+')',temperature.mesh(),equation_().dimensions());
        int f=0;forAll(result.ref(),i)result.ref()[i]=values[f++];
        forAll(result.ref().boundaryField(),p)forAll(result.ref().boundaryField()[p],i)
            result.ref().boundaryFieldRef()[p][i]=values[f++];
        return result;
    }
    const double* residentBoundaryFlux(){fluxValues(true,false);return resident_.deviceBoundaryFlux();}
    Foam::List<Foam::scalarField> boundaryFlux(){
        const auto& boundary=equation_().psi().boundaryField();const auto values=fluxValues(true);
        Foam::List<Foam::scalarField> result(boundary.size());int face=0;
        forAll(boundary,p){result[p].setSize(boundary[p].size());forAll(result[p],f)result[p][f]=values[face++];}
        return result;
    }

};
inline Foam::solverPerformance solveFoamScalarMatrix(FoamResidentThermalEquation& equation,
    const Foam::dictionary& controls,SparseConjugateGradient&){return equation.solve(controls);}

template<class Conductivity> inline FoamResidentThermalEquation residentImplicitThermalEquation(
    ImplicitDiffusion& resident,SparseConjugateGradient& sparse,
    const Conductivity& conductivity,const Foam::volScalarField& temperature,
    const Foam::volScalarField& capacity,const Foam::volScalarField& rhs,
    double dt,bool sourceFromCapacity=false,const double* deviceCapacity=nullptr,
    const double* deviceRhs=nullptr,void* readyStream=nullptr,const double* deviceConductivity=nullptr,bool retainSolution=false,
    const Foam::volScalarField* harmonicSource=nullptr,const Foam::surfaceScalarField* harmonicBoundary=nullptr,
    const double* deviceBoundaryConductivity=nullptr){
    std::vector<double> c,b;
    if(!deviceCapacity)c.assign(capacity.primitiveField().begin(),capacity.primitiveField().end());
    if(!deviceRhs)b.assign(rhs.primitiveField().begin(),rhs.primitiveField().end());
    FoamTransientAssembly request;request.input={&c,&b,dt,sourceFromCapacity,deviceCapacity,deviceRhs,readyStream};request.resident=&resident;request.deviceConductivity=deviceConductivity;request.conductivityStream=readyStream;request.deviceBoundaryConductivity=deviceBoundaryConductivity;
    if(harmonicSource&&deviceConductivity){request.interpolatedConductivity=harmonicSource;
        request.conductivityWeights=&temperature.mesh().weights();request.harmonicConductivity=true;
        request.conductivityBoundaryValues=harmonicBoundary;}
    FoamTransientAssembly::Scope scope(request);
    if(!retainSolution)resident.uploadSolution(temperature.primitiveField().begin(),temperature.size());
    auto selected=Foam::fv::laplacianScheme<Foam::scalar,Foam::scalar>::New(temperature.mesh(),
        ugkp::solverControls::thermalDiscretization(temperature.mesh()).selection("laplacianSchemes","laplacian("+conductivity.name()+','+temperature.name()+')'));
    Foam::tmp<Foam::fvScalarMatrix> equation;
    if constexpr(std::is_same<Conductivity,Foam::surfaceScalarField>::value){
        if(harmonicSource&&!dynamic_cast<FoamImplicitDiffusion*>(&selected.ref())){
            Foam::surfaceScalarField nativeConductivity(conductivity);
            interpolateThermalField(*harmonicSource,temperature.mesh().weights(),nativeConductivity,true);
            request.interpolatedConductivity=nullptr;
            equation=selected.ref().fvmLaplacian(nativeConductivity,temperature);
        }else equation=selected.ref().fvmLaplacian(conductivity,temperature);
    }else equation=selected.ref().fvmLaplacian(conductivity,temperature);
    if(!request.consumed){
        // This is the previously supported native runtime-selected non-Gauss
        // discretization, not a numerical retry. Its own factors are retained.
        formGpuImplicitEquation(equation.ref(),capacity,rhs,dt,sourceFromCapacity);
        const auto& mesh=temperature.mesh();DiffusionInput input;DiffusionMatrix factors;
        input.cells=mesh.nCells();input.internalFaces=mesh.nInternalFaces();
        input.owner.assign(mesh.owner().begin(),mesh.owner().end());input.neighbour.assign(mesh.neighbour().begin(),mesh.neighbour().end());
        factors.diagonal.assign(equation().diag().begin(),equation().diag().end());factors.source.assign(equation().source().begin(),equation().source().end());
        const Foam::lduMatrix& ldu=equation();
        if(mesh.nInternalFaces())factors.upper.assign(ldu.upper().begin(),ldu.upper().end());
        if(equation().asymmetric())factors.lower.assign(ldu.lower().begin(),ldu.lower().end());
        forAll(equation.ref().internalCoeffs(),p)forAll(equation.ref().internalCoeffs()[p],f){
            factors.boundaryDiagonal.push_back(equation.ref().internalCoeffs()[p][f]);factors.boundarySource.push_back(equation.ref().boundaryCoeffs()[p][f]);}
        if(const auto* correction=equation.ref().faceFluxCorrectionPtr()){
            factors.fluxCorrection.assign(correction->primitiveField().begin(),correction->primitiveField().end());
            forAll(correction->boundaryField(),p)forAll(correction->boundaryField()[p],f)factors.fluxCorrection.push_back(correction->boundaryField()[p][f]);}
        resident.retain(input,factors);
    }
    return FoamResidentThermalEquation(equation,resident,sparse,true);
}

// Harmonic material conduction already overrides all internal interpolated
// values. Keep only the selected native boundary policy before that operator;
// the standard linear policy needs no intermediate full face-field transfer.
inline Foam::tmp<Foam::surfaceScalarField> thermalHarmonicBoundaryValues(const Foam::volScalarField& field){
    using namespace Foam;
    const auto selected=surfaceInterpolationScheme<scalar>::New(field.mesh(),ugkp::solverControls::thermalDiscretization(field.mesh()).selection("interpolationSchemes","interpolate("+field.name()+')'));
    if(!dynamic_cast<const FoamThermalLinear*>(&selected()))return selected().interpolate(field);
    auto result=surfaceScalarField::New("interpolate("+field.name()+')',field.mesh(),field.dimensions());
    result.ref().primitiveFieldRef()=0;
    forAll(result.ref().boundaryField(),p)result.ref().boundaryFieldRef()[p]=field.boundaryField()[p];
    return result;
}

}}
#endif
