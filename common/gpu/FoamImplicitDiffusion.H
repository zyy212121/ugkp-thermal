#ifndef UGKP_GPU_FOAM_IMPLICIT_DIFFUSION_H
#define UGKP_GPU_FOAM_IMPLICIT_DIFFUSION_H
#include "ImplicitDiffusion.H"
#include "FoamThermalFieldOperators.H"
#include "gaussLaplacianScheme.H"
#include "fvMesh.H"
#include "fvSchemes.H"
#include "fvmLaplacian.H"
namespace ugkp { namespace gpu {
// A scoped request follows the native runtime-selected Gauss/interpolation/
// snGrad path. It exists only while constructing one equation, never in a
// matrix-address cache; a nested or failed construction restores its parent.
struct FoamTransientAssembly {
    DiffusionTransientInput input;
    bool consumed=false;
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
    Foam::tmp<Foam::fvMatrix<Foam::scalar>> fvmLaplacian(const Foam::surfaceScalarField& gamma,const Foam::volScalarField& temperature) override {
        using namespace Foam;
        const auto& mesh=this->mesh();
        const auto delta=this->tsnGradScheme_().deltaCoeffs(temperature);
        const bool corrected=this->tsnGradScheme_().corrected();
        tmp<surfaceScalarField> correction;
        if(corrected)correction=this->tsnGradScheme_().correction(temperature);
        DiffusionInput input;input.cells=mesh.nCells();input.internalFaces=mesh.nInternalFaces();input.corrected=corrected;
        input.volume.assign(mesh.V().begin(),mesh.V().end());
        forAll(mesh.neighbour(),f){
            input.owner.push_back(mesh.owner()[f]);input.neighbour.push_back(mesh.neighbour()[f]);
            input.gamma.push_back(gamma[f]);input.area.push_back(mesh.magSf()[f]);input.delta.push_back(delta()[f]);
            input.correction.push_back(corrected?correction()[f]:0);
        }
        forAll(temperature.boundaryField(),p){
            const auto& field=temperature.boundaryField()[p];const auto& cells=mesh.boundary()[p].faceCells();
            const scalarField internal(field.coupled()?field.gradientInternalCoeffs(delta().boundaryField()[p]):field.gradientInternalCoeffs());
            const scalarField boundary(field.coupled()?field.gradientBoundaryCoeffs(delta().boundaryField()[p]):field.gradientBoundaryCoeffs());
            forAll(field,f){
                input.owner.push_back(cells[f]);input.gamma.push_back(gamma.boundaryField()[p][f]);input.area.push_back(mesh.magSf().boundaryField()[p][f]);
                input.delta.push_back(delta().boundaryField()[p][f]);input.correction.push_back(corrected?correction().boundaryField()[p][f]:0);
                input.gradientInternal.push_back(internal[f]);input.gradientBoundary.push_back(boundary[f]);
            }
        }
        static thread_local ImplicitDiffusion backend;DiffusionMatrix coefficients;auto* transient=FoamTransientAssembly::active();
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
    auto result=Foam::fvm::laplacian(conductivity,temperature);
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

}}
#endif
