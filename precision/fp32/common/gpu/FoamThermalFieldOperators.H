#ifndef UGKP_GPU_FOAM_THERMAL_FIELD_OPERATORS_H
#define UGKP_GPU_FOAM_THERMAL_FIELD_OPERATORS_H
#include "ThermalFieldOperators.H"
#include "FoamThermalNativeSchemes.H"
#include "../FoamThermalDiscretization.H"
#include "fvMesh.H"
#include "linear.H"
#include "gaussGrad.H"
#include "correctedSnGrad.H"
#include "extrapolatedCalculatedFvPatchFields.H"
#include "fvSchemes.H"
#include "wedgeFvPatch.H"
#include "symmetryFvPatch.H"
#include "symmetryPlaneFvPatch.H"
namespace ugkp { namespace gpu {
inline ThermalVector thermalVector(const Foam::vector& v){return {GpuReal(v.x()),GpuReal(v.y()),GpuReal(v.z())};}
inline Foam::vector foamVector(const ThermalVector& v){return {v.x,v.y,v.z};}
// The gradient field created by Gauss uses native geometric constraint
// patches. Store their linear vector transformation once with the topology.
inline Foam::vector thermalGradientBoundaryValue(const Foam::fvPatch& patch,Foam::label face,const Foam::vector& value){
    using namespace Foam;
    if(isA<wedgeFvPatch>(patch))return transform(refCast<const wedgeFvPatch>(patch).faceT(),value);
    if(isA<symmetryPlaneFvPatch>(patch)){
        const vector normal=refCast<const symmetryPlaneFvPatch>(patch).n();
        return (value+transform(I-2.0*sqr(normal),value))/2.0;
    }
    if(isA<symmetryFvPatch>(patch)){
        const vector normal=patch.Sf()[face]/patch.magSf()[face];
        return (value+transform(I-2.0*sqr(normal),value))/2.0;
    }
    return value;
}
inline ThermalFieldOperators& thermalFieldBackend(){static thread_local ThermalFieldOperators backend;return backend;}
inline void interpolateThermalField(const Foam::volScalarField& field,const Foam::surfaceScalarField& weights,
    Foam::surfaceScalarField& result,bool harmonic){
    using namespace Foam;const auto& mesh=field.mesh();std::vector<ScalarFaceInterpolation> input;
    forAll(mesh.neighbour(),f)input.push_back({GpuReal(field[mesh.owner()[f]]),GpuReal(field[mesh.neighbour()[f]]),GpuReal(weights[f]),0});
    forAll(field.boundaryField(),p){
        const auto& patch=field.boundaryField()[p];const auto& cells=mesh.boundary()[p].faceCells();
        scalarField adjacent;if(patch.coupled())adjacent=patch.patchNeighbourField();
        forAll(patch,f)input.push_back(patch.coupled()?ScalarFaceInterpolation{GpuReal(field[cells[f]]),GpuReal(adjacent[f]),GpuReal(weights.boundaryField()[p][f]),1}
            :ScalarFaceInterpolation{GpuReal(harmonic?result.boundaryField()[p][f]:patch[f]),0,0,2});
    }
    std::vector<GpuReal> values;thermalFieldBackend().interpolate(input,harmonic,values);
    int f=0;forAll(result,i)result[i]=values[f++];forAll(result.boundaryField(),p)forAll(result.boundaryField()[p],i)result.boundaryFieldRef()[p][i]=values[f++];
}
class FoamThermalLinear : public Foam::linear<Foam::scalar> {
public:
    explicit FoamThermalLinear(const Foam::fvMesh& mesh):Foam::linear<Foam::scalar>(mesh){}
    Foam::tmp<Foam::surfaceScalarField> interpolate(const Foam::volScalarField& field) const override {
        auto result=Foam::surfaceScalarField::New("interpolate("+field.name()+')',field.mesh(),field.dimensions());
        interpolateThermalField(field,this->weights(field)(),result.ref(),false);return result;
    }
};
class FoamThermalGaussGradient : public Foam::fv::gradScheme<Foam::scalar> {
    Foam::tmp<Foam::surfaceInterpolationScheme<Foam::scalar>> interpolation_;
public:
    FoamThermalGaussGradient(const Foam::fvMesh& mesh,Foam::Istream& input):Foam::fv::gradScheme<Foam::scalar>(mesh),
        interpolation_(input.eof()?Foam::tmp<Foam::surfaceInterpolationScheme<Foam::scalar>>(new FoamThermalLinear(mesh)):
            Foam::surfaceInterpolationScheme<Foam::scalar>::New(mesh,input)){}
    const Foam::word& type()const override{return Foam::fv::gaussGrad<Foam::scalar>::typeName;}
    bool linearInterpolation()const{return dynamic_cast<const FoamThermalLinear*>(&interpolation_())!=nullptr;}
    Foam::tmp<Foam::volVectorField> calcGrad(const Foam::volScalarField& field,const Foam::word& name)const override {
        using namespace Foam;const auto& mesh=field.mesh();const auto faces=interpolation_().interpolate(field);
        GaussGradientInput input;input.cells=mesh.nCells();input.internalFaces=mesh.nInternalFaces();input.volume.assign(mesh.V().begin(),mesh.V().end());
        forAll(mesh.neighbour(),f){input.owner.push_back(mesh.owner()[f]);input.neighbour.push_back(mesh.neighbour()[f]);input.faceValue.push_back(faces()[f]);input.area.push_back(thermalVector(mesh.Sf()[f]));}
        forAll(mesh.boundary(),p)forAll(mesh.boundary()[p],f){input.owner.push_back(mesh.boundary()[p].faceCells()[f]);input.faceValue.push_back(faces().boundaryField()[p][f]);input.area.push_back(thermalVector(mesh.Sf().boundaryField()[p][f]));}
        std::vector<ThermalVector> values;thermalFieldBackend().gradient(input,values);
        auto result=volVectorField::New(name,mesh,dimensionedVector("0",field.dimensions()/dimLength,Zero),extrapolatedCalculatedFvPatchVectorField::typeName);
        auto& gradient=result.ref();forAll(gradient,c)gradient[c]=foamVector(values[c]);gradient.correctBoundaryConditions();
        std::vector<GradientBoundaryCorrection> boundary;
        forAll(field.boundaryField(),p)if(!field.boundaryField()[p].coupled()){
            const scalarField normalGradient(field.boundaryField()[p].snGrad());
            forAll(field.boundaryField()[p],f)boundary.push_back({thermalVector(gradient.boundaryField()[p][f]),thermalVector(mesh.Sf().boundaryField()[p][f]),GpuReal(mesh.magSf().boundaryField()[p][f]),GpuReal(normalGradient[f])});
        }
        thermalFieldBackend().correctGradient(boundary,values);int index=0;
        forAll(field.boundaryField(),p)if(!field.boundaryField()[p].coupled())forAll(field.boundaryField()[p],f)gradient.boundaryFieldRef()[p][f]=foamVector(values[index++]);
        return result;
    }
};
class FoamThermalCorrectedGradient : public Foam::fv::correctedSnGrad<Foam::scalar> {
public:
    explicit FoamThermalCorrectedGradient(const Foam::fvMesh& mesh):Foam::fv::correctedSnGrad<Foam::scalar>(mesh){}
    Foam::tmp<Foam::surfaceScalarField> correction(const Foam::volScalarField& field)const override {
        using namespace Foam;const auto& mesh=field.mesh();
        const auto gradient=fv::gradScheme<scalar>::New(mesh,ugkp::solverControls::thermalDiscretization(mesh).selection("gradSchemes","grad("+field.name()+')'))().grad(field,"grad("+field.name()+')');
        const auto& g=gradient();const auto& weights=mesh.weights();const auto& directions=mesh.nonOrthCorrectionVectors();
        std::vector<VectorFaceInterpolation> input;
        forAll(mesh.neighbour(),f)input.push_back({thermalVector(g[mesh.owner()[f]]),thermalVector(g[mesh.neighbour()[f]]),thermalVector(directions[f]),GpuReal(weights[f]),0});
        forAll(g.boundaryField(),p){const auto& patch=g.boundaryField()[p];const auto& cells=mesh.boundary()[p].faceCells();
            vectorField adjacent;if(patch.coupled())adjacent=patch.patchNeighbourField();
            forAll(patch,f)input.push_back({thermalVector(patch.coupled()?g[cells[f]]:patch[f]),patch.coupled()?thermalVector(adjacent[f]):ThermalVector{},thermalVector(directions.boundaryField()[p][f]),GpuReal(weights.boundaryField()[p][f]),patch.coupled()?1:2});}
        std::vector<GpuReal> values;thermalFieldBackend().dotInterpolate(input,values);
        auto result=surfaceScalarField::New("snGradCorr("+field.name()+')',mesh,field.dimensions()*mesh.nonOrthDeltaCoeffs().dimensions());
        auto& output=result.ref();int f=0;forAll(output,i)output[i]=values[f++];forAll(output.boundaryField(),p)forAll(output.boundaryField()[p],i)output.boundaryFieldRef()[p][i]=values[f++];
        return result;
    }
};
inline void installGpuThermalFields(){
    static const bool installed=[](){
        using Interpolation=Foam::surfaceInterpolationScheme<Foam::scalar>;
        using Gradient=Foam::fv::gradScheme<Foam::scalar>;
        using NormalGradient=Foam::fv::snGradScheme<Foam::scalar>;
        Interpolation::MeshConstructorTablePtr_->set("linear",+[](const Foam::fvMesh& mesh,Foam::Istream&)->Foam::tmp<Interpolation>{return Foam::tmp<Interpolation>(new FoamThermalLinear(mesh));});
        Interpolation::MeshFluxConstructorTablePtr_->set("linear",+[](const Foam::fvMesh& mesh,const Foam::surfaceScalarField&,Foam::Istream&)->Foam::tmp<Interpolation>{return Foam::tmp<Interpolation>(new FoamThermalLinear(mesh));});
        Interpolation::MeshConstructorTablePtr_->set("linearUpwind",+[](const Foam::fvMesh& mesh,Foam::Istream& input)->Foam::tmp<Interpolation>{return Foam::tmp<Interpolation>(new FoamThermalLinearUpwind(mesh,input));});
        Interpolation::MeshFluxConstructorTablePtr_->set("linearUpwind",+[](const Foam::fvMesh& mesh,const Foam::surfaceScalarField& flux,Foam::Istream& input)->Foam::tmp<Interpolation>{return Foam::tmp<Interpolation>(new FoamThermalLinearUpwind(mesh,flux,input));});
        NormalGradient::MeshConstructorTablePtr_->set("limited",+[](const Foam::fvMesh& mesh,Foam::Istream& input)->Foam::tmp<NormalGradient>{return thermalLimitedSnGrad(mesh,input);});
        Gradient::IstreamConstructorTablePtr_->set("Gauss",+[](const Foam::fvMesh& mesh,Foam::Istream& input)->Foam::tmp<Gradient>{return Foam::tmp<Gradient>(new FoamThermalGaussGradient(mesh,input));});
        NormalGradient::MeshConstructorTablePtr_->set("corrected",+[](const Foam::fvMesh& mesh,Foam::Istream&)->Foam::tmp<NormalGradient>{return Foam::tmp<NormalGradient>(new FoamThermalCorrectedGradient(mesh));});
        return true;
    }();(void)installed;
}
}}
#endif
