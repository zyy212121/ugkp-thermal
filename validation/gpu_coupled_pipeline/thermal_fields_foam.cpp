#include "fvCFD.H"
#include "gpu/FoamThermalFieldOperators.H"
#include "fixedValueFvPatchFields.H"
#include "IStringStream.H"
#include <iostream>
#include <stdexcept>
template<class Field> double difference(const Field& a,const Field& b){
    double error=0;forAll(a,c)error=std::max(error,double(Foam::mag(a[c]-b[c])));
    forAll(a.boundaryField(),p)forAll(a.boundaryField()[p],f)error=std::max(error,double(Foam::mag(a.boundaryField()[p][f]-b.boundaryField()[p][f])));
    return error;
}
int main(int argc,char** argv){
    using namespace Foam;
    #include "setRootCase.H"
    #include "createTime.H"
    fvMesh mesh(IOobject("solid",runTime.timeName(),runTime,IOobject::MUST_READ));
    wordList patches(mesh.boundary().size(),fixedValueFvPatchScalarField::typeName);
    forAll(mesh.boundary(),p)if(mesh.boundary()[p].coupled()||mesh.boundary()[p].type()=="empty")patches[p]=mesh.boundary()[p].type();
    volScalarField field(IOobject("fieldOracle",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE),mesh,dimensionedScalar("initial",dimTemperature,300),patches);
    forAll(field,c){const auto& x=mesh.C()[c];field[c]=300+90*std::sin(9*x.x()+3*x.y()-7*x.z());}
    forAll(field.boundaryField(),p)if(!field.boundaryField()[p].coupled())forAll(field.boundaryField()[p],f){const auto x=mesh.Cf().boundaryField()[p][f];field.boundaryFieldRef()[p][f]=400+90*x.x()+50*x.y();}
    field.correctBoundaryConditions();
    const auto a=linear<scalar>(mesh).interpolate(field);
    const auto b=ugkp::gpu::FoamThermalLinear(mesh).interpolate(field);
    const auto ga=fv::gaussGrad<scalar>(mesh).grad(field,"nativeGradient");
    IStringStream empty("linear");ugkp::gpu::FoamThermalGaussGradient device(mesh,empty);
    const auto gb=device.grad(field,"deviceGradient");
    const auto sa=fv::correctedSnGrad<scalar>(mesh).correction(field);
    const auto sb=ugkp::gpu::FoamThermalCorrectedGradient(mesh).correction(field);
    const double errors[]={difference(a(),b()),difference(ga(),gb()),difference(sa(),sb())};
    for(double error:errors)if(error>5e-11)throw std::runtime_error("native/device thermal field mismatch");
    auto harmonic=surfaceScalarField::New("harmonic",b());
    ugkp::gpu::interpolateThermalField(field,mesh.weights(),harmonic.ref(),true);
    double harmonicError=0;
    forAll(mesh.neighbour(),f){const double own=field[mesh.owner()[f]],neighbour=field[mesh.neighbour()[f]],w=mesh.weights()[f];const double expected=own>0&&neighbour>0?1/((1-w)/own+w/neighbour):0;harmonicError=std::max(harmonicError,std::abs(harmonic()[f]-expected));}
    if(harmonicError!=0)throw std::runtime_error("harmonic interpolation differs");
    std::cout<<"FIELD_RESULT "<<mesh.nCells()<<' '<<errors[0]<<' '<<errors[1]<<' '<<errors[2]<<' '<<harmonicError<<'\n';
    std::cout<<"PASS native/GPU interpolation, Gauss gradient, boundary correction and corrected normal derivative\n";
}
