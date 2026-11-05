#include "fvCFD.H"
#include "GpuSolidThermalCoupler.H"
#include "GpuSolidThermalProperties.H"
#include "IStringStream.H"
#include <iomanip>
#include <iostream>
using namespace Foam;
using namespace Foam::gpuThermal;
int main(int argc,char** argv){
    #include "setRootCase.H"
    #include "createTime.H"
    fvMesh mesh(IOobject("solid",runTime.timeName(),runTime,IOobject::MUST_READ));
    volScalarField temperature(IOobject("T",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE),mesh);
    labelList interfaces(2);interfaces[0]=mesh.boundaryMesh().findPatchID("left");interfaces[1]=mesh.boundaryMesh().findPatchID("right");
    temperature.correctBoundaryConditions();
    IStringStream text("rho {type constant; value 1;} Cp {type constant; value 1;} kappa {type constant; value 1;} emissivity {type constant; value 0;}");
    dictionary dict(text);GpuSolidThermalProperties properties(dict);
    GpuSolidThermalControls controls;controls.nonlinearMaxIterations=100;controls.nonlinearRelativeTolerance=1e-12;controls.energyRelativeTolerance=1e-12;controls.energyAbsoluteTolerance=1e-10;
    List<scalarField> energy(2);forAll(interfaces,p)energy[p]=scalarField(mesh.boundary()[interfaces[p]].size(),0);
    GpuSolidThermalCandidateSolver solver(mesh,properties,interfaces,controls);
    const auto result=solver.solveTemporarySolidCandidate(temperature,energy,1.);
    std::cout<<std::setprecision(17)<<"OWNER_RESULT "<<result.nonlinearIterations<<' '<<result.integratedEnergyResidualJ<<'\n';
    forAll(temperature,c)std::cout<<"OWNER_CELL "<<c<<' '<<result.temperature()[c]<<'\n';
    const auto gradient=fvc::snGrad(result.temperature());
    forAll(gradient().boundaryField(),p){const auto& grad=gradient().boundaryField()[p];
        forAll(grad,f)std::cout<<"OWNER_FLUX "<<p<<' '<<f<<' '<<-grad[f]*mesh.magSf().boundaryField()[p][f]<<'\n';}
    return std::abs(result.integratedEnergyResidualJ)>1e-8?1:0;
}
