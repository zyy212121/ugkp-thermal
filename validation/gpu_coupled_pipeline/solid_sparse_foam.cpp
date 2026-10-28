#include "fvCFD.H"
#include "gpu/FoamSparseSolve.H"
#include "gpu/FoamImplicitDiffusion.H"
#include "IStringStream.H"
#include "fixedValueFvPatchFields.H"
#include "zeroGradientFvPatchFields.H"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cstdlib>
Foam::scalarField linearResidual(Foam::fvScalarMatrix& equation){
    using namespace Foam;
    const scalarField saved(equation.diag());scalarField source(equation.source()),product(source.size());
    forAll(equation.internalCoeffs(),p){const auto& cells=equation.lduAddr().patchAddr(p);
        forAll(cells,f){equation.diag()[cells[f]]+=equation.internalCoeffs()[p][f];
            if(!equation.psi().boundaryField()[p].coupled())source[cells[f]]+=equation.boundaryCoeffs()[p][f];}}
    equation.Amul(product,equation.psi().primitiveField(),equation.boundaryCoeffs(),equation.psi().boundaryField().scalarInterfaces(),0);
    equation.diag()=saved;return source-product;
}
int main(int argc,char** argv){
    using namespace Foam;
    #include "setRootCase.H"
    #include "createTime.H"
    fvMesh mesh(IOobject("solid",runTime.timeName(),runTime,IOobject::MUST_READ));
    wordList patches(mesh.boundary().size(),fixedValueFvPatchScalarField::typeName);
    forAll(mesh.boundary(),p)if(mesh.boundary()[p].coupled()||mesh.boundary()[p].type()=="empty")patches[p]=mesh.boundary()[p].type();
    volScalarField cpu(IOobject("sparseOracleT",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE),mesh,dimensionedScalar("initial",dimTemperature,300),patches);
    volScalarField gpu(IOobject("sparseDeviceT",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE),cpu);
    const dimensionSet capacityDimensions=dimEnergy/dimVolume/dimTemperature;
    const dimensionSet conductivityDimensions=dimPower/dimLength/dimTemperature;
    wordList coefficients(mesh.boundary().size(),zeroGradientFvPatchScalarField::typeName);
    forAll(mesh.boundary(),p)if(mesh.boundary()[p].coupled()||mesh.boundary()[p].type()=="empty")coefficients[p]=mesh.boundary()[p].type();
    volScalarField capacity(IOobject("capacity",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE),mesh,dimensionedScalar("one",capacityDimensions,1),coefficients);
    volScalarField kappa(IOobject("kappa",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE),mesh,dimensionedScalar("one",conductivityDimensions,1),coefficients);
    volScalarField source(IOobject("source",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE),mesh,dimensionedScalar("one",dimPower/dimVolume,1));
    dictionary controls;controls.add("solver",word("PCG"));controls.add("preconditioner",word("DIC"));controls.add("tolerance",scalar(1e-12));controls.add("relTol",scalar(0));controls.add("maxIter",label(2000));
    ugkp::gpu::SparseConjugateGradient backend;
    const std::string scheme = std::getenv("UGKP_TEST_DIFFUSION") ? std::getenv("UGKP_TEST_DIFFUSION") : "linear corrected";
    IStringStream nativeSpecification(scheme), diffusionSpecification(scheme);
    fv::gaussLaplacianScheme<scalar,scalar> nativeDiffusion(mesh,nativeSpecification);
    ugkp::gpu::FoamImplicitDiffusion deviceDiffusion(mesh,diffusionSpecification);
    mesh.schemes().setFluxRequired(cpu.name());mesh.schemes().setFluxRequired(gpu.name());
    for(int sample=0;sample<3;++sample){
        const dimensionedScalar dt("dt",dimTime, sample==0?.001:sample==1?.1:10.);
        forAll(cpu,c){
            const auto& centre=mesh.C()[c];const double initial=300+11*std::sin(centre.x()*9+centre.y()*3);
            cpu[c]=gpu[c]=initial;capacity[c]=1+.3*std::abs(std::sin(centre.y()*7));
            kappa[c]=.1+10*std::abs(std::sin(centre.x()*4+centre.y()*5));source[c]=capacity[c]/dt.value()*initial;
        }
        forAll(cpu.boundaryField(),p)if(!mesh.boundary()[p].coupled()&&mesh.boundary()[p].type()!="empty")
            forAll(cpu.boundaryField()[p],f){const auto centre=mesh.Cf().boundaryField()[p][f];cpu.boundaryFieldRef()[p][f]=gpu.boundaryFieldRef()[p][f]=400+90*centre.x()+50*centre.y();}
        capacity.correctBoundaryConditions();kappa.correctBoundaryConditions();cpu.correctBoundaryConditions();gpu.correctBoundaryConditions();
        double cpuSeconds=0,gpuSeconds=0,maxError=0;int nativeIterations=0,deviceIterations=0;
        for(int correction=0;correction<4;++correction){
            fvScalarMatrix native(fvm::Sp(capacity/dt,cpu)-static_cast<fv::laplacianScheme<scalar,scalar>&>(nativeDiffusion).fvmLaplacian(kappa,cpu)==source);
            fvScalarMatrix device(static_cast<fv::laplacianScheme<scalar,scalar>&>(deviceDiffusion).fvmLaplacian(kappa,gpu));
            ugkp::gpu::formGpuImplicitEquation(device,capacity,source,dt.value());
            auto start=std::chrono::steady_clock::now();auto a=native.solve(controls);
            cpuSeconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            start=std::chrono::steady_clock::now();auto b=ugkp::gpu::solveFoamScalarMatrix(device,controls,backend);
            gpuSeconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            nativeIterations+=a.nIterations();deviceIterations+=b.nIterations();
            if(!a.converged()||!b.converged())throw std::runtime_error("linear residual target not reached");
            const scalarField nativeResidual(linearResidual(native)),deviceResidual(linearResidual(device));
            double nativeL1=0,deviceL1=0,scale=0;
            forAll(cpu,c){maxError=std::max(maxError,std::abs(cpu[c]-gpu[c]));nativeL1+=std::abs(nativeResidual[c]);deviceL1+=std::abs(deviceResidual[c]);scale+=std::abs(source[c]*mesh.V()[c]);}
            forAll(device.boundaryCoeffs(),p)if(!device.psi().boundaryField()[p].coupled())for(const double value:device.boundaryCoeffs()[p])scale+=std::abs(value);
            std::cerr<<"CHECK "<<maxError<<" "<<nativeL1/scale<<" "<<deviceL1/scale<<" "<<b.initialResidual()<<" "<<b.finalResidual()<<" "<<b.nIterations()<<std::endl;
            if(maxError>2e-8||deviceL1>2e-10*scale)throw std::runtime_error("temperature or true integral residual mismatch");
            const auto af=native.flux(),bf=device.flux();double maxFluxError=0;
            forAll(af().boundaryField(),p)forAll(af().boundaryField()[p],f)maxFluxError=std::max(maxFluxError,std::abs(af().boundaryField()[p][f]-bf().boundaryField()[p][f]));
            if(maxFluxError>2e-8)throw std::runtime_error("boundary heat flux mismatch");
        }
        std::cout<<std::setprecision(17)<<"SPARSE_RESULT "<<mesh.nCells()<<' '<<sample<<' '<<maxError<<' '<<nativeIterations<<' '<<deviceIterations<<' '<<cpuSeconds<<' '<<gpuSeconds<<'\n';
    }
    std::cout<<"PASS native implicit matrices, corrected faces, boundary heat flux and true residual\n";
}
