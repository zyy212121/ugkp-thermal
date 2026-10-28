#include "fvCFD.H"
#include "gpu/FoamSparseSolve.H"
#include "gpu/FoamImplicitDiffusion.H"
#include "IStringStream.H"
#include <iostream>
#include <iomanip>
#include <cmath>

// Native Amul receives the actual solution field, preserving OpenFOAM's
// identity-sensitive jump term. This audit is independent of GPU assembly.
Foam::scalarField trueResidual(Foam::fvScalarMatrix& equation){
    using namespace Foam;
    const scalarField diagonal(equation.diag());
    scalarField source(equation.source()),product(source.size());
    forAll(equation.internalCoeffs(),p){const auto& cells=equation.lduAddr().patchAddr(p);
        forAll(cells,f){equation.diag()[cells[f]]+=equation.internalCoeffs()[p][f];
            if(!equation.psi().boundaryField()[p].coupled())source[cells[f]]+=equation.boundaryCoeffs()[p][f];}}
    equation.Amul(product,equation.psi().primitiveField(),equation.boundaryCoeffs(),equation.psi().boundaryField().scalarInterfaces(),0);
    equation.diag()=diagonal;return source-product;
}
int main(int argc,char** argv){
    using namespace Foam;
    #include "setRootCase.H"
    #include "createTime.H"
    fvMesh mesh(IOobject("solid",runTime.timeName(),runTime,IOobject::MUST_READ));
    volScalarField initial(IOobject("T",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE),mesh);
    volScalarField cpu(IOobject("jumpCpu",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE),initial);
    volScalarField gpu(IOobject("jumpGpu",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE),initial);
    wordList types(mesh.boundary().size(),"zeroGradient");
    forAll(mesh.boundary(),p)if(mesh.boundary()[p].coupled())types[p]=mesh.boundary()[p].type();
    volScalarField capacity(IOobject("capacity",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE),mesh,dimensionedScalar("c",dimEnergy/dimVolume/dimTemperature,1),types);
    volScalarField kappa(IOobject("kappa",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE),mesh,dimensionedScalar("k",dimPower/dimLength/dimTemperature,1),types);
    volScalarField source(IOobject("source",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE),mesh,dimensionedScalar("s",dimPower/dimVolume,300));
    dictionary controls;controls.add("solver",word("PCG"));controls.add("preconditioner",word("DIC"));controls.add("tolerance",scalar(1e-12));controls.add("relTol",scalar(0));controls.add("maxIter",label(2000));
    ugkp::gpu::SparseConjugateGradient backend;
    IStringStream nativeText("linear corrected"),gpuText("linear corrected");
    fv::gaussLaplacianScheme<scalar,scalar> nativeDiffusion(mesh,nativeText);
    ugkp::gpu::FoamImplicitDiffusion gpuDiffusion(mesh,gpuText);
    mesh.schemes().setFluxRequired(cpu.name());mesh.schemes().setFluxRequired(gpu.name());
    bool passed=true;
    for(int sample=0;sample<4;++sample){
        // Reuse one solver across changing jump/time/field identity. The last
        // sample changes to ordinary cyclic to check stale affine storage.
        runTime.setTime(scalar(sample),sample);
        if(sample==3)forAll(mesh.boundary(),p)if(isA<cyclicFvPatch>(mesh.boundary()[p])){
            cpu.boundaryFieldRef().set(p,new cyclicFvPatchScalarField(mesh.boundary()[p],cpu.internalField()));
            gpu.boundaryFieldRef().set(p,new cyclicFvPatchScalarField(mesh.boundary()[p],gpu.internalField()));
        }
        const dimensionedScalar dt("dt",dimTime,sample==2?.2:1.);
        forAll(cpu,c){const vector x=mesh.C()[c];
            cpu[c]=gpu[c]=sample==0?300:300+21*std::sin(x.y()*2+x.x()+x.z());
            capacity[c]=sample==2?1+.2*x.y():1;
            kappa[c]=sample==2?1+.1*x.y():1;
            source[c]=capacity[c]/dt.value()*cpu[c];
        }
        capacity.correctBoundaryConditions();kappa.correctBoundaryConditions();
        cpu.correctBoundaryConditions();gpu.correctBoundaryConditions();
        for(int correction=0;correction<2;++correction){
            fvScalarMatrix native(fvm::Sp(capacity/dt,cpu)-static_cast<fv::laplacianScheme<scalar,scalar>&>(nativeDiffusion).fvmLaplacian(kappa,cpu)==source);
            fvScalarMatrix device(static_cast<fv::laplacianScheme<scalar,scalar>&>(gpuDiffusion).fvmLaplacian(kappa,gpu));
            ugkp::gpu::formGpuImplicitEquation(device,capacity,source,dt.value());
            const auto a=native.solve(controls),b=ugkp::gpu::solveFoamScalarMatrix(device,controls,backend);
            const scalarField ar(trueResidual(native)),br(trueResidual(device));
            double temperatureError=0,residualMax=0,nativeResidualMax=0,fluxError=0,globalResidual=0;
            forAll(cpu,c){temperatureError=std::max(temperatureError,std::abs(cpu[c]-gpu[c]));
                residualMax=std::max(residualMax,std::abs(br[c]));nativeResidualMax=std::max(nativeResidualMax,std::abs(ar[c]));globalResidual+=br[c];}
            const auto af=native.flux(),bf=device.flux();
            forAll(af().boundaryField(),p)forAll(af().boundaryField()[p],f)
                fluxError=std::max(fluxError,std::abs(af().boundaryField()[p][f]-bf().boundaryField()[p][f]));
            const double normalizationError=std::abs(a.initialResidual()-b.initialResidual());
            const bool ok=a.converged()&&b.converged()&&temperatureError<2e-8&&residualMax<2e-8&&fluxError<2e-8
                &&(correction!=0||normalizationError<2e-10);
            passed=passed&&ok;
            std::cout<<std::setprecision(17)<<"JUMP_RESULT "<<sample<<' '<<correction<<' '<<temperatureError<<' '<<residualMax<<' '<<nativeResidualMax<<' '<<fluxError<<' '<<a.initialResidual()<<' '<<b.initialResidual()<<' '<<b.finalResidual()<<' '<<globalResidual<<' '<<ok<<'\n';
            forAll(cpu,c)std::cout<<"CELL "<<sample<<' '<<correction<<' '<<c<<' '<<cpu[c]<<' '<<gpu[c]<<' '<<ar[c]<<' '<<br[c]<<'\n';
            forAll(af().boundaryField(),p)forAll(af().boundaryField()[p],f)
                std::cout<<"FLUX "<<sample<<' '<<correction<<' '<<p<<' '<<f<<' '<<af().boundaryField()[p][f]<<' '<<bf().boundaryField()[p][f]<<'\n';
        }
    }
    std::cout<<(passed?"PASS":"FAIL")<<" native jump temperature, true cell residual, boundary flux, initial normalization and linear Krylov operations\n";
    return passed?0:1;
}
