#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "IOdictionary.H"
#include "fixedGradientFvPatchFields.H"
#include "GpuSolidThermalCoupler.H"
#include "GpuThermalExchangeState.H"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
#include <stdexcept>
extern int chtSurfaceMaterializations;
extern int chtSurfaceSnapshots;
extern int chtSurfaceFault;
using namespace Foam;
using namespace Foam::gpuThermal;
void near(const char* name, scalar actual, scalar expected, scalar tolerance=1e-8)
{
    if (!std::isfinite(actual) || std::abs(actual-expected)>tolerance*std::max(scalar(1),std::abs(expected)))
    {
        std::cerr << name << ": actual=" << actual << " expected=" << expected << '\n';
        throw std::runtime_error(name);
    }
}
struct RadiationEnergy { List<scalarField> solidWallRadiationEnergyJ; };
int main(int argc, char** argv)
{
    #include "setRootCase.H"
    #include "createTime.H"
    fvMesh mesh(IOobject(fvMesh::defaultRegion, runTime.timeName(), runTime, IOobject::MUST_READ));
    volScalarField T(IOobject("T", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE),mesh);
    IOdictionary config(IOobject("testProperties",runTime.constant(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE));
    GpuSolidThermalProperties properties(config.subDict("properties"));
    const scalar q=readScalar(config.lookup("heatFlux"));
    const scalar dt=0.1;
    const bool balanced=config.lookupOrDefault<bool>("balanced",false);
    const bool variable=config.lookupOrDefault<bool>("variable",false);
    const bool ownerConductivity=config.lookupOrDefault<bool>("ownerConductivity",false);
    const bool mixedOwnerConductivity=config.lookupOrDefault<bool>("mixedOwnerConductivity",false);
    const label left=mesh.boundaryMesh().findPatchID("left");
    const label right=mesh.boundaryMesh().findPatchID("right");
    labelList patches(balanced?2:1,left);
    if (balanced) patches[1]=right;
    List<scalarField> energy(patches.size());
    forAll(patches,p)
    {
        energy[p].setSize(mesh.boundary()[patches[p]].size());
        forAll(energy[p],f) energy[p][f]=(p?-q:q)*dt*mesh.magSf().boundaryField()[patches[p]][f];
    }
    const word channel(config.lookupOrDefault<word>("energyChannel","gas"));
    const label pairI=0;
    const label fluidPatchI=left;
    const polyPatch& fluidPatch=mesh.boundaryMesh()[left];
    List<scalarField> gasPreview(1,scalarField((channel=="gas"?1:channel=="mixed"?0.25:0)*energy[0]));
    scalarField particleContactPreview(mesh.nFaces(),0);
    forAll(energy[0],f) particleContactPreview[fluidPatch.start()+f]=(channel=="particle"?1:channel=="mixed"?0.5:0)*energy[0][f];
    const bool radiationDue=channel=="radiation" || channel=="mixed";
    autoPtr<RadiationEnergy> radiation(new RadiationEnergy);
    radiation().solidWallRadiationEnergyJ.setSize(mesh.boundary().size());
    radiation().solidWallRadiationEnergyJ[left]=scalarField((channel=="radiation"?1:channel=="mixed"?0.25:0)*energy[0]);
    #include "CompletedIntervalEnergy.inc"
    near("unchanged completed-interval aggregate",combined[0],energy[0][0],0);
    energy[0]=combined;
    const List<scalarField> originalEnergy(energy);
    GpuSolidThermalControls controls;
    controls.nonlinearMaxIterations=config.lookupOrDefault<label>("maximumIterations",200);
    controls.nonlinearAitken=config.lookupOrDefault<bool>("aitken",false);
    controls.nonlinearInitialRelaxation=0.5;
    controls.nonlinearRelativeTolerance=1e-12;
    controls.energyAbsoluteTolerance=1e-9;
    const labelList ownerPatches=ownerConductivity?patches:mixedOwnerConductivity?labelList(1,right):labelList();
    GpuSolidThermalCandidateSolver solver(mesh,properties,patches,controls,ownerPatches);
    near("startup surface reconstructed",coupledPatchSurfaceTemperature(T,left)[0],config.lookupOrDefault<scalar>("initialSurface",300));
    T.boundaryFieldRef()[left][0]=123;
    bool staleRejected=false;
    try { coupledPatchSurfaceTemperature(T,left); }
    catch (const GpuSolidThermalSolveError&) { staleRejected=true; }
    if (!staleRejected) throw std::runtime_error("stale surface was accepted");
    T.correctBoundaryConditions();
    fixedGradientFvPatchScalarField& startingPatch=refCast<fixedGradientFvPatchScalarField>(T.boundaryFieldRef()[left]);
    const scalar savedGradient=startingPatch.gradient()[0];
    startingPatch.gradient()[0]=std::numeric_limits<scalar>::quiet_NaN();
    bool invalidGradientRejected=false;
    try { coupledPatchSurfaceTemperature(T,left); }
    catch (const GpuSolidThermalSolveError&) { invalidGradientRejected=true; }
    if (!invalidGradientRejected) throw std::runtime_error("nonfinite surface gradient was accepted");
    startingPatch.gradient()[0]=savedGradient;
    T.correctBoundaryConditions();
    chtSurfaceMaterializations=0;
    chtSurfaceSnapshots=0;
    chtSurfaceFault=config.lookupOrDefault<label>("surfaceFault",0);
    GpuSolidThermalCandidate candidate=solver.solveTemporarySolidCandidate(T,energy,dt);
    if (config.lookupOrDefault<bool>("checkSurfaceWork",false))
    {
        std::cout << "surface work materializations=" << chtSurfaceMaterializations
                  << " snapshots=" << chtSurfaceSnapshots << std::endl;
        const label expectedSnapshots=patches.size()-ownerPatches.size();
        if (chtSurfaceMaterializations!=0 || chtSurfaceSnapshots!=expectedSnapshots)
            throw std::runtime_error("candidate convergence still materializes surface fields per iteration");
        std::cout << "PASS surface work materializations=0 snapshots=" << expectedSnapshots << '\n';
    }
    const scalar ownerExpected=300+(balanced?0:energy[0][0]/(1000*mesh.V()[0]));
    const scalar owner=candidate.temperature()[0];
    const scalar surface=candidate.temperature().boundaryField()[left][0];
    const scalar delta=mesh.boundary()[left].deltaCoeffs()[0];
    forAll(candidate.temperature().primitiveField(),c) near("owner integrated ledger",candidate.temperature()[c],ownerExpected);
    near("registered owner unchanged before publish",T[0],300);
    near("energy ledger unchanged",energy[0][0],originalEnergy[0][0],0);
    near("integrated energy residual",candidate.integratedEnergyResidualJ,0,1e-8);
    const scalar expectedSurface=ownerConductivity ? owner+q/(properties.kappa(owner)*delta) : variable ? (owner+std::sqrt(owner*owner+4*q/(0.02*delta)))/2 : owner+q/(6*delta);
    near("surface closure",surface,expectedSurface);
    near("face Fourier flux",properties.kappa(ownerConductivity?owner:surface)*delta*(surface-owner),q);
    if (balanced && !variable)
    {
        const scalar gasHalfResistance=0.2/2;
        const scalar gasOwner=300+q*(gasHalfResistance+1/(6*delta));
        near("two half-cell analytic interface",surface,gasOwner-q*gasHalfResistance);
        near("two half-cell gas flux",(gasOwner-surface)/gasHalfResistance,q);
        near("opposite solid flux",candidate.temperature().boundaryField()[right][0],owner-q/(6*mesh.boundary()[right].deltaCoeffs()[0]));
    }
    forAll(patches,p)
    {
        const bool useOwner=ownerConductivity || (mixedOwnerConductivity && p==1);
        const scalar faceFlux=p?-q:q;
        const scalarField& faceSurface=candidate.temperature().boundaryField()[patches[p]];
        const fixedGradientFvPatchScalarField& facePatch=
            refCast<const fixedGradientFvPatchScalarField>(candidate.temperature().boundaryField()[patches[p]]);
        forAll(faceSurface,f)
        {
            const scalar ownerT=candidate.temperature()[mesh.boundary()[patches[p]].faceCells()[f]];
            const scalar d=mesh.boundary()[patches[p]].deltaCoeffs()[f];
            const scalar expected=useOwner?ownerT+faceFlux/(properties.kappa(ownerT)*d)
                :variable?(ownerT+std::sqrt(ownerT*ownerT+4*faceFlux/(0.02*d)))/2:ownerT+faceFlux/(6*d);
            near("every coupled face surface closure",faceSurface[f],expected);
            near("every coupled face flux closure",properties.kappa(useOwner?ownerT:faceSurface[f])*facePatch.gradient()[f],faceFlux);
            near("every interface ledger unchanged",energy[p][f],originalEnergy[p][f],0);
        }
    }
    solver.publishSolidCandidate(candidate,T);
    near("published surface",T.boundaryField()[left][0],surface);
    near("published gradient",refCast<const fixedGradientFvPatchScalarField>(T.boundaryField()[left]).gradient()[0],q/properties.kappa(ownerConductivity?owner:surface));
    near("surface accessor",coupledPatchSurfaceTemperature(T,left)[0],surface);
    near("owner accessor stays at node",coupledPatchOwnerTemperature(T,left)[0],owner);
    if (!T.write()) throw std::runtime_error("checkpoint write failed");
    volScalarField restarted(IOobject("T",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE,false),mesh);
    near("restart surface",coupledPatchSurfaceTemperature(restarted,left)[0],surface,0);
    near("restart gradient",refCast<const fixedGradientFvPatchScalarField>(restarted.boundaryField()[left]).gradient()[0],refCast<const fixedGradientFvPatchScalarField>(T.boundaryField()[left]).gradient()[0],0);
    forAll(patches,p)
    {
        const scalarField savedSurface=coupledPatchSurfaceTemperature(restarted,patches[p]);
        const scalarField& expected=T.boundaryField()[patches[p]];
        const scalarField& savedGradient=refCast<const fixedGradientFvPatchScalarField>(restarted.boundaryField()[patches[p]]).gradient();
        const scalarField& expectedGradient=refCast<const fixedGradientFvPatchScalarField>(T.boundaryField()[patches[p]]).gradient();
        forAll(expected,f)
        {
            near("every restart surface",savedSurface[f],expected[f],0);
            near("every restart gradient",savedGradient[f],expectedGradient[f],0);
        }
    }
    const word acceptedHash=ThermalRestartPreflight::canonicalWallTemperatureSha1(T,patches);
    const word restartHash=ThermalRestartPreflight::canonicalWallTemperatureSha1(restarted,patches);
    if (acceptedHash!=restartHash) throw std::runtime_error("restart surface checksum mismatch");
    restarted.boundaryFieldRef()[left][0]=owner;
    if (acceptedHash==ThermalRestartPreflight::canonicalWallTemperatureSha1(restarted,patches))
        throw std::runtime_error("owner-era checkpoint checksum accepted as surface");
    std::cout << std::setprecision(17);
    forAll(T.primitiveField(),c) std::cout << "CELL " << c << ' ' << T[c] << '\n';
    forAll(patches,p)
    {
        const scalarField& surface=T.boundaryField()[patches[p]];
        const scalarField& gradient=refCast<const fixedGradientFvPatchScalarField>(T.boundaryField()[patches[p]]).gradient();
        forAll(surface,f) std::cout << "FACE " << p << ' ' << f << ' ' << surface[f] << ' ' << gradient[f] << '\n';
    }
    std::cout << "STATE owner=" << owner
              << " surface=" << surface << " gradient="
              << refCast<const fixedGradientFvPatchScalarField>(T.boundaryField()[left]).gradient()[0]
              << " energyResidual=" << candidate.integratedEnergyResidualJ
              << " iterations=" << candidate.nonlinearIterations << " hash=" << acceptedHash << '\n';
    std::cout << "PASS surface and ledger closure iterations=" << candidate.nonlinearIterations << '\n';
}
