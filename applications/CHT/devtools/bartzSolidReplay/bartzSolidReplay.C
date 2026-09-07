#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "IOdictionary.H"
#include "instant.H"
#include "GpuSolidThermalProperties.H"
#include "GpuSolidThermalCoupler.H"

#include <cmath>
#include <fstream>
#include <string>

using namespace Foam;
using namespace Foam::gpuThermal;

namespace
{
GpuSolidThermalControls readControls(const dictionary& coupling)
{
    GpuSolidThermalControls controls;
    controls.nonlinearMaxIterations =
        readLabel(coupling.lookup("nonlinearMaxIterations"));
    controls.nonlinearRelativeTolerance =
        readScalar(coupling.lookup("nonlinearRelativeTolerance"));
    controls.energyRelativeTolerance =
        readScalar(coupling.lookup("energyRelativeTolerance"));
    controls.energyAbsoluteTolerance =
        readScalar(coupling.lookup("energyAbsoluteTolerance"));
    controls.nonlinearAitken =
        coupling.lookupOrDefault<bool>("nonlinearAitken", false);
    controls.nonlinearInitialRelaxation =
        coupling.lookupOrDefault<scalar>("nonlinearInitialRelaxation", 1);
    controls.nonlinearMinimumRelaxation =
        coupling.lookupOrDefault<scalar>("nonlinearMinimumRelaxation", 0.05);
    controls.nonlinearMaximumRelaxation =
        coupling.lookupOrDefault<scalar>("nonlinearMaximumRelaxation", 1);
    return controls;
}
}

int main(int argc, char *argv[])
{
    argList::addOption("fluxFile", "file", "Bartz wall heat-flux schedule");
    #include "setRootCase.H"
    #include "createTime.H"

    const fileName fluxFile(args.optionRead<fileName>("fluxFile"));

    IOdictionary solidRegionProperties
    (
        IOobject
        (
            "solidRegionProperties",
            runTime.constant(),
            runTime,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );
    const dictionary& coupling =
        solidRegionProperties.subDict("solidThermalCoupling");
    const word solidRegion(coupling.lookup("solidRegion"));
    const word solidPatchName(coupling.lookup("solidPatch"));

    fvMesh solidMesh
    (
        IOobject
        (
            solidRegion,
            runTime.timeName(),
            runTime,
            IOobject::MUST_READ
        )
    );
    volScalarField sourceTemperature
    (
        IOobject
        (
            "T",
            runTime.timeName(),
            solidMesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        ),
        solidMesh
    );
    volScalarField Tbartz
    (
        IOobject
        (
            "Tbartz",
            runTime.timeName(),
            solidMesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        sourceTemperature
    );

    const label solidPatchId =
        solidMesh.boundaryMesh().findPatchID(solidPatchName);
    if (solidPatchId < 0)
    {
        FatalErrorInFunction
            << "Solid patch is absent: " << solidPatchName
            << exit(FatalError);
    }

    GpuSolidThermalProperties properties(coupling.subDict("properties"));
    labelList coupledPatchIds(1, solidPatchId);
    GpuSolidThermalCandidateSolver solver
    (
        solidMesh,
        properties,
        coupledPatchIds,
        readControls(coupling)
    );

    std::ifstream input(fluxFile.c_str());
    label recordCount = 0;
    if (!(input >> recordCount) || recordCount <= 0)
    {
        FatalErrorInFunction
            << "Invalid Bartz schedule: " << fluxFile
            << exit(FatalError);
    }

    const fvPatch& solidPatch = solidMesh.boundary()[solidPatchId];
    for (label recordI = 0; recordI < recordCount; ++recordI)
    {
        std::string directoryName;
        scalar targetTime = 0;
        scalar deltaT = 0;
        label faceCount = 0;
        if (!(input >> directoryName >> targetTime >> deltaT >> faceCount))
        {
            FatalErrorInFunction
                << "Malformed Bartz schedule record " << recordI
                << exit(FatalError);
        }
        if
        (
            !std::isfinite(static_cast<double>(targetTime))
         || !std::isfinite(static_cast<double>(deltaT))
         || deltaT < 0
         || (deltaT == 0 && faceCount != 0)
         || (deltaT > 0 && faceCount != solidPatch.size())
        )
        {
            FatalErrorInFunction
                << "Invalid Bartz schedule dimensions at record " << recordI
                << exit(FatalError);
        }

        runTime.setTime
        (
            instant(targetTime, word(directoryName)),
            recordI
        );

        if (deltaT > 0)
        {
            List<scalarField> interfaceEnergyJ(1);
            interfaceEnergyJ[0].setSize(faceCount);
            forAll(interfaceEnergyJ[0], faceI)
            {
                scalar wallHeatFlux = 0;
                if
                (
                    !(input >> wallHeatFlux)
                 || !std::isfinite(static_cast<double>(wallHeatFlux))
                )
                {
                    FatalErrorInFunction
                        << "Invalid Bartz heat flux at record " << recordI
                        << ", face " << faceI
                        << exit(FatalError);
                }
                interfaceEnergyJ[0][faceI] =
                    wallHeatFlux
                   *solidMesh.magSf().boundaryField()[solidPatchId][faceI]
                   *deltaT;
            }
            GpuSolidThermalCandidate candidate =
                solver.solveTemporarySolidCandidate
                (
                    Tbartz,
                    interfaceEnergyJ,
                    deltaT
                );
            solver.publishSolidCandidate(candidate, Tbartz);
        }

        if (!Tbartz.write())
        {
            FatalErrorInFunction
                << "Failed to write Tbartz at " << directoryName
                << exit(FatalError);
        }
        Info
            << "Tbartz time=" << directoryName
            << " deltaT=" << deltaT << nl;
    }

    Info<< "Tbartz records=" << recordCount << nl;
    return 0;
}
