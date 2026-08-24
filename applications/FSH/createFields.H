#ifndef createFields_H
#define createFields_H

#include "readGpuGasConfiguration.H"

Switch solveParticleTemperature
(
    ugkwpProps.lookupOrDefault<Switch>
    (
        "particleTemperatureTransport",
        Switch(false)
    )
);

dimensionedScalar rhoS
(
    "rhoS",
    dimensionSet(1, -3, 0, 0, 0, 0, 0),
    ugkwpProps.lookupOrDefault<scalar>("rhoS", 2500.0)
);

dimensionedScalar TpMin
(
    "TpMin",
    dimTemperature,
    ugkwpProps.lookupOrDefault<scalar>
    (
        "particleTMin",
        scalar(300.0)
    )
);

dimensionedScalar TpMax
(
    "TpMax",
    dimTemperature,
    ugkwpProps.lookupOrDefault<scalar>
    (
        "particleTMax",
        scalar(5000.0)
    )
);

dimensionedScalar dS
(
    "dS",
    dimensionSet(0, 1, 0, 0, 0, 0, 0),
    ugkwpProps.lookupOrDefault<scalar>("dS", 1.0e-3)
);

dimensionedScalar parcelMass
(
    "parcelMass",
    dimensionSet(1, 0, 0, 0, 0, 0, 0),
    ugkwpProps.lookupOrDefault<scalar>("parcelMass", 1.0e-9)
);

dimensionedScalar injectionParcelMass
(
    "injectionParcelMass",
    dimensionSet(1, 0, 0, 0, 0, 0, 0),
    ugkwpProps.lookupOrDefault<scalar>
    (
        "injectionParcelMass",
        parcelMass.value()
    )
);

dimensionedScalar legacyRestartParcelMass
(
    "legacyRestartParcelMass",
    dimensionSet(1, 0, 0, 0, 0, 0, 0),
    ugkwpProps.lookupOrDefault<scalar>
    (
        "legacyRestartParcelMass",
        parcelMass.value()
    )
);

if (!finiteScalar(rhoS.value()) || rhoS.value() <= SMALL)
{
    FatalErrorInFunction
        << "ugkwpProperties: rhoS must be positive. Current rhoS = "
        << rhoS.value() << exit(FatalError);
}

if (!finiteScalar(dS.value()) || dS.value() <= SMALL)
{
    FatalErrorInFunction
        << "ugkwpProperties: dS must be positive. Current dS = "
        << dS.value() << exit(FatalError);
}

if (!finiteScalar(parcelMass.value()) || parcelMass.value() <= SMALL)
{
    FatalErrorInFunction
        << "ugkwpProperties: parcelMass must be positive. Current parcelMass = "
        << parcelMass.value() << exit(FatalError);
}

if
(
    !finiteScalar(injectionParcelMass.value())
 || injectionParcelMass.value() <= SMALL
)
{
    FatalErrorInFunction
        << "ugkwpProperties: injectionParcelMass must be positive. Current "
        << "injectionParcelMass = " << injectionParcelMass.value()
        << exit(FatalError);
}

if
(
    !finiteScalar(legacyRestartParcelMass.value())
 || legacyRestartParcelMass.value() <= SMALL
)
{
    FatalErrorInFunction
        << "ugkwpProperties: legacyRestartParcelMass must be positive. "
        << "Current legacyRestartParcelMass = "
        << legacyRestartParcelMass.value() << exit(FatalError);
}

typeIOobject<uniformDimensionedVectorField> gravityHeader
(
    "g",
    runTime.constant(),
    mesh,
    IOobject::MUST_READ,
    IOobject::NO_WRITE
);

vector gravityValue(vector::zero);
if (gravityHeader.headerOk())
{
    uniformDimensionedVectorField gravityField(gravityHeader);
    if (gravityField.dimensions() != dimAcceleration)
    {
        FatalErrorInFunction
            << "constant/g must have acceleration dimensions. Current "
            << "dimensions = " << gravityField.dimensions()
            << exit(FatalError);
    }
    gravityValue = gravityField.value();
}
if
(
    !finiteScalar(gravityValue.x())
 || !finiteScalar(gravityValue.y())
 || !finiteScalar(gravityValue.z())
)
{
    FatalErrorInFunction
        << "constant/g must contain a finite vector. Current g = "
        << gravityValue << exit(FatalError);
}

volVectorField U
(
    IOobject("U", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE),
    mesh
);

volScalarField p
(
    IOobject("p", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE),
    mesh
);

volScalarField Tgas
(
    IOobject
    (
        "T",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

volScalarField rho
(
    IOobject
    (
        "rho",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    p/(max(Tgas, TgasMinG)*Rgas)
);

volScalarField epsS
(
    IOobject("epsilonS", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE),
    mesh
);

volVectorField Us
(
    IOobject("Us", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE),
    mesh
);

volScalarField theta
(
    IOobject("theta", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE),
    mesh
);

volScalarField nut
(
    IOobject
    (
        "nut",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar
    (
        "zeroNut",
        dimensionSet(0, 2, -1, 0, 0, 0, 0),
        scalar(0)
    ),
    zeroGradientFvPatchScalarField::typeName
);

IOobject kFieldHeader
(
    "k",
    runTime.timeName(),
    mesh,
    IOobject::READ_IF_PRESENT,
    IOobject::NO_WRITE,
    false
);
IOobject omegaFieldHeader
(
    "omega",
    runTime.timeName(),
    mesh,
    IOobject::READ_IF_PRESENT,
    IOobject::NO_WRITE,
    false
);
if
(
    gasTurbulenceModel == 3
 && (!kFieldHeader.headerOk() || !omegaFieldHeader.headerOk())
)
{
    FatalErrorInFunction
        << "kOmegaSST requires readable volScalarField files k and omega "
        << "at time " << runTime.timeName() << exit(FatalError);
}

volScalarField k
(
    IOobject("k", runTime.timeName(), mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE),
    mesh,
    dimensionedScalar("k0", dimensionSet(0, 2, -2, 0, 0, 0, 0), sstKMin),
    zeroGradientFvPatchScalarField::typeName
);

volScalarField omega
(
    IOobject("omega", runTime.timeName(), mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE),
    mesh,
    dimensionedScalar("omega0", dimless/dimTime, sstOmegaMin),
    zeroGradientFvPatchScalarField::typeName
);

volVectorField rhoU
(
    IOobject("rhoU", runTime.timeName(), mesh, IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE),
    rho*U
);

volScalarField rhoE
(
    IOobject("rhoE", runTime.timeName(), mesh, IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE),
    p/(gammaG - 1.0) + rho*(0.5*magSqr(U))
);

volVectorField rhoUs
(
    IOobject("rhoUs", runTime.timeName(), mesh, IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE),
    epsS*rhoS*Us
);

volScalarField rhoEs
(
    IOobject("rhoEs", runTime.timeName(), mesh, IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE),
    epsS*rhoS*(0.5*magSqr(Us) + 1.5*theta)
);

volScalarField rhoDs
(
    IOobject("rhoDs", runTime.timeName(), mesh, IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE),
    epsS*rhoS*dS
);

volScalarField dMeanCell
(
    IOobject("dMeanCell", runTime.timeName(), mesh, IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE),
    rhoDs/(epsS*rhoS + dimensionedScalar("smallRhoForD", dimDensity, SMALL))
);

autoPtr<volScalarField> TpPtr;
if (solveParticleTemperature)
{
    TpPtr.reset
    (
        new volScalarField
        (
            IOobject("Tp", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE),
            mesh
        )
    );
}
else
{
    TpPtr.reset
    (
        new volScalarField
        (
            IOobject("Tp", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            Tgas
        )
    );
}

volScalarField& Tp = TpPtr();

volScalarField rhoHp
(
    IOobject("rhoHp", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
    epsS*rhoS
   *dimensionedScalar
    (
        "initialParticleSpecificEnthalpy",
        dimensionSet(0, 2, -2, 0, 0, 0, 0),
        0.0
    )
);

const dictionary* gpuResidentStuckModelPtr = nullptr;
if (ugkwpProps.found("gpuResidentStuckModel"))
{
    gpuResidentStuckModelPtr =
        &ugkwpProps.subDict("gpuResidentStuckModel");
}
const bool particleStuckWallHeatTransfer =
    gpuResidentStuckModelPtr != nullptr
 && gpuResidentStuckModelPtr->lookupOrDefault<bool>
    (
        "heatTransfer",
        false
    );
const word particleStuckWallHeatFluxName =
    gpuResidentStuckModelPtr == nullptr
  ? word("particleStuckWallHeatFlux")
  : gpuResidentStuckModelPtr->lookupOrDefault<word>
    (
        "heatFluxField",
        word("particleStuckWallHeatFlux")
    );
const word particleReflectedWallHeatFluxName =
    gpuResidentStuckModelPtr == nullptr
  ? word("particleReflectedWallHeatFlux")
  : gpuResidentStuckModelPtr->lookupOrDefault<word>
    (
        "reflectionHeatFluxField",
        word("particleReflectedWallHeatFlux")
    );

autoPtr<surfaceScalarField> particleStuckWallHeatFluxPtr;
autoPtr<surfaceScalarField> particleReflectedWallHeatFluxPtr;
if (particleStuckWallHeatTransfer)
{
    if (particleStuckWallHeatFluxName == particleReflectedWallHeatFluxName)
    {
        FatalErrorInFunction
            << "gpuResidentStuckModel heatFluxField and "
            << "reflectionHeatFluxField must be different."
            << exit(FatalError);
    }
    particleStuckWallHeatFluxPtr.reset
    (
        new surfaceScalarField
        (
            IOobject
            (
                particleStuckWallHeatFluxName,
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mesh,
            dimensionedScalar
            (
                "zeroParticleStuckWallHeatFlux",
                dimensionSet(1, 0, -3, 0, 0, 0, 0),
                scalar(0)
            )
        )
    );
    particleReflectedWallHeatFluxPtr.reset
    (
        new surfaceScalarField
        (
            IOobject
            (
                particleReflectedWallHeatFluxName,
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mesh,
            dimensionedScalar
            (
                "zeroParticleReflectedWallHeatFlux",
                dimensionSet(1, 0, -3, 0, 0, 0, 0),
                scalar(0)
            )
        )
    );
}

#endif
