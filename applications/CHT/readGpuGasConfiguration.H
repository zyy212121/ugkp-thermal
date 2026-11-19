#ifndef readGpuGasConfiguration_H
#define readGpuGasConfiguration_H

#include "GpuSchedulingConfiguration.H"

                                                               
                                                                           
                                                                              
                                                                       
IOdictionary fluidPropertiesFile
(
    IOobject
    (
        "fluidProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    )
);

IOdictionary schedulingPropertiesFile
(
    IOobject
    (
        "schedulingProperties",
        runTime.constant(),
        runTime,
        IOobject::MUST_READ,
        IOobject::NO_WRITE,
        false
    )
);

const dictionary& physicalProperties = fluidPropertiesFile;
const dictionary momentumTransportProperties =
    ugkp::solverControls::readMomentumTransport(runTime,mesh,physicalProperties);
const dictionary& ugkwpProps = particlePropertiesFile;
const GpuSchedulingConfiguration gpuScheduling =
    readGpuSchedulingConfiguration(schedulingPropertiesFile,
        ugkp::solverControls::readParticleTrackingMaxFaceWalkHops(runTime,mesh.name()));

const dictionary fvSchemesDict = ugkp::solverControls::readGasNumericsDictionary(runTime,mesh.name());
const dictionary& fvSolutionDict = mesh.solution().dict();
const dictionary* gpuSolutionControls =
    fvSolutionDict.found("UGKP")
  ? &fvSolutionDict.subDict("UGKP")
  : nullptr;

SharedGasModelInput sharedGasModel(runTime,mesh);

if (sharedGasModel.model().mode != ugkwp::GasMode::SingleLegacy)
{
    FatalErrorInFunction
        << "CHT supports single-component gas, with or without particles. "
        << "Use CHMT for frozen-mixture material coupling."
        << exit(FatalError);
}

MolecularTransportInput molecularTransport(runTime,mesh,sharedGasModel.model());

scalar gammaGasValue = 0;
scalar specificGasConstantValue = 0;
scalar gasMuValue = 0;
scalar gasPrValue = 0;

{
    const dictionary& thermoType =
        physicalProperties.subDict("thermoType");
    const word equationOfState(thermoType.lookup("equationOfState"));
    const word thermoModel(thermoType.lookup("thermo"));
    const word transportModel(thermoType.lookup("transport"));
    const word energyVariable(thermoType.lookup("energy"));
    if
    (
        equationOfState != "perfectGas"
     || thermoModel != "hConst"
     || transportModel != "const"
     || energyVariable != "sensibleInternalEnergy"
    )
    {
        FatalErrorInFunction
            << "UGKP currently supports the OpenFOAM thermo combination "
            << "equationOfState=perfectGas, thermo=hConst, transport=const, "
            << "energy=sensibleInternalEnergy. Current thermoType is "
            << thermoType << exit(FatalError);
    }

    const dictionary& mixture = physicalProperties.subDict("mixture");
    const dictionary& specie = mixture.subDict("specie");
    const dictionary& thermodynamics =
        mixture.subDict("thermodynamics");
    const dictionary& transport = mixture.subDict("transport");
    const scalar molWeight = readScalar(specie.lookup("molWeight"));
    const scalar gasCp = readScalar(thermodynamics.lookup("Cp"));
    gasMuValue = readScalar(transport.lookup("mu"));
    gasPrValue = readScalar(transport.lookup("Pr"));
    specificGasConstantValue =
        constant::physicoChemical::RR.value()/molWeight;
    const scalar gasCv = gasCp - specificGasConstantValue;
    gammaGasValue = gasCp/gasCv;

}

const scalar rhoMinValue =
    gpuSolutionControls
  ? gpuSolutionControls->lookupOrDefault<scalar>("rhoMin", 1.0e-12)
  : scalar(1.0e-12);
const scalar gasTMinValue =
    gpuSolutionControls
  ? gpuSolutionControls->lookupOrDefault<scalar>("TMin", SMALL)
  : ugkwpProps.lookupOrDefault<scalar>("TgasMin", SMALL);

dimensionedScalar gammaG("gamma", dimless, gammaGasValue);
dimensionedScalar Rgas
(
    "R",
    dimensionSet(0, 2, -2, -1, 0, 0, 0),
    specificGasConstantValue
);
dimensionedScalar muG
(
    "mu",
    dimensionSet(1, -1, -1, 0, 0, 0, 0),
    gasMuValue
);
dimensionedScalar PrG("Pr", dimless, gasPrValue);
dimensionedScalar rhoMinG
(
    "rhoMin",
    dimensionSet(1, -3, 0, 0, 0, 0, 0),
    rhoMinValue
);
dimensionedScalar TgasMinG("TMin", dimTemperature, gasTMinValue);

if (!fvSchemesDict.found("fluxScheme"))
{
    FatalErrorInFunction
        << "constant/scheme gasNumerics must define fluxScheme."
        << exit(FatalError);
}
const bool strictOpenFoamGasBoundaries = true;
word gasFluxSchemeName("Kurganov");
word gasReconstructionName("firstOrder");
word gasLimiterName("none");
word gasTimeIntegratorName("Euler");

{
    fvSchemesDict.readIfPresent("fluxScheme", gasFluxSchemeName);
    const dictionary& ddtSchemes =
        fvSchemesDict.subDict("ddtSchemes");
    ITstream ddtScheme(ddtSchemes.lookup("default"));
    ddtScheme >> gasTimeIntegratorName;
    if (ddtScheme.nRemainingTokens() != 0)
    {
        FatalErrorInFunction
            << "ddtSchemes/default must contain one explicit time scheme "
            << "(Euler, SSPRK2 or SSPRK3)."
            << exit(FatalError);
    }

    const dictionary& gradSchemes =
        fvSchemesDict.subDict("gradSchemes");
    ITstream gradDefault(gradSchemes.lookup("default"));
    word gradIntegration;
    word gradInterpolation;
    gradDefault >> gradIntegration >> gradInterpolation;
    if
    (
        gradIntegration != "Gauss"
     || gradInterpolation != "linear"
     || gradDefault.nRemainingTokens() != 0
    )
    {
        FatalErrorInFunction
            << "UGKP currently requires gradSchemes/default "
            << "'Gauss linear' because that is the resident Green-Gauss "
            << "operator implemented on the GPU."
            << exit(FatalError);
    }

    const dictionary& interpolationSchemes =
        fvSchemesDict.subDict("interpolationSchemes");
    ITstream interpolationDefault
    (
        interpolationSchemes.lookup("default")
    );
    word defaultInterpolation;
    interpolationDefault >> defaultInterpolation;
    if
    (
        defaultInterpolation != "linear"
     || interpolationDefault.nRemainingTokens() != 0
    )
    {
        FatalErrorInFunction
            << "UGKP currently requires "
            << "interpolationSchemes/default 'linear'."
            << exit(FatalError);
    }

    const dictionary& snGradSchemes =
        fvSchemesDict.subDict("snGradSchemes");
    ITstream snGradDefault(snGradSchemes.lookup("default"));
    word defaultSnGrad;
    snGradDefault >> defaultSnGrad;
    if
    (
        defaultSnGrad != "corrected"
     || snGradDefault.nRemainingTokens() != 0
    )
    {
        FatalErrorInFunction
            << "UGKP currently requires snGradSchemes/default "
            << "'corrected'."
            << exit(FatalError);
    }

    const dictionary& laplacianSchemes =
        fvSchemesDict.subDict("laplacianSchemes");
    ITstream laplacianDefault(laplacianSchemes.lookup("default"));
    word laplacianIntegration;
    word laplacianInterpolation;
    word laplacianCorrection;
    laplacianDefault
        >> laplacianIntegration
        >> laplacianInterpolation
        >> laplacianCorrection;
    if
    (
        laplacianIntegration != "Gauss"
     || laplacianInterpolation != "linear"
     || laplacianCorrection != "corrected"
     || laplacianDefault.nRemainingTokens() != 0
    )
    {
        FatalErrorInFunction
            << "UGKP currently requires laplacianSchemes/default "
            << "'Gauss linear corrected'."
            << exit(FatalError);
    }

    const dictionary& divSchemes =
        fvSchemesDict.subDict("divSchemes");
    auto readDivInterpolation =
        [&divSchemes]
        (
            const word& entryName,
            scalar& coefficient
        ) -> word
        {
            ITstream scheme(divSchemes.lookup(entryName));
            word integration;
            word interpolation;
            scheme >> integration >> interpolation;
            if (integration != "Gauss")
            {
                FatalErrorInFunction
                    << entryName << " must use Gauss integration; found "
                    << integration << exit(FatalError);
            }
            coefficient = scalar(1);
            if
            (
                interpolation == "limitedLinear"
             && scheme.nRemainingTokens() > 0
            )
            {
                scheme >> coefficient;
            }
            if (scheme.nRemainingTokens() != 0)
            {
                FatalErrorInFunction
                    << entryName << " contains unsupported trailing "
                    << "scheme coefficients." << exit(FatalError);
            }
            return interpolation;
        };

    scalar uCoefficient = 1;
    scalar eCoefficient = 1;
    scalar kCoefficient = 1;
    scalar pByRhoCoefficient = 1;
    const word divU =
        readDivInterpolation("div(phi,U)", uCoefficient);
    const word divE =
        readDivInterpolation("div(phi,e)", eCoefficient);
    const word divK =
        readDivInterpolation("div(phi,K)", kCoefficient);
    const word divPByRho =
        readDivInterpolation
        (
            "div(phi,(p|rho))",
            pByRhoCoefficient
        );
    scalar viscousCoefficient = 1;
    const word divViscous =
        readDivInterpolation
        (
            "div(((rho*nuEff)*dev2(T(grad(U)))))",
            viscousCoefficient
        );
    if (divViscous != "linear")
    {
        FatalErrorInFunction
            << "UGKP currently requires "
            << "div(((rho*nuEff)*dev2(T(grad(U))))) 'Gauss linear'."
            << exit(FatalError);
    }

    const bool momentumUpwind = divU == "upwind";
    const bool momentumMUSCL = divU == "MUSCL";
    const bool energyUpwind =
        divE == "upwind"
     && divK == "upwind"
     && divPByRho == "upwind";
    const bool energyLimitedLinear =
        divE == "limitedLinear"
     && divK == "limitedLinear"
     && divPByRho == "limitedLinear"
     && mag(eCoefficient - scalar(1)) <= SMALL
     && mag(kCoefficient - scalar(1)) <= SMALL
     && mag(pByRhoCoefficient - scalar(1)) <= SMALL;
    const bool energyMUSCL =
        divE == "MUSCL"
     && divK == "MUSCL"
     && divPByRho == "MUSCL";

    if (momentumUpwind && energyUpwind)
    {
        gasReconstructionName = "firstOrder";
    }
    else if (momentumUpwind && energyLimitedLinear)
    {
        gasReconstructionName = "energyLimitedLinear";
    }
    else if (momentumMUSCL && energyMUSCL)
    {
        gasReconstructionName = "MUSCL";
    }
    else
    {
        FatalErrorInFunction
            << "Unsupported OpenFOAM divSchemes combination. The verified "
            << "GPU path currently accepts one of: (i) Gauss upwind for "
            << "div(phi,U) with Gauss upwind for div(phi,e), div(phi,K), "
            << "and div(phi,(p|rho)); (ii) Gauss upwind for div(phi,U) "
            << "with Gauss limitedLinear 1 for div(phi,e), div(phi,K), "
            << "and div(phi,(p|rho)); (iii) Gauss MUSCL for all of "
            << "div(phi,U), div(phi,e), div(phi,K), and div(phi,(p|rho))."
            << exit(FatalError);
    }

}

label gasFluxScheme = -1;
if
(
    gasFluxSchemeName == "Tadmor"
 || gasFluxSchemeName == "rusanovTadmor"
 || gasFluxSchemeName == "rusanov"
 || gasFluxSchemeName == "tadmor"
)
{
    gasFluxScheme = 1;
}
else if
(
    gasFluxSchemeName == "Kurganov"
 || gasFluxSchemeName == "hllKurganov"
 || gasFluxSchemeName == "hll"
 || gasFluxSchemeName == "kurganov"
)
{
    gasFluxScheme = 2;
}
else if (gasFluxSchemeName == "HLLE" || gasFluxSchemeName == "hlle")
{
    gasFluxScheme = 3;
}
else if (gasFluxSchemeName == "HLLC" || gasFluxSchemeName == "hllc")
{
    gasFluxScheme = 4;
}
else if (gasFluxSchemeName == "Roe" || gasFluxSchemeName == "roe")
{
    gasFluxScheme = 5;
}
else if (gasFluxSchemeName == "HLLEM" || gasFluxSchemeName == "hllem")
{
    gasFluxScheme = 6;
}
else if
(
    gasFluxSchemeName == "HLLC-ADC"
 || gasFluxSchemeName == "hllcAdc"
 || gasFluxSchemeName == "hllcADC"
)
{
    gasFluxScheme = 7;
}
else if (gasFluxSchemeName == "SLAU2" || gasFluxSchemeName == "slau2")
{
    gasFluxScheme = 8;
}
else if
(
    gasFluxSchemeName == "SLAU2.2"
 || gasFluxSchemeName == "slau2_2"
 || gasFluxSchemeName == "slau22"
)
{
    gasFluxScheme = 9;
}
else
{
    FatalErrorInFunction
        << "Unsupported fvSchemes fluxScheme '" << gasFluxSchemeName
        << "'. Valid values are Tadmor, Kurganov, HLLE, HLLC, Roe, "
        << "HLLEM, HLLC-ADC, SLAU2 and SLAU2.2."
        << exit(FatalError);
}

label gasReconstruction = -1;
if (gasReconstructionName == "firstOrder")
{
    gasReconstruction = 0;
}
else if (gasReconstructionName == "MUSCL")
{
    gasReconstruction = 1;
}
else if (gasReconstructionName == "energyLimitedLinear")
{
    gasReconstruction = 2;
}
else
{
    FatalErrorInFunction
        << "Unsupported gas reconstruction '" << gasReconstructionName
        << "'." << exit(FatalError);
}
label gasLimiter = -1;
if (gasLimiterName == "none")
{
    gasLimiter = 0;
}
else if (gasLimiterName == "barthJespersen")
{
    gasLimiter = 1;
}
else if (gasLimiterName == "venkatakrishnan")
{
    gasLimiter = 2;
}
else
{
    FatalErrorInFunction
        << "Unsupported gas limiter '" << gasLimiterName << "'."
        << exit(FatalError);
}

label gasTimeIntegrator = -1;
if (gasTimeIntegratorName == "Euler")
{
    gasTimeIntegrator = 1;
}
else if (gasTimeIntegratorName == "SSPRK2")
{
    gasTimeIntegrator = 2;
}
else if (gasTimeIntegratorName == "SSPRK3")
{
    gasTimeIntegrator = 3;
}
else
{
    FatalErrorInFunction
        << "Unsupported ddtSchemes/default '" << gasTimeIntegratorName
        << "'. Valid values for this explicit GPU solver are Euler, "
        << "SSPRK2 and SSPRK3." << exit(FatalError);
}

                                                                           

#include "FoamGasTurbulenceControls.H"

BoundaryLayerInput boundaryLayerInput(momentumTransportProperties,sharedGasModel.model(),gasTurbulenceModel,
    gpuScheduling.pureGasOnly,turbulentPrandtl);
boundaryLayerInput.configureSolverControls(runTime,mesh.name());
if(boundaryLayerInput.active()){sstWallTreatment=2;sstWallTreatmentName="boundaryLayer";}
sharedGasModel.validateNumerics(gasFluxScheme,gasReconstruction,gasLimiter,gasTimeIntegrator,
    gasTurbulenceModel,!gpuScheduling.pureGasOnly,sstWallTreatment);
#endif
