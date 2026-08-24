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

IOdictionary particlePropertiesFile
(
    IOobject
    (
        "particleProperties",
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
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    )
);

const label fluidSchemaVersion =
    fluidPropertiesFile.lookupOrDefault<label>("schemaVersion", -1);
const label particleSchemaVersion =
    particlePropertiesFile.lookupOrDefault<label>("schemaVersion", -1);
const label schedulingSchemaVersion =
    schedulingPropertiesFile.lookupOrDefault<label>("schemaVersion", -1);
if
(
    fluidSchemaVersion != 1
 || particleSchemaVersion != 1
 || schedulingSchemaVersion != 1
)
{
    FatalErrorInFunction
        << "UGKP unified cases require schemaVersion 1 in "
        << "constant/fluidProperties, constant/particleProperties and "
        << "constant/schedulingProperties."
        << exit(FatalError);
}
if (!fluidPropertiesFile.found("turbulence"))
{
    FatalErrorInFunction
        << "fluidProperties/turbulence is required."
        << exit(FatalError);
}

const dictionary& physicalProperties = fluidPropertiesFile;
const dictionary& momentumTransportProperties =
    fluidPropertiesFile.subDict("turbulence");
const dictionary& ugkwpProps = particlePropertiesFile;
const GpuSchedulingConfiguration gpuScheduling =
    readGpuSchedulingConfiguration(schedulingPropertiesFile);

const dictionary& fvSchemesDict = mesh.schemes().dict();
const dictionary& fvSolutionDict = mesh.solution().dict();
const dictionary* gpuSolutionControls =
    fvSolutionDict.found("UGKP")
  ? &fvSolutionDict.subDict("UGKP")
  : nullptr;

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

    if
    (
        !finiteScalar(molWeight)
     || molWeight <= SMALL
     || !finiteScalar(gasCp)
     || gasCv <= SMALL
    )
    {
        FatalErrorInFunction
            << "physicalProperties requires positive molWeight and Cp > R. "
            << "molWeight=" << molWeight << ", Cp=" << gasCp
            << ", R=" << specificGasConstantValue << exit(FatalError);
    }
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

if
(
    !finiteScalar(gammaG.value())
 || gammaG.value() <= scalar(1) + SMALL
 || gammaG.value() > scalar(5.0/3.0)
)
{
    FatalErrorInFunction
        << "Gas gamma must be in (1, 5/3]. Current gamma = "
        << gammaG.value() << exit(FatalError);
}
if (!finiteScalar(Rgas.value()) || Rgas.value() <= SMALL)
{
    FatalErrorInFunction
        << "Gas R must be positive. Current R = "
        << Rgas.value() << exit(FatalError);
}
if (!finiteScalar(rhoMinG.value()) || rhoMinG.value() <= SMALL)
{
    FatalErrorInFunction
        << "fvSolution/UGKP rhoMin must be positive. Current rhoMin = "
        << rhoMinG.value() << exit(FatalError);
}
if (!finiteScalar(TgasMinG.value()) || TgasMinG.value() <= scalar(0))
{
    FatalErrorInFunction
        << "fvSolution/UGKP TMin must be positive. Current TMin = "
        << TgasMinG.value() << exit(FatalError);
}
if (!finiteScalar(muG.value()) || muG.value() < scalar(0))
{
    FatalErrorInFunction
        << "physicalProperties mu must be non-negative. Current mu = "
        << muG.value() << exit(FatalError);
}
if (!finiteScalar(PrG.value()) || PrG.value() <= SMALL)
{
    FatalErrorInFunction
        << "physicalProperties Pr must be positive. Current Pr = "
        << PrG.value() << exit(FatalError);
}

if (!fvSchemesDict.found("fluxScheme"))
{
    FatalErrorInFunction
        << "system/fvSchemes must define top-level fluxScheme."
        << exit(FatalError);
}
word gasFluxSchemeName("Kurganov");
word gasReconstructionName("firstOrder");
word gasLimiterName("none");
word gasTimeIntegratorName("Euler");
Switch gasRobustFallback(true);

{
    fvSchemesDict.readIfPresent("fluxScheme", gasFluxSchemeName);
    fvSchemesDict.readIfPresent("gasLimiter", gasLimiterName);
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

    if (gpuSolutionControls)
    {
        gasRobustFallback =
            gpuSolutionControls->lookupOrDefault<Switch>
            (
                "robustFallback",
                true
            );
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

if (!gasRobustFallback && gasFluxScheme >= 4)
{
    FatalErrorInFunction
        << "fvSolution/UGKP robustFallback must remain true for "
        << "contact-resolving and AUSM-family fluxes."
        << exit(FatalError);
}

word gasTurbulenceModelName("laminar");
scalar lesDeltaCoeff = scalar(1);
scalar turbulentPrandtl = scalar(0.9);
scalar waleCw = scalar(0.325);
scalar smagorinskyCs = scalar(0.17);
scalar sstAlphaK1 = scalar(0.85);
scalar sstAlphaK2 = scalar(1.0);
scalar sstAlphaOmega1 = scalar(0.5);
scalar sstAlphaOmega2 = scalar(0.856);
scalar sstBeta1 = scalar(0.075);
scalar sstBeta2 = scalar(0.0828);
scalar sstBetaStar = scalar(0.09);
scalar sstGamma1 = scalar(5.0/9.0);
scalar sstGamma2 = scalar(0.44);
scalar sstA1 = scalar(0.31);
scalar sstB1 = scalar(1.0);
scalar sstC1 = scalar(10.0);
scalar sstKMin = scalar(1.0e-12);
scalar sstOmegaMin = scalar(1.0e-6);
scalar sstMaxSourceNumber = scalar(0.25);
word sstWallTreatmentName("lowRe");
label sstWallTreatment = 0;
scalar sstWallKappa = scalar(0.41);
scalar sstWallE = scalar(9.8);
scalar sstWallCmu = scalar(0.09);

if (!momentumTransportProperties.empty())
{
    const word simulationType
    (
        momentumTransportProperties.lookupOrDefault<word>
        (
            "simulationType",
            "laminar"
        )
    );
    if (simulationType == "laminar")
    {
        gasTurbulenceModelName = "laminar";
    }
    else if (simulationType == "LES")
    {
        const dictionary& les =
            momentumTransportProperties.subDict("LES");
        gasTurbulenceModelName = word(les.lookup("model"));
        const dictionary deltaCoeffs =
            les.subOrEmptyDict("cubeRootVolCoeffs");
        lesDeltaCoeff =
            deltaCoeffs.lookupOrDefault<scalar>("deltaCoeff", scalar(1));
        const dictionary modelCoeffs =
            les.subOrEmptyDict(gasTurbulenceModelName + "Coeffs");
        waleCw = modelCoeffs.lookupOrDefault<scalar>("Cw", waleCw);
        smagorinskyCs =
            modelCoeffs.lookupOrDefault<scalar>("Cs", smagorinskyCs);
    }
    else if (simulationType == "RAS")
    {
        const dictionary& ras =
            momentumTransportProperties.subDict("RAS");
        const Switch turbulenceEnabled =
            ras.lookupOrDefault<Switch>("turbulence", Switch(true));
        gasTurbulenceModelName =
            turbulenceEnabled ? word(ras.lookup("model")) : word("laminar");
        turbulentPrandtl = ras.lookupOrDefault<scalar>
        (
            "turbulentPrandtl",
            turbulentPrandtl
        );

        if (turbulenceEnabled && gasTurbulenceModelName == "kOmegaSST")
        {
            const dictionary modelCoeffs =
                ras.subOrEmptyDict("kOmegaSSTCoeffs");
            sstAlphaK1 = modelCoeffs.lookupOrDefault<scalar>("alphaK1", scalar(0.85));
            sstAlphaK2 = modelCoeffs.lookupOrDefault<scalar>("alphaK2", scalar(1.0));
            sstAlphaOmega1 = modelCoeffs.lookupOrDefault<scalar>("alphaOmega1", scalar(0.5));
            sstAlphaOmega2 = modelCoeffs.lookupOrDefault<scalar>("alphaOmega2", scalar(0.856));
            sstBeta1 = modelCoeffs.lookupOrDefault<scalar>("beta1", scalar(0.075));
            sstBeta2 = modelCoeffs.lookupOrDefault<scalar>("beta2", scalar(0.0828));
            sstBetaStar = modelCoeffs.lookupOrDefault<scalar>("betaStar", scalar(0.09));
            sstGamma1 = modelCoeffs.lookupOrDefault<scalar>("gamma1", scalar(5.0/9.0));
            sstGamma2 = modelCoeffs.lookupOrDefault<scalar>("gamma2", scalar(0.44));
            sstA1 = modelCoeffs.lookupOrDefault<scalar>("a1", scalar(0.31));
            sstB1 = modelCoeffs.lookupOrDefault<scalar>("b1", scalar(1.0));
            sstC1 = modelCoeffs.lookupOrDefault<scalar>("c1", scalar(10.0));
            sstKMin = modelCoeffs.lookupOrDefault<scalar>("kMin", scalar(1.0e-12));
            sstOmegaMin = modelCoeffs.lookupOrDefault<scalar>("omegaMin", scalar(1.0e-6));
            sstMaxSourceNumber = modelCoeffs.lookupOrDefault<scalar>("maxSourceNumber", scalar(0.25));
            sstWallTreatmentName =
                modelCoeffs.lookupOrDefault<word>("wallTreatment", "lowRe");
            sstWallKappa =
                modelCoeffs.lookupOrDefault<scalar>("wallKappa", scalar(0.41));
            sstWallE =
                modelCoeffs.lookupOrDefault<scalar>("wallE", scalar(9.8));
            sstWallCmu =
                modelCoeffs.lookupOrDefault<scalar>("wallCmu", scalar(0.09));
            const Switch f3 =
                modelCoeffs.lookupOrDefault<Switch>("F3", Switch(false));
            if (f3)
            {
                FatalErrorInFunction
                    << "UGKP implements the smooth-wall OpenFOAM-10 "
                    << "kOmegaSST model and therefore requires F3=false."
                    << exit(FatalError);
            }
        }
    }
    else
    {
        FatalErrorInFunction
            << "momentumTransport simulationType '" << simulationType
            << "' is unsupported. Valid values are laminar, LES and RAS."
            << exit(FatalError);
    }
}
const scalar maxDiffusionNumber =
    gpuSolutionControls
  ? gpuSolutionControls->lookupOrDefault<scalar>
    (
        "maxDiffusionNumber",
        scalar(0.25)
    )
  : scalar(0.25);

label gasTurbulenceModel = -1;
if (gasTurbulenceModelName == "laminar")
{
    gasTurbulenceModel = 0;
}
else if (gasTurbulenceModelName == "WALE")
{
    gasTurbulenceModel = 1;
}
else if (gasTurbulenceModelName == "Smagorinsky")
{
    gasTurbulenceModel = 2;
}
else if (gasTurbulenceModelName == "kOmegaSST")
{
    gasTurbulenceModel = 3;
    if (sstWallTreatmentName == "lowRe")
    {
        sstWallTreatment = 0;
    }
    else if (sstWallTreatmentName == "wallFunction")
    {
        sstWallTreatment = 1;
    }
    else
    {
        FatalErrorInFunction
            << "Unsupported kOmegaSST wallTreatment '"
            << sstWallTreatmentName
            << "'. Valid values are lowRe and wallFunction."
            << exit(FatalError);
    }
}
else
{
    FatalErrorInFunction
        << "Unsupported momentumTransport model '"
        << gasTurbulenceModelName
        << "'. Valid values are laminar, WALE, Smagorinsky and kOmegaSST."
        << exit(FatalError);
}

if (!finiteScalar(lesDeltaCoeff) || lesDeltaCoeff <= SMALL)
{
    FatalErrorInFunction
        << "LES deltaCoeff must be finite and positive."
        << exit(FatalError);
}
if (!finiteScalar(turbulentPrandtl) || turbulentPrandtl <= SMALL)
{
    FatalErrorInFunction
        << "Turbulent Prandtl number must be finite and positive."
        << exit(FatalError);
}
if (!finiteScalar(waleCw) || waleCw < scalar(0))
{
    FatalErrorInFunction
        << "WALE Cw must be finite and non-negative."
        << exit(FatalError);
}
if (!finiteScalar(smagorinskyCs) || smagorinskyCs < scalar(0))
{
    FatalErrorInFunction
        << "Smagorinsky Cs must be finite and non-negative."
        << exit(FatalError);
}
if (!finiteScalar(maxDiffusionNumber) || maxDiffusionNumber <= SMALL)
{
    FatalErrorInFunction
        << "fvSolution/UGKP maxDiffusionNumber must be positive."
        << exit(FatalError);
}
if
(
    !finiteScalar(sstAlphaK1) || sstAlphaK1 <= SMALL
 || !finiteScalar(sstAlphaK2) || sstAlphaK2 <= SMALL
 || !finiteScalar(sstAlphaOmega1) || sstAlphaOmega1 <= SMALL
 || !finiteScalar(sstAlphaOmega2) || sstAlphaOmega2 <= SMALL
 || !finiteScalar(sstBeta1) || sstBeta1 <= SMALL
 || !finiteScalar(sstBeta2) || sstBeta2 <= SMALL
 || !finiteScalar(sstBetaStar) || sstBetaStar <= SMALL
 || !finiteScalar(sstGamma1) || sstGamma1 <= SMALL
 || !finiteScalar(sstGamma2) || sstGamma2 <= SMALL
 || !finiteScalar(sstA1) || sstA1 <= SMALL
 || !finiteScalar(sstB1) || sstB1 <= SMALL
 || !finiteScalar(sstC1) || sstC1 <= SMALL
 || !finiteScalar(sstKMin) || sstKMin <= scalar(0)
 || !finiteScalar(sstOmegaMin) || sstOmegaMin <= scalar(0)
 || !finiteScalar(sstMaxSourceNumber) || sstMaxSourceNumber <= SMALL
 || !finiteScalar(sstWallKappa) || sstWallKappa <= SMALL
 || !finiteScalar(sstWallE) || sstWallE <= scalar(1)
 || !finiteScalar(sstWallCmu) || sstWallCmu <= SMALL
)
{
    FatalErrorInFunction
        << "kOmegaSST coefficients, wall constants, floors and "
        << "maxSourceNumber must be "
        << "finite and positive." << exit(FatalError);
}

#endif
