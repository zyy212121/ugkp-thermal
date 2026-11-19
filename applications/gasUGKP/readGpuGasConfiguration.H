#ifndef readGpuGasConfiguration_H
#define readGpuGasConfiguration_H

#include "GpuSchedulingConfiguration.H"

SharedGasModelInput sharedGasModel(runTime, mesh);

                       
                                                               
                                                                           
                                                                              
                                                                       
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

MolecularTransportInput molecularTransport(runTime,mesh,sharedGasModel.model());

scalar gammaGasValue = 0;
scalar specificGasConstantValue = 0;
scalar gasMuValue = 0;
scalar gasPrValue = 0;

if (!sharedGasModel.active())
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
else
{



    gammaGasValue = 1.4;
    specificGasConstantValue = 1.0;
    const dictionary& transport = physicalProperties.subDict("mixture").subDict("transport");
    gasMuValue = readScalar(transport.lookup("mu"));
    gasPrValue = readScalar(transport.lookup("Pr"));
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

const GasNumericsConfiguration gasNumerics =
    readGasNumericsConfiguration(fvSchemesDict);
const word gasFluxSchemeName = gasNumerics.gasFluxSchemeName;
const word gasReconstructionName = gasNumerics.gasReconstructionName;
const word gasLimiterName = gasNumerics.gasLimiterName;
const word gasTimeIntegratorName = gasNumerics.gasTimeIntegratorName;
const label gasFluxScheme = gasNumerics.gasFluxScheme;
const label gasReconstruction = gasNumerics.gasReconstruction;
const label gasLimiter = gasNumerics.gasLimiter;
const label gasTimeIntegrator = gasNumerics.gasTimeIntegrator;

#include "FoamGasTurbulenceControls.H"

if (momentumTransportProperties.lookupOrDefault<word>("simulationType","laminar")=="laminar")
{
    sstWallTreatmentName=momentumTransportProperties.lookupOrDefault<word>("wallTreatment","lowRe");
    if (sstWallTreatmentName=="boundaryLayer") sstWallTreatment=2;
    else if (sstWallTreatmentName!="lowRe")
        FatalErrorInFunction << "Laminar wallTreatment supports lowRe or boundaryLayer" << exit(FatalError);
}

BoundaryLayerInput boundaryLayerInput
(
    momentumTransportProperties, sharedGasModel.model(), gasTurbulenceModel,
    gpuScheduling.pureGasOnly, turbulentPrandtl
);
boundaryLayerInput.configureSolverControls(runTime,mesh.name());
if (boundaryLayerInput.active())
{
    sstWallTreatment = 2;
    sstWallTreatmentName = "boundaryLayer";
}

sharedGasModel.validateNumerics
(
    gasFluxScheme, gasReconstruction, gasLimiter, gasTimeIntegrator,
    gasTurbulenceModel,
    !gpuScheduling.pureGasOnly,
    sstWallTreatment,
    boundaryLayerInput.model(),
    true
);

#endif
