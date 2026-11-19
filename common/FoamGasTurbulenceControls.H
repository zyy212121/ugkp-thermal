word gasTurbulenceModelName("laminar");
scalar lesDeltaCoeff = scalar(1);
scalar turbulentPrandtl = scalar(0.9);
scalar waleCw = scalar(0.325);
scalar smagorinskyCs = scalar(0.17);
ugkp::solverControls::FoamSstCoefficients sstCoefficients;
scalar sstB1 = scalar(1.0);
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
        gasTurbulenceModelName=ugkp::solverControls::rasModel(ras);
        turbulentPrandtl = ras.lookupOrDefault<scalar>
        (
            "turbulentPrandtl",
            turbulentPrandtl
        );

        if (gasTurbulenceModelName == "kOmegaSST")
        {
            const dictionary modelCoeffs =
                ras.subOrEmptyDict("kOmegaSSTCoeffs");
            ugkp::solverControls::readSstCoefficients(modelCoeffs,sstCoefficients);
            sstB1 = modelCoeffs.lookupOrDefault<scalar>("b1", scalar(1.0));
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
    else if(sstWallTreatmentName=="boundaryLayer")sstWallTreatment=2;
    else
    {
        FatalErrorInFunction
            << "Unsupported kOmegaSST wallTreatment '"
            << sstWallTreatmentName
            << "'. Valid values are lowRe, wallFunction and boundaryLayer."
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

const scalar sstKMin=sstCoefficients.minimumK;
const scalar sstOmegaMin=sstCoefficients.minimumOmega;
