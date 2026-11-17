


























#ifndef UGKP_FOAM_THERMAL_NATIVE_SCHEMES_H
#define UGKP_FOAM_THERMAL_NATIVE_SCHEMES_H
#include "linearUpwind.H"
#include "limitedSnGrad.H"
#include "OStringStream.H"
#include "../FoamThermalDiscretization.H"
namespace ugkp { namespace gpu {
class FoamThermalLinearUpwind : public Foam::upwind<Foam::scalar> {
    Foam::word gradSchemeName_;
public:
    FoamThermalLinearUpwind(const Foam::fvMesh& mesh,Foam::Istream& input)
    :Foam::upwind<Foam::scalar>(mesh,input),gradSchemeName_(input){}
    FoamThermalLinearUpwind(const Foam::fvMesh& mesh,const Foam::surfaceScalarField& flux,Foam::Istream& input)
    :Foam::upwind<Foam::scalar>(mesh,flux,input),gradSchemeName_(input){}
    const Foam::word& type() const override {return Foam::linearUpwind<Foam::scalar>::typeName;}
    bool corrected() const override {return true;}
    Foam::tmp<Foam::surfaceScalarField> correction(const Foam::volScalarField& vf) const override {
        using namespace Foam;
    const fvMesh& mesh = this->mesh();

    tmp<GeometricField<scalar, fvsPatchField, surfaceMesh>> tsfCorr
    (
        GeometricField<scalar, fvsPatchField, surfaceMesh>::New
        (
            "linearUpwind::correction(" + vf.name() + ')',
            mesh,
            dimensioned<scalar>(vf.name(), vf.dimensions(), Zero)
        )
    );

    GeometricField<scalar, fvsPatchField, surfaceMesh>& sfCorr = tsfCorr.ref();

    const surfaceScalarField& faceFlux = this->faceFlux_;

    const labelList& owner = mesh.owner();
    const labelList& neighbour = mesh.neighbour();

    const volVectorField& C = mesh.C();
    const surfaceVectorField& Cf = mesh.Cf();

    tmp<fv::gradScheme<scalar>> gradScheme_
    (
        fv::gradScheme<scalar>::New
        (
            mesh,
            ugkp::solverControls::thermalDiscretization(mesh).selection("gradSchemes",gradSchemeName_)
        )
    );

    for (direction cmpt = 0; cmpt < pTraits<scalar>::nComponents; cmpt++)
    {
        tmp<volVectorField> tgradVf =
            gradScheme_().grad(vf.component(cmpt), gradSchemeName_);

        const volVectorField& gradVf = tgradVf();

        forAll(faceFlux, facei)
        {
            const label celli =
                (faceFlux[facei] > 0) ? owner[facei] : neighbour[facei];

            setComponent(sfCorr[facei], cmpt) =
                (Cf[facei] - C[celli]) & gradVf[celli];
        }

        typename GeometricField<scalar, fvsPatchField, surfaceMesh>::
            Boundary& bSfCorr = sfCorr.boundaryFieldRef();

        forAll(bSfCorr, patchi)
        {
            fvsPatchField<scalar>& pSfCorr = bSfCorr[patchi];

            if (pSfCorr.coupled())
            {
                const labelUList& pOwner = mesh.boundary()[patchi].faceCells();
                const vectorField& pCf = Cf.boundaryField()[patchi];
                const scalarField& pFaceFlux = faceFlux.boundaryField()[patchi];

                const vectorField pGradVfNei
                (
                    gradVf.boundaryField()[patchi].patchNeighbourField()
                );


                const vectorField pd
                (
                    Cf.boundaryField()[patchi].patch().delta()
                );

                forAll(pOwner, facei)
                {
                    label own = pOwner[facei];

                    if (pFaceFlux[facei] > 0)
                    {
                        setComponent(pSfCorr[facei], cmpt) =
                            (pCf[facei] - C[own])
                          & gradVf[own];
                    }
                    else
                    {
                        setComponent(pSfCorr[facei], cmpt) =
                            (pCf[facei] - pd[facei] - C[own])
                          & pGradVfNei[facei];
                    }
                }
            }
        }
    }

    return tsfCorr;
    }
};
inline Foam::tmp<Foam::fv::snGradScheme<Foam::scalar>> thermalLimitedSnGrad(
    const Foam::fvMesh& mesh,Foam::Istream& input) {
    Foam::token first(input);input.putBack(first);
    if(first.isNumber()) {
        Foam::scalar coefficient;input>>coefficient;
        Foam::OStringStream expanded;expanded.precision(17);
        expanded<<"corrected "<<coefficient;
        Foam::IStringStream explicitForm(expanded.str());
        return Foam::tmp<Foam::fv::snGradScheme<Foam::scalar>>(
            new Foam::fv::limitedSnGrad<Foam::scalar>(mesh,explicitForm));
    }
    return Foam::tmp<Foam::fv::snGradScheme<Foam::scalar>>(
        new Foam::fv::limitedSnGrad<Foam::scalar>(mesh,input));
}
}}
#endif
