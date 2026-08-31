#ifndef GpuDragModel_H
#define GpuDragModel_H

#include <cmath>

#include "FixedList.H"
#include "autoPtr.H"
#include "dictionary.H"
#include "error.H"
#include "label.H"
#include "scalar.H"
#include "word.H"

namespace Foam
{

class GpuDragModel
{
public:
    virtual ~GpuDragModel() = default;

    virtual label modelId() const = 0;
    virtual const word& modelName() const = 0;
    virtual const FixedList<scalar, 4>& parameters() const = 0;

    static autoPtr<GpuDragModel> New(const dictionary& dict);
};


class GpuSchillerNaumannDragModel final : public GpuDragModel
{
    word modelName_;
    FixedList<scalar, 4> parameters_;

public:
    GpuSchillerNaumannDragModel()
    :
        modelName_("SchillerNaumann"),
        parameters_()
    {
        parameters_ = scalar(0);
    }

    label modelId() const override
    {
        return 0;
    }

    const word& modelName() const override
    {
        return modelName_;
    }

    const FixedList<scalar, 4>& parameters() const override
    {
        return parameters_;
    }
};


class GpuGidaspowErgunWenYuDragModel final : public GpuDragModel
{
    word modelName_;
    FixedList<scalar, 4> parameters_;

public:
    explicit GpuGidaspowErgunWenYuDragModel(const dictionary& dict)
    :
        modelName_("GidaspowErgunWenYu"),
        parameters_()
    {
        parameters_ = scalar(0);
        const dictionary& coeffs =
            dict.subDict("GidaspowErgunWenYuCoeffs");
        const scalar residualRe = readScalar(coeffs.lookup("residualRe"));
        if (!std::isfinite(residualRe) || residualRe <= 0)
        {
            FatalErrorInFunction
                << "GidaspowErgunWenYu residualRe must be finite and positive, "
                << "received " << residualRe << exit(FatalError);
        }
        parameters_[0] = residualRe;
    }

    label modelId() const override
    {
        return 1;
    }

    const word& modelName() const override
    {
        return modelName_;
    }

    const FixedList<scalar, 4>& parameters() const override
    {
        return parameters_;
    }
};


                                                                          
                                                                          
                                                                          
class GpuConstantResponseTimeDragModel final : public GpuDragModel
{
    word modelName_;
    FixedList<scalar, 4> parameters_;

public:
    explicit GpuConstantResponseTimeDragModel(const dictionary& dict)
    :
        modelName_("constantResponseTime"),
        parameters_()
    {
        parameters_ = scalar(0);
        const scalar particleResponseTime = readScalar
        (
            dict.lookup("gpuResidentParticleResponseTime")
        );
        const word heatTransferModel
        (
            dict.lookupOrDefault<word>
            (
                "gpuResidentHeatTransferModel",
                "RanzMarshall"
            )
        );
        const scalar particleThermalResponseTime = readScalar
        (
            dict.lookup("gpuResidentParticleThermalResponseTime")
        );
        const scalar gasHeatCapacity = readScalar
        (
            dict.lookup("gpuResidentGasHeatCapacity")
        );

        if (!std::isfinite(particleResponseTime) || particleResponseTime <= 0)
        {
            FatalErrorInFunction
                << "gpuResidentParticleResponseTime must be finite and "
                << "positive, received " << particleResponseTime
                << exit(FatalError);
        }
        if (heatTransferModel != "constantResponseTime")
        {
            FatalErrorInFunction
                << "The constant-response-time validation model "
                << "requires gpuResidentHeatTransferModel "
                << "constantResponseTime, received " << heatTransferModel
                << exit(FatalError);
        }
        if
        (
            !std::isfinite(particleThermalResponseTime)
         || particleThermalResponseTime <= 0
        )
        {
            FatalErrorInFunction
                << "gpuResidentParticleThermalResponseTime must be finite "
                << "and positive, received " << particleThermalResponseTime
                << exit(FatalError);
        }
        if (!std::isfinite(gasHeatCapacity) || gasHeatCapacity <= 0)
        {
            FatalErrorInFunction
                << "gpuResidentGasHeatCapacity must be finite and positive, "
                << "received " << gasHeatCapacity << exit(FatalError);
        }

        parameters_[0] = particleResponseTime;
        parameters_[1] = particleThermalResponseTime;
        parameters_[2] = gasHeatCapacity;
    }

    label modelId() const override
    {
        return 2;
    }

    const word& modelName() const override
    {
        return modelName_;
    }

    const FixedList<scalar, 4>& parameters() const override
    {
        return parameters_;
    }
};


inline autoPtr<GpuDragModel> GpuDragModel::New(const dictionary& dict)
{
    const word selected
    (
        dict.lookupOrDefault<word>
        (
            "gpuResidentDragModel",
            dict.lookupOrDefault<word>("dragModel", "SchillerNaumann")
        )
    );

    if (selected == "SchillerNaumann")
    {
        return autoPtr<GpuDragModel>(new GpuSchillerNaumannDragModel());
    }

    if (selected == "GidaspowErgunWenYu")
    {
        return autoPtr<GpuDragModel>
        (
            new GpuGidaspowErgunWenYuDragModel(dict)
        );
    }

    if (selected == "constantResponseTime")
    {
        return autoPtr<GpuDragModel>
        (
            new GpuConstantResponseTimeDragModel(dict)
        );
    }

    FatalErrorInFunction
        << "Unknown GPU dragModel '" << selected << "'. Available models: "
        << "SchillerNaumann, "
        << "GidaspowErgunWenYu, constantResponseTime" << exit(FatalError);

    return autoPtr<GpuDragModel>(nullptr);
}

}                  

#endif
