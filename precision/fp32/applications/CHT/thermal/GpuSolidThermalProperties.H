#ifndef GPU_THERMAL_GPU_SOLID_THERMAL_PROPERTIES_H
#define GPU_THERMAL_GPU_SOLID_THERMAL_PROPERTIES_H

#include "dictionary.H"
#include "Field.H"
#include "word.H"

#include <stdexcept>
#include <array>
#include "../../../common/gpu/SolidThermalMath.H"

namespace Foam
{
namespace gpuThermal
{

class GpuSolidThermalPropertiesError
:
    public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

class GpuSolidThermalProperties
{
    struct PropertyModel
    {
        word name;
        bool constant;
        bool clampOutOfBounds;
        GpuReal constantValue;
        Field<GpuReal> temperatures;
        Field<GpuReal> values;
    };

    const PropertyModel rho_;
    const PropertyModel Cp_;
    const PropertyModel kappa_;
    const PropertyModel emissivity_;
    const GpuReal referenceTemperature_;
    const Field<GpuReal> rhoCpKnots_;

    static PropertyModel readProperty
    (
        const dictionary& propertiesDictionary,
        const word& propertyName
    );

    static GpuReal evaluate(const PropertyModel& property, GpuReal temperature);
    static GpuReal lowerBound(const PropertyModel& property);
    static GpuReal upperBound(const PropertyModel& property);

    static Field<GpuReal> buildRhoCpKnotUnion
    (
        const PropertyModel& rho,
        const PropertyModel& Cp
    );

    static GpuReal selectReferenceTemperature
    (
        const PropertyModel& rho,
        const PropertyModel& Cp
    );

    static void validateCommonTemperatureRange
    (
        const PropertyModel& rho,
        const PropertyModel& Cp,
        const PropertyModel& kappa,
        const PropertyModel& emissivity
    );

    GpuReal integrateRhoCp(GpuReal lowerTemperature, GpuReal upperTemperature) const;

public:
    explicit GpuSolidThermalProperties
    (
        const dictionary& propertiesDictionary
    );

    scalar rho(scalar temperature) const;
    scalar Cp(scalar temperature) const;
    scalar kappa(scalar temperature) const;
    scalar emissivity(scalar temperature) const;


    scalar Hv(scalar temperature) const;



    scalar deltaHv(scalar newTemperature, scalar oldTemperature) const;


    scalar Csec(scalar trialTemperature, scalar oldTemperature) const;

    scalar referenceTemperature() const noexcept;
    std::array<ugkp::gpu::ThermalPropertyDeviceTable,4> deviceTables() const;
    const Field<GpuReal>& caloricKnots() const noexcept { return rhoCpKnots_; }
};

}
}

#endif
