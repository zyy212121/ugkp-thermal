#ifndef GPU_THERMAL_GPU_SOLID_THERMAL_PROPERTIES_H
#define GPU_THERMAL_GPU_SOLID_THERMAL_PROPERTIES_H

#include "dictionary.H"
#include "scalarField.H"
#include "word.H"

#include <stdexcept>

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
        scalar constantValue;
        scalarField temperatures;
        scalarField values;
    };

    const PropertyModel rho_;
    const PropertyModel Cp_;
    const PropertyModel kappa_;
    const PropertyModel emissivity_;
    const scalar referenceTemperature_;
    const scalarField rhoCpKnots_;

    static PropertyModel readProperty
    (
        const dictionary& propertiesDictionary,
        const word& propertyName
    );

    static scalar evaluate(const PropertyModel& property, scalar temperature);
    static scalar lowerBound(const PropertyModel& property);
    static scalar upperBound(const PropertyModel& property);

    static scalarField buildRhoCpKnotUnion
    (
        const PropertyModel& rho,
        const PropertyModel& Cp
    );

    static scalar selectReferenceTemperature
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

    scalar integrateRhoCp(scalar lowerTemperature, scalar upperTemperature) const;

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
};

}                        
}                  

#endif
