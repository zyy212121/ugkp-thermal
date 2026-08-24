#include "GpuSolidThermalProperties.H"

#include "List.H"
#include "Tuple2.H"
#include "entry.H"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace Foam
{
namespace gpuThermal
{

namespace
{

[[noreturn]] void propertiesFailure
(
    const word& propertyName,
    const std::string& reason
)
{
    throw GpuSolidThermalPropertiesError
    (
        std::string(propertyName.c_str()) + ": " + reason
    );
}

bool finiteScalar(const scalar value)
{
    return std::isfinite(static_cast<double>(value));
}

void validatePropertyValue
(
    const word& propertyName,
    const scalar value,
    const label nodeI
)
{
    std::ostringstream context;
    context.precision(17);
    if (nodeI >= 0)
    {
        context << "table value at node " << nodeI;
    }
    else
    {
        context << "constant value";
    }

    if (!finiteScalar(value))
    {
        propertiesFailure
        (
            propertyName,
            context.str() + " must be finite"
        );
    }

    if (propertyName == "emissivity")
    {
        if (value < scalar(0) || value > scalar(1))
        {
            context << " must be in [0,1], actual=" << value;
            propertiesFailure(propertyName, context.str());
        }
    }
    else if (!(value > scalar(0)))
    {
        context << " must be positive, actual=" << value;
        propertiesFailure(propertyName, context.str());
    }
}

scalar checkedPositiveProduct
(
    const scalar first,
    const scalar second,
    const std::string& context
)
{
    const scalar product = first*second;
    if (!finiteScalar(product) || !(product > scalar(0)))
    {
        propertiesFailure
        (
            "rho/Cp",
            context + " product is nonfinite or nonpositive"
        );
    }
    return product;
}

}             

GpuSolidThermalProperties::PropertyModel
GpuSolidThermalProperties::readProperty
(
    const dictionary& propertiesDictionary,
    const word& propertyName
)
{
    const entry* const propertyEntry =
        propertiesDictionary.lookupEntryPtr(propertyName, false, false);
    if (propertyEntry == nullptr || !propertyEntry->isDict())
    {
        propertiesFailure(propertyName, "required property dictionary is missing");
    }

    const dictionary& propertyDictionary = propertyEntry->dict();
    if (!propertyDictionary.found("type", false, false))
    {
        propertiesFailure(propertyName, "required type entry is missing");
    }

    const word type(propertyDictionary.lookup<word>("type", false, false));
    PropertyModel property;
    property.name = propertyName;
    property.constant = false;
    property.clampOutOfBounds = false;
    property.constantValue = scalar(0);

    if (type == "constant")
    {
        if (!propertyDictionary.found("value", false, false))
        {
            propertiesFailure(propertyName, "constant property requires value");
        }

        property.constant = true;
        property.constantValue =
            propertyDictionary.lookup<scalar>("value", false, false);
        validatePropertyValue(propertyName, property.constantValue, -1);
        return property;
    }

    if (type != "table")
    {
        propertiesFailure
        (
            propertyName,
            "unsupported type '" + std::string(type.c_str())
          + "' (required constant or table)"
        );
    }

    if (!propertyDictionary.found("outOfBounds", false, false))
    {
        propertiesFailure
        (
            propertyName,
            "table requires explicit outOfBounds error or clamp"
        );
    }
    const word outOfBounds
    (
        propertyDictionary.lookup<word>("outOfBounds", false, false)
    );
    if (outOfBounds != "error" && outOfBounds != "clamp")
    {
        propertiesFailure
        (
            propertyName,
            "table requires outOfBounds error or clamp; warn/repeat are forbidden"
        );
    }
    property.clampOutOfBounds = (outOfBounds == "clamp");

    if (propertyDictionary.found("interpolationScheme", false, false))
    {
        const word interpolationScheme
        (
            propertyDictionary.lookup<word>
            (
                "interpolationScheme",
                false,
                false
            )
        );
        if (interpolationScheme != "linear")
        {
            propertiesFailure
            (
                propertyName,
                "v1 table interpolation is piecewise linear"
            );
        }
    }

    if (!propertyDictionary.found("values", false, false))
    {
        propertiesFailure(propertyName, "table requires embedded values");
    }

    List<Tuple2<scalar, scalar>> nodes;
    propertyDictionary.lookup("values", false, false) >> nodes;
    if (nodes.size() < 2)
    {
        propertiesFailure(propertyName, "table requires at least two nodes");
    }

    property.temperatures.setSize(nodes.size());
    property.values.setSize(nodes.size());
    forAll(nodes, nodeI)
    {
        const scalar temperature = nodes[nodeI].first();
        const scalar value = nodes[nodeI].second();
        if (!finiteScalar(temperature))
        {
            propertiesFailure
            (
                propertyName,
                "table temperature axis must be finite"
            );
        }
        if
        (
            nodeI > 0
         && !(temperature > property.temperatures[nodeI - 1])
        )
        {
            propertiesFailure
            (
                propertyName,
                "table temperature axis must be strictly increasing"
            );
        }
        validatePropertyValue(propertyName, value, nodeI);
        property.temperatures[nodeI] = temperature;
        property.values[nodeI] = value;
    }

    return property;
}

scalar GpuSolidThermalProperties::lowerBound
(
    const PropertyModel& property
)
{
    return property.constant
      ? -std::numeric_limits<scalar>::infinity()
      : property.temperatures.first();
}

scalar GpuSolidThermalProperties::upperBound
(
    const PropertyModel& property
)
{
    return property.constant
      ? std::numeric_limits<scalar>::infinity()
      : property.temperatures.last();
}

scalar GpuSolidThermalProperties::evaluate
(
    const PropertyModel& property,
    const scalar temperature
)
{
    if (!finiteScalar(temperature))
    {
        propertiesFailure(property.name, "query temperature must be finite");
    }
    if (property.constant)
    {
        return property.constantValue;
    }

    if
    (
        temperature < property.temperatures.first()
     || temperature > property.temperatures.last()
    )
    {
        if (property.clampOutOfBounds)
        {
            return temperature < property.temperatures.first()
              ? property.values.first()
              : property.values.last();
        }

        std::ostringstream reason;
        reason.precision(17);
        reason
            << "temperature " << temperature
            << " is outside [" << property.temperatures.first()
            << ',' << property.temperatures.last()
            << "] with outOfBounds error";
        propertiesFailure(property.name, reason.str());
    }

    if (temperature == property.temperatures.last())
    {
        return property.values.last();
    }

    const scalar* const begin = property.temperatures.begin();
    const scalar* const end = property.temperatures.end();
    const scalar* const upper = std::upper_bound(begin, end, temperature);
    const label lowerI = static_cast<label>(upper - begin - 1);
    const scalar x0 = property.temperatures[lowerI];
    const scalar x1 = property.temperatures[lowerI + 1];
    if (temperature == x0)
    {
        return property.values[lowerI];
    }
    if (temperature == x1)
    {
        return property.values[lowerI + 1];
    }

    const scalar directWidth = x1 - x0;
    scalar fraction = scalar(0);
    if (finiteScalar(directWidth))
    {
        fraction = (temperature - x0)/directWidth;
    }
    else
    {
                                                                       
                                                                             
        const scalar scale = std::max(std::abs(x0), std::abs(x1));
        const scalar scaledX0 = x0/scale;
        const scalar scaledX1 = x1/scale;
        const scalar scaledTemperature = temperature/scale;
        fraction =
            (scaledTemperature - scaledX0)/(scaledX1 - scaledX0);
    }
    if
    (
        !finiteScalar(fraction)
     || fraction < scalar(0)
     || fraction > scalar(1)
    )
    {
        propertiesFailure(property.name, "invalid interpolation fraction");
    }

    const scalar lowerValue = property.values[lowerI];
    const scalar upperValue = property.values[lowerI + 1];
    const scalar result = fraction <= scalar(0.5)
      ? lowerValue + fraction*(upperValue - lowerValue)
      : upperValue + (scalar(1) - fraction)*(lowerValue - upperValue);
    if (!finiteScalar(result))
    {
        propertiesFailure(property.name, "interpolated value is nonfinite");
    }
    if
    (
        (property.name == "emissivity"
          && (result < scalar(0) || result > scalar(1)))
     || (property.name != "emissivity" && !(result > scalar(0)))
    )
    {
        propertiesFailure(property.name, "interpolated value is out of range");
    }
    return result;
}

scalarField GpuSolidThermalProperties::buildRhoCpKnotUnion
(
    const PropertyModel& rho,
    const PropertyModel& Cp
)
{
    std::vector<scalar> knots;
    knots.reserve
    (
        static_cast<std::size_t>
        (
            rho.temperatures.size() + Cp.temperatures.size()
        )
    );
    knots.insert
    (
        knots.end(),
        rho.temperatures.begin(),
        rho.temperatures.end()
    );
    knots.insert
    (
        knots.end(),
        Cp.temperatures.begin(),
        Cp.temperatures.end()
    );
    std::sort(knots.begin(), knots.end());
    knots.erase(std::unique(knots.begin(), knots.end()), knots.end());

    scalarField result(static_cast<label>(knots.size()));
    forAll(result, knotI)
    {
        result[knotI] = knots[static_cast<std::size_t>(knotI)];
    }
    return result;
}

void GpuSolidThermalProperties::validateCommonTemperatureRange
(
    const PropertyModel& rho,
    const PropertyModel& Cp,
    const PropertyModel& kappa,
    const PropertyModel& emissivity
)
{
    const scalar commonLower = std::max
    (
        std::max(lowerBound(rho), lowerBound(Cp)),
        std::max(lowerBound(kappa), lowerBound(emissivity))
    );
    const scalar commonUpper = std::min
    (
        std::min(upperBound(rho), upperBound(Cp)),
        std::min(upperBound(kappa), upperBound(emissivity))
    );
    if (!(commonLower < commonUpper))
    {
        propertiesFailure
        (
            "properties",
            "rho/Cp/kappa/emissivity tables have no common temperature interval"
        );
    }
}

scalar GpuSolidThermalProperties::selectReferenceTemperature
(
    const PropertyModel& rho,
    const PropertyModel& Cp
)
{
    if (rho.constant && Cp.constant)
    {
        return scalar(0);
    }

    const scalar reference = std::max(lowerBound(rho), lowerBound(Cp));
    const scalar commonUpper = std::min(upperBound(rho), upperBound(Cp));
    if (!finiteScalar(reference) || !(reference < commonUpper))
    {
        propertiesFailure
        (
            "rho/Cp",
            "tables have no common finite reference temperature"
        );
    }
    return reference;
}

GpuSolidThermalProperties::GpuSolidThermalProperties
(
    const dictionary& propertiesDictionary
)
:
    rho_(readProperty(propertiesDictionary, "rho")),
    Cp_(readProperty(propertiesDictionary, "Cp")),
    kappa_(readProperty(propertiesDictionary, "kappa")),
    emissivity_(readProperty(propertiesDictionary, "emissivity")),
    referenceTemperature_(selectReferenceTemperature(rho_, Cp_)),
    rhoCpKnots_(buildRhoCpKnotUnion(rho_, Cp_))
{
    validateCommonTemperatureRange(rho_, Cp_, kappa_, emissivity_);
}

scalar GpuSolidThermalProperties::rho(const scalar temperature) const
{
    return evaluate(rho_, temperature);
}

scalar GpuSolidThermalProperties::Cp(const scalar temperature) const
{
    return evaluate(Cp_, temperature);
}

scalar GpuSolidThermalProperties::kappa(const scalar temperature) const
{
    return evaluate(kappa_, temperature);
}

scalar GpuSolidThermalProperties::emissivity(const scalar temperature) const
{
    return evaluate(emissivity_, temperature);
}

scalar GpuSolidThermalProperties::integrateRhoCp
(
    const scalar lowerTemperature,
    const scalar upperTemperature
) const
{
    if (!finiteScalar(lowerTemperature) || !finiteScalar(upperTemperature))
    {
        propertiesFailure("rho/Cp", "enthalpy temperatures must be finite");
    }

                                                                             
                             
    (void)evaluate(rho_, lowerTemperature);
    (void)evaluate(Cp_, lowerTemperature);
    (void)evaluate(rho_, upperTemperature);
    (void)evaluate(Cp_, upperTemperature);

    if (lowerTemperature == upperTemperature)
    {
        return scalar(0);
    }
    if (upperTemperature < lowerTemperature)
    {
        return -integrateRhoCp(upperTemperature, lowerTemperature);
    }

    scalar integral = scalar(0);
    const auto integrateSegment =
    [&](const scalar a, const scalar b)
    {
        const scalar width = b - a;
        if (!(width > scalar(0)) || !finiteScalar(width))
        {
            propertiesFailure("rho/Cp", "invalid enthalpy integration interval");
        }

        const scalar rho0 = evaluate(rho_, a);
        const scalar rho1 = evaluate(rho_, b);
        const scalar Cp0 = evaluate(Cp_, a);
        const scalar Cp1 = evaluate(Cp_, b);
        const scalar rho0Cp0 = checkedPositiveProduct
        (
            rho0,
            Cp0,
            "segment lower endpoint"
        );
        const scalar rho0Cp1 = checkedPositiveProduct
        (
            rho0,
            Cp1,
            "segment cross endpoint rho0*Cp1"
        );
        const scalar rho1Cp0 = checkedPositiveProduct
        (
            rho1,
            Cp0,
            "segment cross endpoint rho1*Cp0"
        );
        const scalar rho1Cp1 = checkedPositiveProduct
        (
            rho1,
            Cp1,
            "segment upper endpoint"
        );
        const scalar weightedEndpointProducts =
            scalar(2)*rho0Cp0
          + rho0Cp1
          + rho1Cp0
          + scalar(2)*rho1Cp1;
        if
        (
            !finiteScalar(weightedEndpointProducts)
         || !(weightedEndpointProducts > scalar(0))
        )
        {
            propertiesFailure
            (
                "rho/Cp",
                "weighted endpoint products are nonfinite or nonpositive"
            );
        }

        const scalar segmentIntegral =
            (width/scalar(6))*weightedEndpointProducts;
        if (!finiteScalar(segmentIntegral) || !(segmentIntegral > scalar(0)))
        {
            propertiesFailure
            (
                "rho/Cp",
                "forward enthalpy segment is nonfinite or nonpositive"
            );
        }

        const scalar accumulated = integral + segmentIntegral;
        if (!finiteScalar(accumulated) || !(accumulated > integral))
        {
            propertiesFailure
            (
                "rho/Cp",
                "forward enthalpy integral is nonfinite or failed to increase"
            );
        }
        integral = accumulated;
    };

    scalar segmentLower = lowerTemperature;
    forAll(rhoCpKnots_, knotI)
    {
        const scalar knot = rhoCpKnots_[knotI];
        if (knot > lowerTemperature && knot < upperTemperature)
        {
            integrateSegment(segmentLower, knot);
            segmentLower = knot;
        }
    }
    integrateSegment(segmentLower, upperTemperature);

    if (!finiteScalar(integral) || !(integral > scalar(0)))
    {
        propertiesFailure
        (
            "rho/Cp",
            "forward enthalpy integral is nonfinite or nonpositive"
        );
    }

    return integral;
}

scalar GpuSolidThermalProperties::Hv(const scalar temperature) const
{
    return integrateRhoCp(referenceTemperature_, temperature);
}

scalar GpuSolidThermalProperties::deltaHv
(
    const scalar newTemperature,
    const scalar oldTemperature
) const
{
    return integrateRhoCp(oldTemperature, newTemperature);
}

scalar GpuSolidThermalProperties::Csec
(
    const scalar trialTemperature,
    const scalar oldTemperature
) const
{
                                                                            
                                                                      
    const scalar oldRhoCp = checkedPositiveProduct
    (
        rho(oldTemperature),
        Cp(oldTemperature),
        "coincident rho(Told)*Cp(Told)"
    );
    (void)rho(trialTemperature);
    (void)Cp(trialTemperature);

    if (trialTemperature == oldTemperature)
    {
        return oldRhoCp;
    }

    const scalar deltaTemperature = trialTemperature - oldTemperature;
    if (!finiteScalar(deltaTemperature) || deltaTemperature == scalar(0))
    {
        propertiesFailure
        (
            "rho/Cp",
            "nonzero secant temperature difference is invalid"
        );
    }
    const scalar secant =
        deltaHv(trialTemperature, oldTemperature)/deltaTemperature;
    if (!finiteScalar(secant) || !(secant > scalar(0)))
    {
        propertiesFailure("rho/Cp", "secant heat capacity is nonpositive/nonfinite");
    }
    return secant;
}

scalar GpuSolidThermalProperties::referenceTemperature() const noexcept
{
    return referenceTemperature_;
}

}                        
}                  
