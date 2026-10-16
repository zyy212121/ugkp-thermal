#include "GpuSolidThermalProperties.H"

#include "List.H"
#include "Tuple2.H"
#include "entry.H"
#include "error.H"

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

scalar checkedProduct
(
    const scalar first,
    const scalar second,
    const char* context
)
{
    const scalar product = first*second;
    if (!finiteScalar(product) || product <= scalar(0))
    {
        FatalErrorInFunction
            << "invalid consumed solid rhoCp; context=" << context
            << "; first=" << first << "; second=" << second
            << "; rhoCp=" << product << exit(FatalError);
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
    if (!finiteScalar(temperature) || temperature < scalar(0))
    {
        FatalErrorInFunction
            << "invalid consumed solid temperature=" << temperature
            << "; property=" << property.name << exit(FatalError);
    }
    const auto consumedValue = [&](const scalar value)->scalar
    {
        if
        (
            !finiteScalar(value)
         || (property.name != "emissivity" && value <= scalar(0))
        )
        {
            FatalErrorInFunction
                << "invalid consumed solid property; property=" << property.name
                << "; temperature=" << temperature << "; value=" << value
                << exit(FatalError);
        }
        return value;
    };
    if (property.constant)
    {
        return consumedValue(property.constantValue);
    }

    if
    (
        temperature < property.temperatures.first()
     || temperature > property.temperatures.last()
    )
    {
        if (property.clampOutOfBounds)
        {
            return consumedValue
            (
                temperature < property.temperatures.first()
              ? property.values.first() : property.values.last()
            );
        }
    }

    if (temperature == property.temperatures.last())
    {
        return consumedValue(property.values.last());
    }



    label first = 0;
    label count = property.temperatures.size();
    while (count)
    {
        const label step = count/2;
        const label middle = first + step;
        if (!(temperature < property.temperatures[middle]))
        {
            first = middle + 1;
            count -= step + 1;
        }
        else
        {
            count = step;
        }
    }
    const label lowerI = temperature < property.temperatures.first()
      ? 0
      : temperature > property.temperatures.last()
      ? property.temperatures.size() - 2
      : std::max(label(0), std::min(first - 1, property.temperatures.size() - 2));
    const scalar x0 = property.temperatures[lowerI];
    const scalar x1 = property.temperatures[lowerI + 1];
    if (temperature == x0)
    {
        return consumedValue(property.values[lowerI]);
    }
    if (temperature == x1)
    {
        return consumedValue(property.values[lowerI + 1]);
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

    const scalar lowerValue = property.values[lowerI];
    const scalar upperValue = property.values[lowerI + 1];
    const scalar result = fraction <= scalar(0.5)
      ? lowerValue + fraction*(upperValue - lowerValue)
      : upperValue + (scalar(1) - fraction)*(lowerValue - upperValue);

    return consumedValue(result);
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


    std::sort(knots.begin(), knots.end(), [](scalar a, scalar b)
    {
        return std::isnan(b) ? !std::isnan(a) : (!std::isnan(a) && a < b);
    });
    knots.erase(std::unique(knots.begin(), knots.end()), knots.end());

    scalarField result(static_cast<label>(knots.size()));
    forAll(result, knotI)
    {
        result[knotI] = knots[static_cast<std::size_t>(knotI)];
    }
    return result;
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

        const scalar rho0 = evaluate(rho_, a);
        const scalar rho1 = evaluate(rho_, b);
        const scalar Cp0 = evaluate(Cp_, a);
        const scalar Cp1 = evaluate(Cp_, b);
        const scalar rho0Cp0 = checkedProduct
        (
            rho0,
            Cp0,
            "segment lower endpoint"
        );
        const scalar rho0Cp1 = checkedProduct
        (
            rho0,
            Cp1,
            "segment cross endpoint rho0*Cp1"
        );
        const scalar rho1Cp0 = checkedProduct
        (
            rho1,
            Cp0,
            "segment cross endpoint rho1*Cp0"
        );
        const scalar rho1Cp1 = checkedProduct
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

        const scalar segmentIntegral =
            (width/scalar(6))*weightedEndpointProducts;

        const scalar accumulated = integral + segmentIntegral;

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
    if (trialTemperature == oldTemperature)
    {
        return checkedProduct
        (
            rho(oldTemperature), Cp(oldTemperature),
            "coincident rho(Told)*Cp(Told)"
        );
    }

    const scalar deltaTemperature = trialTemperature - oldTemperature;

    const scalar secant =
        deltaHv(trialTemperature, oldTemperature)/deltaTemperature;

    if (!finiteScalar(secant) || secant <= scalar(0))
    {
        FatalErrorInFunction
            << "invalid consumed solid Csec=" << secant
            << "; trialTemperature=" << trialTemperature
            << "; oldTemperature=" << oldTemperature << exit(FatalError);
    }
    return secant;
}

scalar GpuSolidThermalProperties::referenceTemperature() const noexcept
{
    return referenceTemperature_;
}

}
}
