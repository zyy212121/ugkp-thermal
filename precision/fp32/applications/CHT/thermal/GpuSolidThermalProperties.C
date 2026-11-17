#include "GpuSolidThermalProperties.H"
#include "ThermalHostDiagnostics.H"

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

bool finiteScalar(const GpuReal value)
{
    return std::isfinite(value);
}

GpuReal checkedProduct
(
    const GpuReal first,
    const GpuReal second,
    const char* context
)
{
    const GpuReal product = first*second;
    if (!finiteScalar(product) || product <= GpuReal(0))
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
    property.constantValue = GpuReal(0);

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
        const GpuReal temperature = nodes[nodeI].first();
        const GpuReal value = nodes[nodeI].second();

        property.temperatures[nodeI] = temperature;
        property.values[nodeI] = value;
    }

    return property;
}

GpuReal GpuSolidThermalProperties::lowerBound
(
    const PropertyModel& property
)
{
    return property.constant
      ? -std::numeric_limits<GpuReal>::infinity()
      : property.temperatures.first();
}

GpuReal GpuSolidThermalProperties::upperBound
(
    const PropertyModel& property
)
{
    return property.constant
      ? std::numeric_limits<GpuReal>::infinity()
      : property.temperatures.last();
}

GpuReal GpuSolidThermalProperties::evaluate
(
    const PropertyModel& property,
    const GpuReal temperature
)
{
    if (!finiteScalar(temperature) || temperature < GpuReal(0))
    {
        FatalErrorInFunction
            << "invalid consumed solid temperature=" << temperature
            << "; property=" << property.name << exit(FatalError);
    }
    const auto consumedValue = [&](const GpuReal value)->GpuReal
    {
        if
        (
            !finiteScalar(value)
         || (property.name != "emissivity" && value <= GpuReal(0))
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
        if (hostDiagnosticWarningsEnabled())
        {
            WarningInFunction
                << "finite solid table extrapolation; legacy outOfBounds=error"
                << "; property=" << property.name
                << "; temperature=" << temperature
                << "; tableMinimum=" << property.temperatures.first()
                << "; tableMaximum=" << property.temperatures.last() << '\n';
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
    const GpuReal x0 = property.temperatures[lowerI];
    const GpuReal x1 = property.temperatures[lowerI + 1];
    if (temperature == x0)
    {
        return consumedValue(property.values[lowerI]);
    }
    if (temperature == x1)
    {
        return consumedValue(property.values[lowerI + 1]);
    }

    const GpuReal directWidth = x1 - x0;
    GpuReal fraction = GpuReal(0);
    if (finiteScalar(directWidth))
    {
        fraction = (temperature - x0)/directWidth;
    }
    else
    {
        const GpuReal scale = std::max(std::abs(x0), std::abs(x1));
        const GpuReal scaledX0 = x0/scale;
        const GpuReal scaledX1 = x1/scale;
        const GpuReal scaledTemperature = temperature/scale;
        fraction =
            (scaledTemperature - scaledX0)/(scaledX1 - scaledX0);
    }

    const GpuReal lowerValue = property.values[lowerI];
    const GpuReal upperValue = property.values[lowerI + 1];
    const GpuReal result = fraction <= GpuReal(0.5)
      ? lowerValue + fraction*(upperValue - lowerValue)
      : upperValue + (GpuReal(1) - fraction)*(lowerValue - upperValue);

    return consumedValue(result);
}

Field<GpuReal> GpuSolidThermalProperties::buildRhoCpKnotUnion
(
    const PropertyModel& rho,
    const PropertyModel& Cp
)
{
    std::vector<GpuReal> knots;
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


    std::sort(knots.begin(), knots.end(), [](GpuReal a, GpuReal b)
    {
        return std::isnan(b) ? !std::isnan(a) : (!std::isnan(a) && a < b);
    });
    knots.erase(std::unique(knots.begin(), knots.end()), knots.end());

    Field<GpuReal> result(static_cast<label>(knots.size()));
    forAll(result, knotI)
    {
        result[knotI] = knots[static_cast<std::size_t>(knotI)];
    }
    return result;
}

GpuReal GpuSolidThermalProperties::selectReferenceTemperature
(
    const PropertyModel& rho,
    const PropertyModel& Cp
)
{
    if (rho.constant && Cp.constant)
    {
        return GpuReal(0);
    }

    const GpuReal reference = std::max(lowerBound(rho), lowerBound(Cp));

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

scalar GpuSolidThermalProperties::rho(const scalar inputTemperature) const
{
    const GpuReal temperature=GpuReal(inputTemperature);
    return evaluate(rho_, temperature);
}

scalar GpuSolidThermalProperties::Cp(const scalar inputTemperature) const
{
    const GpuReal temperature=GpuReal(inputTemperature);
    return evaluate(Cp_, temperature);
}

scalar GpuSolidThermalProperties::kappa(const scalar inputTemperature) const
{
    const GpuReal temperature=GpuReal(inputTemperature);
    return evaluate(kappa_, temperature);
}

scalar GpuSolidThermalProperties::emissivity(const scalar inputTemperature) const
{
    const GpuReal temperature=GpuReal(inputTemperature);
    return evaluate(emissivity_, temperature);
}

GpuReal GpuSolidThermalProperties::integrateRhoCp
(
    const GpuReal lowerTemperature,
    const GpuReal upperTemperature
) const
{
    if (lowerTemperature == upperTemperature)
    {
        return GpuReal(0);
    }
    if (upperTemperature < lowerTemperature)
    {
        return -integrateRhoCp(upperTemperature, lowerTemperature);
    }

    GpuReal integral = GpuReal(0);
    const auto integrateSegment =
    [&](const GpuReal a, const GpuReal b)
    {
        const GpuReal width = b - a;

        const GpuReal rho0 = evaluate(rho_, a);
        const GpuReal rho1 = evaluate(rho_, b);
        const GpuReal Cp0 = evaluate(Cp_, a);
        const GpuReal Cp1 = evaluate(Cp_, b);
        const GpuReal rho0Cp0 = checkedProduct
        (
            rho0,
            Cp0,
            "segment lower endpoint"
        );
        const GpuReal rho0Cp1 = checkedProduct
        (
            rho0,
            Cp1,
            "segment cross endpoint rho0*Cp1"
        );
        const GpuReal rho1Cp0 = checkedProduct
        (
            rho1,
            Cp0,
            "segment cross endpoint rho1*Cp0"
        );
        const GpuReal rho1Cp1 = checkedProduct
        (
            rho1,
            Cp1,
            "segment upper endpoint"
        );
        const GpuReal weightedEndpointProducts =
            GpuReal(2)*rho0Cp0
          + rho0Cp1
          + rho1Cp0
          + GpuReal(2)*rho1Cp1;

        const GpuReal segmentIntegral =
            (width/GpuReal(6))*weightedEndpointProducts;

        const GpuReal accumulated = integral + segmentIntegral;

        integral = accumulated;
    };

    GpuReal segmentLower = lowerTemperature;
    forAll(rhoCpKnots_, knotI)
    {
        const GpuReal knot = rhoCpKnots_[knotI];
        if (knot > lowerTemperature && knot < upperTemperature)
        {
            integrateSegment(segmentLower, knot);
            segmentLower = knot;
        }
    }
    integrateSegment(segmentLower, upperTemperature);

    return integral;
}

scalar GpuSolidThermalProperties::Hv(const scalar inputTemperature) const
{
    const GpuReal temperature=GpuReal(inputTemperature);
    return integrateRhoCp(referenceTemperature_, temperature);
}

scalar GpuSolidThermalProperties::deltaHv
(
    const scalar inputnewTemperature,
    const scalar inputoldTemperature
) const
{
    const GpuReal newTemperature=GpuReal(inputnewTemperature), oldTemperature=GpuReal(inputoldTemperature);
    return integrateRhoCp(oldTemperature, newTemperature);
}

scalar GpuSolidThermalProperties::Csec
(
    const scalar inputtrialTemperature,
    const scalar inputoldTemperature
) const
{
    const GpuReal trialTemperature=GpuReal(inputtrialTemperature), oldTemperature=GpuReal(inputoldTemperature);
    if (trialTemperature == oldTemperature)
    {
        return checkedProduct
        (
            evaluate(rho_,oldTemperature), evaluate(Cp_,oldTemperature),
            "coincident rho(Told)*Cp(Told)"
        );
    }

    const GpuReal deltaTemperature = trialTemperature - oldTemperature;

    const GpuReal secant =
        integrateRhoCp(oldTemperature, trialTemperature)/deltaTemperature;

    if (!finiteScalar(secant) || secant <= GpuReal(0))
    {
        FatalErrorInFunction
            << "invalid consumed solid Csec=" << secant
            << "; trialTemperature=" << trialTemperature
            << "; oldTemperature=" << oldTemperature << exit(FatalError);
    }
    return secant;
}

std::array<ugkp::gpu::ThermalPropertyDeviceTable,4>
GpuSolidThermalProperties::deviceTables() const
{
    const PropertyModel* properties[] = {&rho_, &Cp_, &kappa_, &emissivity_};
    std::array<ugkp::gpu::ThermalPropertyDeviceTable,4> tables;
    for (int i=0; i<4; ++i)
    {
        const auto& property = *properties[i];
        tables[i] = {property.constant, property.clampOutOfBounds, i==3,
            property.constantValue, int(property.temperatures.size()),
            property.temperatures.begin(), property.values.begin()};
    }
    return tables;
}

scalar GpuSolidThermalProperties::referenceTemperature() const noexcept
{
    return referenceTemperature_;
}

}
}
