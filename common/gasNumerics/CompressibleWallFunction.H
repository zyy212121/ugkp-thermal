#pragma once

#include "OpenFoamWallFunctions.cuh"

#if defined(__CUDACC__)
#define UGKP_COMPRESSIBLE_WALL_HD __host__ __device__ inline
#else
#define UGKP_COMPRESSIBLE_WALL_HD inline
#endif

class CompressibleWallFunction
{
    using Real = GpuReal;

public:
    struct Input
    {
        Real slipSpeed;
        Real inverseDistance;
        Real referenceDensity;
        Real viscosity;
        Real conductivity;
        Real cp;
        Real turbulentPrandtl;
        Real kappa;
        Real E;
        Real wallTemperature;
        Real cellTemperature;
        Real thermalYPlus;
        bool prescribedWallTemperature;
    };

    struct Result
    {
        ugkpwall::SpaldingWallState momentum{};
        ugkpwall::JayatillekeThermalTransport thermal{};
        Real wallTemperature = Real(0);
        Real recoveryRise = Real(0);
        Real residual = Real(0);
    };

private:
    struct SpaldingMoments
    {
        Real distance;
        Real slope;
        Real velocityMoment;
    };

    struct Profile
    {
        Real innerSpeedPlus;
        Real distancePlus;
        Real conductiveDistancePlus;
        Real outerResistancePerSpeed;
        Real thermalResistancePlus;
        Real recoveryRise;
        Real wallTemperature;
        Real energyDrive;
        Real innerTemperatureRatio;
    };

    UGKP_COMPRESSIBLE_WALL_HD static SpaldingMoments moments(const Real speedPlus, const Input& in)
    {
        const Real x = in.kappa*speedPlus;
        Real remainder, derivative, moment;
        if (x < Real(1))
        {
            Real derivativeTerm = x*x*x/Real(6);
            remainder = derivative = moment = Real(0);
            for (int order = 4; order <= 15; ++order)
            {
                const Real term = derivativeTerm*x/Real(order);
                remainder += term;
                derivative += derivativeTerm;
                moment += term*Real(order)/Real(order + 1);
                derivativeTerm = term;
            }
        }
        else
        {
            const Real x2 = x*x;
            const Real exponential = expm1(x);
            remainder = exponential - x - x2*(Real(0.5) + x/Real(6));
            derivative = exponential - x - Real(0.5)*x2;
            moment = ((x - Real(1))*exponential + x
                - x2*(Real(0.5) + x*(Real(1)/Real(3) + x/Real(8))))/x;
        }
        return {speedPlus + remainder/in.E, Real(1) + in.kappa*derivative/in.E,
            speedPlus*(Real(0.5)*speedPlus + moment/in.E)};
    }

    UGKP_COMPRESSIBLE_WALL_HD static Real thermalTransitionSpeed(const Input& in, const int iterations,
        const Real tolerance)
    {
        Real lower = Real(0), upper = in.thermalYPlus;
        Real speed = ugkpwall::minimum(upper, log(Real(1) + in.E*upper)/in.kappa);
        for (int i = 0; i < iterations; ++i)
        {
            const auto state = moments(speed, in);
            const Real residual = state.distance - in.thermalYPlus;
            if (fabs(residual) <= tolerance*in.thermalYPlus)
                break;
            if (residual > Real(0)) upper = speed;
            else lower = speed;
            const Real candidate = speed - residual/state.slope;
            const Real next = candidate > lower && candidate < upper
                ? candidate : lower + Real(0.5)*(upper - lower);
            if (next == speed) break;
            speed = next;
        }
        return speed;
    }

    UGKP_COMPRESSIBLE_WALL_HD static Real positiveQuadraticRoot(const Real a, const Real b, const Real c)
    {
        const Real scaled = c/b;
        return Real(2)*scaled/(Real(1) + sqrt(Real(1) + Real(4)*a*(scaled/b)));
    }

    struct Trial
    {
        Real speedPlus{};
        Profile layer{};
        Real residual{};
        Real densityResidual{};
    };

    struct RootBracket
    {
        Real lower, upper, lowerResidual, upperResidual;
        int previousSide = 0;

        UGKP_COMPRESSIBLE_WALL_HD Real next() const
        {
            const Real fraction = -lowerResidual/(upperResidual-lowerResidual);
            const Real candidate = lower + fraction*(upper-lower);
            return candidate > lower && candidate < upper
                ? candidate : lower + Real(0.5)*(upper-lower);
        }

        UGKP_COMPRESSIBLE_WALL_HD void update(const Real value, const Real residual)
        {
            const int side = residual > Real(0) ? 1 : -1;
            if (side > 0)
            {
                upper = value;
                upperResidual = residual;
                if (side == previousSide) lowerResidual *= Real(0.5);
            }
            else
            {
                lower = value;
                lowerResidual = residual;
                if (side == previousSide) upperResidual *= Real(0.5);
            }
            previousSide = side;
        }
    };

    UGKP_COMPRESSIBLE_WALL_HD static Profile molecularProfile(const Input& in,
        const Real pr, const Real speedPlus, const Real referenceReynolds)
    {
        const auto inner = moments(speedPlus, in);
        const Real uTau = in.slipSpeed/speedPlus;
        const Real recoveryPerDistance = (uTau/in.cp)*uTau*pr*(inner.velocityMoment/inner.distance);
        const Real referenceDistance = referenceReynolds/speedPlus;
        const Real temperature = in.prescribedWallTemperature ? in.wallTemperature
            : in.cellTemperature + positiveQuadraticRoot(Real(1), in.cellTemperature,
                recoveryPerDistance*referenceDistance*in.cellTemperature);
        const Real distance = in.prescribedWallTemperature ? referenceDistance
            : referenceDistance*(in.cellTemperature/temperature);
        const Real recovery = recoveryPerDistance*distance;
        const Real drive = in.prescribedWallTemperature
            ? (in.cellTemperature - in.wallTemperature) + recovery : Real(0);
        return {speedPlus, distance, distance, Real(0), pr*distance, recovery, temperature,
            drive, in.cellTemperature/temperature};
    }

    UGKP_COMPRESSIBLE_WALL_HD static Profile outerProfile(const Input& in, const Real pr,
        const Real speedPlus, const Real transformed, const Real distance,
        const Real thermalSpeed, const SpaldingMoments& transition)
    {
        const Real outerSpeed = speedPlus - thermalSpeed;
        const Real outerDistance = distance - in.thermalYPlus;
        const Real densityFactor = (transformed-thermalSpeed)/outerSpeed;
        const Real prandtlRatio = in.turbulentPrandtl/pr;
        const Real interfaceConductivity = prandtlRatio
            + densityFactor*(in.kappa*in.thermalYPlus-prandtlRatio);
        const Real relativeIncrease = in.kappa*densityFactor*outerDistance/interfaceConductivity;
        const Real logMeanFactor = relativeIncrease == Real(0)
            ? Real(1) : log1p(relativeIncrease)/relativeIncrease;
        const Real molecularFraction = (prandtlRatio/interfaceConductivity)*logMeanFactor;
        const Real outerConductiveDistance = outerDistance*molecularFraction;
        const Real conductiveDistance = molecularFraction >= Real(0.5)
            ? distance - outerDistance*(Real(1)-molecularFraction)
            : in.thermalYPlus + outerConductiveDistance;
        const Real outerResistance = pr*outerConductiveDistance;
        const Real resistance = pr*conductiveDistance;
        const Real uTau = in.slipSpeed/speedPlus;
        const Real kineticScale = (uTau/in.cp)*uTau;
        const Real innerRecovery = kineticScale*pr*transition.velocityMoment;
        const Real outerRecovery = kineticScale*(speedPlus+thermalSpeed)*Real(0.5)*outerResistance;
        const Real recovery = innerRecovery + outerRecovery;
        const Real temperature = in.prescribedWallTemperature
            ? in.wallTemperature : in.cellTemperature+recovery;
        const Real drive = in.prescribedWallTemperature
            ? (in.cellTemperature-in.wallTemperature)+recovery : Real(0);
        const Real innerTemperature = (in.cellTemperature+outerRecovery
            - drive*(outerResistance/resistance))/temperature;
        return {thermalSpeed, distance, conductiveDistance, outerResistance/outerSpeed,
            resistance, recovery, temperature, drive, innerTemperature};
    }

    UGKP_COMPRESSIBLE_WALL_HD static Real transformedSpeed(const Input& in, const Profile& layer,
        const Real edgeSpeedPlus)
    {
        const Real interval = edgeSpeedPlus - layer.innerSpeedPlus;
        if (interval == Real(0)) return edgeSpeedPlus;

        const Real leftRoot = sqrt(layer.innerTemperatureRatio);
        const Real edgeTheta = in.cellTemperature/layer.wallTemperature;
        const Real rightRoot = sqrt(edgeTheta);
        const Real rootSum = leftRoot + rightRoot;
        const Real uTau = in.slipSpeed/edgeSpeedPlus;
        const Real curvature = Real(0.5)*layer.outerResistancePerSpeed
            *(uTau/in.cp)*(uTau/layer.wallTemperature);
        if (curvature*interval*interval
            <= Real(8)*GPU_REAL_EPSILON*ugkpwall::minimum(layer.innerTemperatureRatio, edgeTheta))
            return layer.innerSpeedPlus + Real(2)*interval/rootSum;

        const Real scale = sqrt(curvature);
        const Real linear = layer.outerResistancePerSpeed*layer.energyDrive
            /(layer.wallTemperature*layer.thermalResistancePlus);
        const Real startNumerator = Real(2)*curvature*layer.innerSpeedPlus - linear;
        const Real endNumerator = startNumerator + Real(2)*curvature*interval;
        const Real temperatureRootDifference = (edgeTheta-layer.innerTemperatureRatio)/rootSum;
        const Real cross = Real(2)*scale
            *(Real(2)*curvature*interval*leftRoot - startNumerator*temperatureRootDifference);
        const Real dot = Real(4)*curvature*leftRoot*rightRoot + startNumerator*endNumerator;
        return layer.innerSpeedPlus + atan2(cross, dot)/scale;
    }

    UGKP_COMPRESSIBLE_WALL_HD static Real wallDensity(const Input& in, const Profile& layer)
    {
        return in.prescribedWallTemperature ? in.referenceDensity
            : in.referenceDensity*(in.cellTemperature/layer.wallTemperature);
    }

    UGKP_COMPRESSIBLE_WALL_HD static Trial molecularTrial(const Input& in, const Real pr,
        const Real speed, const Real referenceReynolds)
    {
        const auto layer = molecularProfile(in,pr,speed,referenceReynolds);
        const Real reynolds = layer.distancePlus*speed;
        const Real residual = ugkpwall::spaldingReynoldsRatio
            (speed,reynolds,log(reynolds),in.kappa,in.E)-Real(1);
        return {speed,layer,residual,Real(0)};
    }

    UGKP_COMPRESSIBLE_WALL_HD static Trial outerTrial(const Input& in, const Real pr,
        const Real transformed, const Real thermalSpeed, const SpaldingMoments& transition,
        const Real referenceReynolds, const Real transitionRecovery,
        const int maximumIterations, const Real tolerance)
    {
        const Real remainder = moments(transformed,in).distance-transformed
            + (in.thermalYPlus-transition.distance);
        const Real rootReynolds = sqrt(referenceReynolds);
        const Real scaledRemainder = (Real(0.5)*remainder)/rootReynolds;
        const Real speedAtReference = rootReynolds/(hypot(Real(1),scaledRemainder)+scaledRemainder);
        const Real outside = (transformed-thermalSpeed)/transformed;
        if (speedAtReference <= thermalSpeed)
            return {Real(0),{},outside,Real(0)};

        Real speed = speedAtReference;
        auto layer = outerProfile(in,pr,speed,transformed,speed+remainder,thermalSpeed,transition);
        Real densityResidual = Real(0);
        if (!in.prescribedWallTemperature)
        {
            const Real threshold = ((thermalSpeed/referenceReynolds)*(thermalSpeed+remainder))
                *(Real(1)+transitionRecovery/in.cellTemperature)-Real(1);
            if (threshold >= Real(0))
                return {Real(0),{},outside,Real(0)};
            densityResidual = layer.recoveryRise/in.cellTemperature;
            RootBracket bounds{thermalSpeed,speedAtReference,threshold,densityResidual};
            for (int i = 0; i < maximumIterations && fabs(densityResidual) > tolerance; ++i)
            {
                const Real next = bounds.next();
                if (next == bounds.lower || next == bounds.upper) break;
                speed = next;
                layer = outerProfile(in,pr,speed,transformed,speed+remainder,thermalSpeed,transition);
                densityResidual = ((speed/referenceReynolds)*layer.distancePlus)
                    *(layer.wallTemperature/in.cellTemperature)-Real(1);
                bounds.update(speed,densityResidual);
            }
        }
        const Real residual = (transformed-transformedSpeed(in,layer,speed))/transformed;
        return {speed,layer,residual,densityResidual};
    }

    UGKP_COMPRESSIBLE_WALL_HD static Trial solve(const Input& in, const Real pr,
        const Real referenceReynolds, const int maximumIterations, const Real tolerance)
    {
        const Real thermalSpeed = thermalTransitionSpeed(in,maximumIterations,tolerance);
        const auto transition = moments(thermalSpeed,in);
        const Real transitionUTau = in.slipSpeed/thermalSpeed;
        const Real transitionRecovery = (transitionUTau/in.cp)*transitionUTau*pr*transition.velocityMoment;
        const auto interfaceTrial = molecularTrial(in,pr,thermalSpeed,referenceReynolds);
        if (fabs(interfaceTrial.residual) <= tolerance)
            return interfaceTrial;
        if (interfaceTrial.residual > Real(0))
        {
            Real lower = Real(0), upper = sqrt(referenceReynolds);
            auto terminal = molecularTrial(in,pr,upper,referenceReynolds);
            for (int i = 0; i < maximumIterations && fabs(terminal.residual) > tolerance; ++i)
            {
                const Real speed = lower+Real(0.5)*(upper-lower);
                if (speed == lower || speed == upper) break;
                terminal = molecularTrial(in,pr,speed,referenceReynolds);
                if (terminal.residual > Real(0)) upper = speed; else lower = speed;
            }
            return terminal;
        }

        const Real upper = ugkpwall::maximum(Real(4),
            log(referenceReynolds/thermalSpeed)+log(Real(2)*in.E))/in.kappa;
        auto terminal = interfaceTrial;
        RootBracket bounds{thermalSpeed,upper,interfaceTrial.residual,(upper-thermalSpeed)/upper};
        for (int i = 0; i < maximumIterations && fabs(terminal.residual) > tolerance; ++i)
        {
            const Real transformed = bounds.next();
            if (transformed == bounds.lower || transformed == bounds.upper) break;
            const auto candidate = outerTrial(in,pr,transformed,thermalSpeed,transition,
                referenceReynolds,transitionRecovery,maximumIterations,tolerance);
            if (!std::isfinite(candidate.residual)) return {};
            if (candidate.speedPlus > Real(0) && fabs(candidate.residual) < fabs(terminal.residual))
                terminal = candidate;
            bounds.update(transformed,candidate.residual);
        }
        return terminal;
    }

public:
    UGKP_COMPRESSIBLE_WALL_HD static Result evaluate(const Input& in, const int maximumIterations,
        const Real tolerance)
    {
        const Real positiveInputs[] = {in.inverseDistance, in.referenceDensity, in.viscosity,
            in.conductivity, in.cp, in.turbulentPrandtl, in.kappa, in.E,
            in.cellTemperature, in.thermalYPlus, tolerance};
        for (const Real value : positiveInputs)
            if (!std::isfinite(value) || value <= Real(0)) return {};
        if (!std::isfinite(in.slipSpeed) || in.slipSpeed < Real(0)
            || (in.prescribedWallTemperature && (!std::isfinite(in.wallTemperature)
                || in.wallTemperature <= Real(0))) || maximumIterations < 1)
            return {};

        const Real initialWallTemperature = in.prescribedWallTemperature
            ? in.wallTemperature : in.cellTemperature;
        if (in.slipSpeed == Real(0))
        {
            const Real flux = in.conductivity*in.inverseDistance
                *(in.cellTemperature - initialWallTemperature);
            return {{}, {in.conductivity, std::isfinite(flux), flux},
                initialWallTemperature, Real(0), Real(0)};
        }

        const Real pr = in.viscosity*in.cp/in.conductivity;
        const Real referenceReynolds = in.referenceDensity*in.slipSpeed
            /(in.viscosity*in.inverseDistance);
        if (!std::isfinite(pr) || !std::isfinite(referenceReynolds)
            || pr <= Real(0) || referenceReynolds <= Real(0)) return {};
        const auto terminal = solve(in,pr,referenceReynolds,maximumIterations,tolerance);
        const Real speedPlus = terminal.speedPlus;
        const Profile& accepted = terminal.layer;
        const Real residual = fabs(terminal.residual) >= fabs(terminal.densityResidual)
            ? terminal.residual : terminal.densityResidual;
        if (speedPlus <= Real(0) || !std::isfinite(residual)) return {};
        const Real uTau = in.slipSpeed/speedPlus;
        const Real density = wallDensity(in, accepted);
        const Real nu = in.viscosity/density;
        const Real conductivity = in.conductivity
            *(accepted.distancePlus/accepted.conductiveDistancePlus);
        const Real heatFlux = conductivity*in.inverseDistance*accepted.energyDrive;
        const Real dynamicViscosity = (density*uTau/speedPlus)/in.inverseDistance;
        const Real nut = ugkpwall::maximum(Real(0), (dynamicViscosity-in.viscosity)/in.referenceDensity);
        const Real yPlus = uTau/(in.inverseDistance*nu);
        const bool valid = std::isfinite(heatFlux) && std::isfinite(conductivity)
            && conductivity > Real(0) && std::isfinite(nut) && std::isfinite(yPlus)
            && std::isfinite(accepted.wallTemperature) && accepted.wallTemperature > Real(0);
        return {{uTau, yPlus, nut}, {conductivity, valid, heatFlux},
            accepted.wallTemperature, accepted.recoveryRise, residual};
    }
};

#undef UGKP_COMPRESSIBLE_WALL_HD
