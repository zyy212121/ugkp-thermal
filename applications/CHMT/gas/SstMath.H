#ifndef CHMT_GAS_SSTMATH_H
#define CHMT_GAS_SSTMATH_H
#include "gas/AleFlux.H"
#include "../../../common/gasNumerics/GpuSstAlgebra.cuh"
namespace chmt {
    static_assert(sizeof(GpuReal) == sizeof(Real), "CHMT SST requires the original helper's FP64 contract");
    CHMT_HD inline ugkwp::SstCoefficients sstCoefficients(const SstConfig& s) {
        return {
            s.sigmaK1, s.sigmaK2, s.sigmaOmega1, s.sigmaOmega2, s.beta1, s.beta2, s.betaStar, s.gamma1,
                s.gamma2, s.a1, 1, s.productionLimit
        };
    }
    struct VelocityInvariants {
        Real divergence = 0, strain2 = 0, productionPerNut = 0;
    };
    CHMT_HD inline VelocityInvariants velocityInvariants(const GasGradient& g) {
        VelocityInvariants r;
        r.divergence = g.velocity[0].x+g.velocity[1].y+g.velocity[2].z;
        for (int i = 0; i<3; ++i)for (int j = 0; j<3; ++j) {
            const Real twice = component(g.velocity[i], j)+component(g.velocity[j], i);
            r.strain2 += 0.5*twice*twice;
            r.productionPerNut += (twice-(i == j?2*r.divergence/3:0))*component(g.velocity[i], j);
        }
        r.productionPerNut = maxValue(0, r.productionPerNut);
        return r;
    }
    CHMT_HD inline Real wallOmega(Real nu, Real distance, const SstConfig& s) {
        if (!finite(nu) || nu <= 0 || !finite(distance) || distance <= 0)return undefinedValue();
        return ugkwp::sstLowReWallOmega(nu, distance, sstCoefficients(s));
    }
    // The wall primitive is an explicit EOS-consistent interface state. Missing
    // T/composition/rho/p is an error, never an owner-cell viscosity fallback.
    CHMT_HD inline bool wallOmegaTarget(const GasPrimitive& wall, Real endpointDistance, const PhysicsConfig& p,
        Real& target) {
        GasPrimitive recovered;
        if (!explicitBoundaryPrimitive(wall, p, recovered))return false;
        const Real value = wallOmega(p.gasViscosity/recovered.rho, endpointDistance, p.sst);
        if (!finite(value) || value <= 0 || value<p.sst.minimumOmega)return false;
        target = value;
        return true;
    }
    CHMT_HD inline bool sstStageAdmissible(const SstQ& state, Real gasMass,
        const PhysicsConfig& physics, bool wallConstraintPending) {
        if (!finite(gasMass) || gasMass <= 0 || !finite(state.rhoK) || !finite(state.rhoOmega)
            || state.rhoK < physics.sst.minimumK*gasMass || state.rhoOmega <= 0)return false;
        // A positive wall-adjacent provisional omega may fall below its floor
        // during source integration. Its mandatory endpoint projection comes next.
        return wallConstraintPending || state.rhoOmega >= physics.sst.minimumOmega*gasMass;
    }
    CHMT_HD inline bool sstEndpointAdmissible(const SstQ& state, Real gasMass,
        const PhysicsConfig& physics, Real constraintDelta) {
        // Explicit finite check: NaN must never bypass a residual comparison.
        return finite(constraintDelta) && sstStageAdmissible(state, gasMass, physics, false);
    }
    CHMT_HD inline bool applyWallOmegaConstraint(const GasQ& candidateGas, const SstQ& evolved,
        Real faceTargetSum, int wallCount, const PhysicsConfig& p, SstQ& constrained, Real& delta) {
        if (wallCount < 0 || !sstStageAdmissible(evolved, candidateGas.mass, p, wallCount > 0))return false;
        constrained = evolved;
        delta = 0;
        if (wallCount == 0)return sstEndpointAdmissible(constrained, candidateGas.mass, p, delta);
        const Real target = faceTargetSum/wallCount;
        // Equal face weighting, as in the original OF-aligned kernel.
        if (!finite(target) || target <= 0 || target<p.sst.minimumOmega)return false;
        constrained.rhoOmega = candidateGas.mass*target;
        delta = constrained.rhoOmega-evolved.rhoOmega;
        // kg/s inventory change, not power or modeled-k heat.
        return sstEndpointAdmissible(constrained, candidateGas.mass, p, delta);
    }
    CHMT_HD inline SstPrimitive lowReSstWallTrace(const SstPrimitive& cell) {
        SstPrimitive wall = cell;
        wall.k = 0;
        wall.eddyViscosity = 0;
        wall.blend1 = wall.blend2 = 1;
        // The adjacent-cell omega constraint is copied, not re-imposed at a face.
        return wall;
    }
    CHMT_HD inline Vec3 lowReWallOmegaGradient(Vec3 gradient, Vec3 area) {
        const Vec3 normal = normalized(area);
        return gradient-normal*dot(gradient, normal);
    }
    CHMT_HD inline bool sstAlgebra(const SstQ& q, Real rho, Real volume, const GasPrimitive&,
        const GasGradient& g, Vec3 gradK, Vec3 gradOmega, Real distance, const PhysicsConfig& p,
        SstPrimitive& out) {
        const Real mass = rho*volume;
        if (!(mass>0) || !finite(mass) || !finite(distance) || distance <= 0 || !finite(q.rhoK)
            || !finite(q.rhoOmega) || q.rhoK<0 || q.rhoOmega <= 0)return false;
        SstPrimitive s;
        s.k = q.rhoK/mass;
        s.omega = q.rhoOmega/mass;
        if (s.k<p.sst.minimumK || s.omega<p.sst.minimumOmega)return false;
        const auto c = sstCoefficients(p.sst);
        const auto inv = velocityInvariants(g);
        const Real cd = ugkwp::sstCrossDiffusion(s.omega, dot(gradK, gradOmega), c), nu = p.gasViscosity/rho;
        s.blend1 = ugkwp::sstF1(s.k, s.omega, nu, distance, cd, c);
        s.blend2 = ugkwp::sstF2(s.k, s.omega, nu, distance, c);
        s.eddyViscosity = ugkwp::sstNut(s.k, s.omega, inv.strain2, s.blend2, c);
        if (!finite(s.eddyViscosity) || !finite(s.blend1) || !finite(s.blend2))return false;
        out = s;
        return true;
    }
    struct SstSource {
        Real productionK = 0, destructionK = 0, productionOmega = 0, destructionOmega = 0;
    };
    CHMT_HD inline SstSource sstSources(const GasPrimitive& w, const SstPrimitive& s, const GasGradient& g,
        Vec3 gradK, Vec3 gradOmega, Real volume, const PhysicsConfig& p) {
        const auto c = sstCoefficients(p.sst);
        const auto inv = velocityInvariants(g);
        const Real mass = w.rho*volume;
        SstSource r;
        r.productionK = mass*maxValue(0, ugkwp::sstKProduction(s.k, s.omega, s.eddyViscosity,
            inv.productionPerNut, c));
        r.destructionK = c.betaStar*s.omega;
        const Real dilK = -(2.0/3)*inv.divergence;
        if (dilK>0)r.productionK += mass*dilK*s.k;
        else r.destructionK -= dilK;
        const Real gamma = ugkwp::sstGamma(s.blend1, c), beta = ugkwp::sstBeta(s.blend1, c);
        r.productionOmega = mass*gamma*minValue(inv.productionPerNut, ugkwp::sstOmegaProductionLimit(s.omega,
            inv.strain2, s.blend2, c));
        r.destructionOmega = beta*s.omega;
        const Real cross = (1-s.blend1)*ugkwp::sstCrossDiffusion(s.omega, dot(gradK, gradOmega),
            c)-(2.0/3)*gamma*inv.divergence*s.omega;
        if (cross >= 0)r.productionOmega += mass*cross;
        else r.destructionOmega -= cross/s.omega;
        return r;
    }
    // Exact frozen-midpoint linear destruction: dq/dt = source - decay*q.
    // Loss is the ACTUAL integrated destruction, for the independent k budget.
    CHMT_HD inline Real exponentialDestruction(Real base, Real source, Real decay, Real dt, Real& loss) {
        if (!finite(base) || !finite(source) || !finite(decay) || !finite(dt) || base<0 || decay<0 || dt<0) {
            loss = undefinedValue();
            return undefinedValue();
        }
        const Real z = decay*dt;
        Real phi = dt, oneMinus = 0;
        if (z>0) {
            oneMinus = -::expm1(-z);
            phi = dt*(oneMinus/z);
        }
        const Real result = base*(1-oneMinus)+source*phi;
        loss = base*oneMinus+source*(dt-phi);
        return result;
    }
    CHMT_HD inline SstQ sstTransportFlux(Real massFlux, const SstPrimitive& l, const SstPrimitive& r,
        Vec3 gradientK, Vec3 gradientOmega, Vec3 area, Real rhoL, Real rhoR, const PhysicsConfig& p,
        bool lowReWall = false) {
        const auto& up = massFlux >= 0?l:r;
        const Real blend = .5*(l.blend1+r.blend1);
        const auto c = sstCoefficients(p.sst);
        const Real mut = lowReWall?0:.5*(rhoL*l.eddyViscosity+rhoR*r.eddyViscosity);
        return {
            massFlux*up.k-(p.gasViscosity+ugkwp::sstAlphaK(blend, c)*mut)*dot(gradientK, area),
                massFlux*up.omega-(p.gasViscosity+ugkwp::sstAlphaOmega(blend, c)*mut)*dot(gradientOmega, area)
        };
    }
}
// namespace chmt
#endif
