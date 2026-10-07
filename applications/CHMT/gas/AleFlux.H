#ifndef CHMT_GAS_ALEFLUX_H
#define CHMT_GAS_ALEFLUX_H
#include "materials/Thermodynamics.H"
namespace chmt {
    // Area-integrated fluxes [kg/s, N, W]. Mesh rate is swept volume / stage dt.
    CHMT_HD inline bool explicitBoundaryPrimitive(const GasPrimitive& supplied,
        const PhysicsConfig& physics, GasPrimitive& recovered) {
        if (!recoverGas(conservativeGas(supplied, 1, physics), 1, physics, recovered))return false;
        const Real pressureScale = maxValue(absValue(supplied.pressure), absValue(recovered.pressure));
        const Real tolerance = physics.tolerances.relativeMass + 64*std::numeric_limits<Real>::epsilon();
        return finite(supplied.pressure) && supplied.pressure > 0
            && absValue(supplied.pressure - recovered.pressure) <= tolerance*pressureScale;
    }
    CHMT_HD inline Real primitiveComponent(const GasPrimitive& w, int j) {
        if (j == 0)return w.rho;
        if (j == 1)return w.velocity.x;
        if (j == 2)return w.velocity.y;
        if (j == 3)return w.velocity.z;
        if (j == 4)return w.temperature;
        if (j == 5)return w.pressure;
        return w.Y[j-6];
    }
    CHMT_HD inline Vec3& gradientComponent(GasGradient& g, int j) {
        if (j == 0)return g.rho;
        if (j<4)return g.velocity[j-1];
        if (j == 4)return g.temperature;
        if (j == 5)return g.pressure;
        return g.Y[j-6];
    }
    CHMT_HD inline Vec3 gradientComponent(const GasGradient& g, int j) {
        if (j == 0)return g.rho;
        if (j<4)return g.velocity[j-1];
        if (j == 4)return g.temperature;
        if (j == 5)return g.pressure;
        return g.Y[j-6];
    }
    struct Symmetric3 {
        Real xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
    };
    CHMT_HD inline void addLeastSquares(Symmetric3& m, Vec3& b, Vec3 dx, Real difference) {
        const Real d2 = dot(dx, dx);
        if (!(d2>0))return;
        const Real w = 1/d2;
        m.xx += w*dx.x*dx.x;
        m.xy += w*dx.x*dx.y;
        m.xz += w*dx.x*dx.z;
        m.yy += w*dx.y*dx.y;
        m.yz += w*dx.y*dx.z;
        m.zz += w*dx.z*dx.z;
        b += dx*(w*difference);
    }
    CHMT_HD inline Vec3 solveLeastSquares(const Symmetric3& m, Vec3 rhs) {
        // Symmetric Jacobi eigensolve gives the Moore-Penrose solution in the
        // resolved subspace. Cofactor inversion is unstable on oblique 1-D/2-D
        // stencils even when an epsilon is added to the diagonal.
        Real a[3][3] = {{m.xx, m.xy, m.xz}, {m.xy, m.yy, m.yz}, {m.xz, m.yz, m.zz}};
        Real vectors[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        const Real scale = maxValue(m.xx, maxValue(m.yy, m.zz));
        if (!(scale>0) || !finite(scale) || !finite(rhs))return {};
        const Real roundoff = 8*std::numeric_limits<Real>::epsilon()*scale;
        for (int iteration = 0; iteration<32; ++iteration) {
            int p = 0, q = 1;
            if (absValue(a[0][2])>absValue(a[p][q])) {
                p = 0;
                q = 2;
            }
            if (absValue(a[1][2])>absValue(a[p][q])) {
                p = 1;
                q = 2;
            }
            const Real off = a[p][q];
            if (absValue(off) <= roundoff)break;
            const Real tau = (a[q][q]-a[p][p])/(2*off);
            const Real t = (tau >= 0?1:-1)/(absValue(tau)+::sqrt(1+tau*tau));
            const Real c = 1/::sqrt(1+t*t), sn = t*c;
            a[p][p] -= t*off;
            a[q][q] += t*off;
            a[p][q] = a[q][p] = 0;
            for (int k = 0; k<3; ++k)if (k != p && k != q) {
                const Real kp = a[k][p], kq = a[k][q];
                a[k][p] = a[p][k] = c*kp-sn*kq;
                a[k][q] = a[q][k] = sn*kp+c*kq;
            }
            for (int k = 0; k<3; ++k) {
                const Real kp = vectors[k][p], kq = vectors[k][q];
                vectors[k][p] = c*kp-sn*kq;
                vectors[k][q] = sn*kp+c*kq;
            }
        }
        const Real largest = maxValue(a[0][0], maxValue(a[1][1], a[2][2]));
        const Real cutoff = 64*std::numeric_limits<Real>::epsilon()*largest;
        Vec3 solution{};
        for (int i = 0; i<3; ++i)if (a[i][i]>cutoff) {
            const Vec3 direction = {vectors[0][i], vectors[1][i], vectors[2][i]};
            solution += direction*(dot(direction, rhs)/a[i][i]);
        }
        return solution;
    }
    // An inlet prescription is a face value. Other boundary samples are mirrored
    // ghost values and therefore use twice the cell-to-face displacement.
    CHMT_HD inline Vec3 boundaryStencilDisplacement(BoundaryKind kind, Vec3 cellToFace) {
        return kind == BoundaryKind::Inlet?cellToFace:cellToFace*2;
    }
    CHMT_HD inline Real barthJespersen(Real value, Real lower, Real upper, Real increment) {
        if (increment>0)return minValue(1, maxValue(0, (upper-value)/increment));
        if (increment<0)return minValue(1, maxValue(0, (lower-value)/increment));
        return 1;
    }
    CHMT_HD inline GasPrimitive incrementPrimitive(const GasPrimitive& w, const GasGradient& g, Vec3 dx,
        Real scale) {
        GasPrimitive r = w;
        r.rho += scale*dot(g.rho, dx);
        r.temperature += scale*dot(g.temperature, dx);
        r.velocity.x += scale*dot(g.velocity[0], dx);
        r.velocity.y += scale*dot(g.velocity[1], dx);
        r.velocity.z += scale*dot(g.velocity[2], dx);
        Real sum = 0;
        for (int s = 0; s<Ns; ++s) {
            r.Y[s] += scale*dot(g.Y[s], dx);
            sum += r.Y[s];
        }
        // Only roundoff closure is permitted. No species inventory is clipped.
        if (absValue(sum-1) <= 64*std::numeric_limits<Real>::epsilon())r.Y[Ns-1] += 1-sum;
        return r;
    }
    CHMT_HD inline bool reconstructAdmissible(const GasPrimitive& w, const GasGradient& g, Vec3 dx,
        Real requested, const PhysicsConfig& p, GasPrimitive& out, Real& accepted) {
        if (!finite(requested) || requested<0 || requested>1)return false;
        Real scale = requested;
        for (int i = 0; i<64; ++i) {
            const GasPrimitive trial = incrementPrimitive(w, g, dx, scale);
            if (recoverGas(conservativeGas(trial, 1, p), 1, p, out)) {
                accepted = scale;
                return true;
            }
            scale *= 0.5;
        }
        accepted = 0;
        return recoverGas(conservativeGas(w, 1, p), 1, p, out);
    }
    CHMT_HD inline bool aleRusanov(const GasPrimitive& left, const GasPrimitive& right, Vec3 area,
        Real meshRate, const PhysicsConfig& p, GasQ& result) {
        const Real A = mag(area);
        if (!(A>0) || !finite(A) || !finite(meshRate))return false;
        GasPrimitive l, r;
        const GasQ ul = conservativeGas(left, 1, p), ur = conservativeGas(right, 1, p);
        if (!recoverGas(ul, 1, p, l) || !recoverGas(ur, 1, p, r))return false;
        const Real vl = dot(l.velocity, area), vr = dot(r.velocity, area), wn = meshRate/A;
        GasQ fl = ul*(vl-meshRate), fr = ur*(vr-meshRate);
        fl.momentum += area*l.pressure;
        fr.momentum += area*r.pressure;
        fl.energy += l.pressure*vl;
        fr.energy += r.pressure*vr;
        const Real radius = A*maxValue(absValue(vl/A-wn)+l.soundSpeed, absValue(vr/A-wn)+r.soundSpeed);
        GasQ flux = (fl+fr)*0.5-(ur-ul)*(0.5*radius);
        // All species use the SAME accepted numerical convective mass flux.
        // Their mixture diffusion correction is added separately below.
        const GasPrimitive& up = flux.mass >= 0?l:r;
        for (int s = 0; s<Ns; ++s)flux.species[s] = flux.mass*up.Y[s];
        result = flux;
        return finite(flux.mass) && finite(flux.energy) && finite(flux.momentum);
    }
    CHMT_HD inline Real component(Vec3 v, int i) {
        return i == 0?v.x:(i == 1?v.y:v.z);
    }
    CHMT_HD inline Vec3 viscousTraction(const GasGradient& g, Vec3 area, Real mu, Real turbulentPressure) {
        const Real div = g.velocity[0].x+g.velocity[1].y+g.velocity[2].z;
        Vec3 t{};
        Real values[3]{};
        for (int i = 0; i<3; ++i)for (int j = 0; j<3; ++j) {
            const Real stress = mu*(component(g.velocity[i], j)+component(g.velocity[j],
                i)-(i == j?2*div/3:0))-(i == j?turbulentPressure:0);
            values[i] += stress*component(area, j);
        }
        t = {values[0], values[1], values[2]};
        return t;
    }
    CHMT_HD inline GasQ diffusiveFlux(const GasPrimitive& w, const GasGradient& g, Vec3 area,
        const PhysicsConfig& p, Real nut, Real k, bool lowReWall = false) {
        if (lowReWall) {
            nut = 0;
            k = 0;
        }
        GasQ f{};
        const Real mu = p.gasViscosity+w.rho*nut;
        const Vec3 traction = viscousTraction(g, area, mu, 2*w.rho*k/3);
        f.momentum = -traction;
        Real cp = 0, raw[Ns]{}, sum = 0;
        for (int s = 0; s<Ns; ++s) {
            cp += w.Y[s]*(p.species[s].cp0+p.species[s].cp1*w.temperature);
            raw[s] = -w.rho*(p.gasDiffusivity[s]+(nut>0?nut/p.sst.turbulentSchmidt:0))*dot(g.Y[s], area);
            sum += raw[s];
        }
        Real correction = 0;
        for (int s = 0; s<Ns; ++s) {
            f.species[s] = raw[s]-w.Y[s]*sum;
            correction += f.species[s];
        }
        f.species[Ns-1] -= correction;
        // roundoff-level closure of the analytic correction.
        f.energy = -dot(traction,
            w.velocity)-(p.gasConductivity+(nut>0?w.rho*cp*nut/p.sst.turbulentPrandtl:0))*dot(g.temperature, area);
        for (int s = 0; s<Ns; ++s)f.energy += speciesH(p.species[s], w.temperature)*f.species[s];
        return f;
    }
    CHMT_HD inline Vec3 correctedGradient(Vec3 gradient, Vec3 dx, Real difference) {
        const Real d2 = dot(dx, dx);
        return d2>0?gradient+dx*((difference-dot(gradient, dx))/d2):gradient;
    }
}
// namespace chmt
#endif
