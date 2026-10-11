#ifndef CHMT_MATERIALS_STENCILMATH_H
#define CHMT_MATERIALS_STENCILMATH_H
#include "core/Types.H"
namespace chmt {
// Material/film-only reconstruction; gas gradients use common/operators.
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
    CHMT_HD inline Real barthJespersen(Real value, Real lower, Real upper, Real increment) {
        if (increment>0)return minValue(1, maxValue(0, (upper-value)/increment));
        if (increment<0)return minValue(1, maxValue(0, (lower-value)/increment));
        return 1;
    }
}
#endif
