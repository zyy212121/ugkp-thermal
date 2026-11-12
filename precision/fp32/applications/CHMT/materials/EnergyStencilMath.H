#ifndef CHMT_MATERIALS_ENERGY_STENCIL_MATH_H
#define CHMT_MATERIALS_ENERGY_STENCIL_MATH_H
#include "core/Types.H"
namespace chmt { namespace materialEnergyStencil {
using Scalar=EnergyReal;
struct Vec3 {Scalar x=0,y=0,z=0;};
CHMT_HD inline Vec3 operator+(Vec3 a,Vec3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
CHMT_HD inline Vec3 operator*(Vec3 a,Scalar b){return {a.x*b,a.y*b,a.z*b};}
CHMT_HD inline Vec3& operator+=(Vec3& a,Vec3 b){a=a+b;return a;}
CHMT_HD inline Scalar dot(Vec3 a,Vec3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
CHMT_HD inline bool finite(Scalar x){return std::isfinite(x);}
CHMT_HD inline bool finite(Vec3 a){return finite(a.x)&&finite(a.y)&&finite(a.z);}
CHMT_HD inline Scalar absValue(Scalar a){return a<0?-a:a;}
CHMT_HD inline Scalar minValue(Scalar a,Scalar b){return a<b?a:b;}
CHMT_HD inline Scalar maxValue(Scalar a,Scalar b){return a>b?a:b;}
CHMT_HD inline Scalar undefinedValue(){return std::numeric_limits<Scalar>::quiet_NaN();}
CHMT_HD inline Vec3 localVector(chmt::Vec3 v){return {Scalar(v.x),Scalar(v.y),Scalar(v.z)};}
// Material/film-only reconstruction; gas gradients use common/operators.
    struct Symmetric3 {
        Scalar xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
    };
    CHMT_HD inline void addLeastSquares(Symmetric3& m, Vec3& b, Vec3 dx, Scalar difference) {
        const Scalar d2 = dot(dx, dx);
        if (!finite(d2)||!finite(difference)) { b={undefinedValue(),undefinedValue(),undefinedValue()}; return; }
        if (!(d2>0))return;
        const Scalar w = 1/d2;
        m.xx += w*dx.x*dx.x;
        m.xy += w*dx.x*dx.y;
        m.xz += w*dx.x*dx.z;
        m.yy += w*dx.y*dx.y;
        m.yz += w*dx.y*dx.z;
        m.zz += w*dx.z*dx.z;
        b += dx*(w*difference);
    }
    CHMT_HD inline Vec3 solveLeastSquares(const Symmetric3& m, Vec3 rhs,int maximumIterations=32) {
        // Symmetric Jacobi eigensolve gives the Moore-Penrose solution in the
        // resolved subspace. Cofactor inversion is unstable on oblique 1-D/2-D
        // stencils even when an epsilon is added to the diagonal.
        Scalar a[3][3] = {{m.xx, m.xy, m.xz}, {m.xy, m.yy, m.yz}, {m.xz, m.yz, m.zz}};
        Scalar vectors[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        const Scalar scale = maxValue(m.xx, maxValue(m.yy, m.zz));
        if (!finite(m.xx)||!finite(m.xy)||!finite(m.xz)||!finite(m.yy)||!finite(m.yz)||!finite(m.zz)||!finite(rhs))
            return {undefinedValue(),undefinedValue(),undefinedValue()};
        if (!(scale>0))return {};
        const Scalar roundoff = 8*std::numeric_limits<Scalar>::epsilon()*scale;
        for (int iteration = 0; iteration<maximumIterations; ++iteration) {
            int p = 0, q = 1;
            if (absValue(a[0][2])>absValue(a[p][q])) {
                p = 0;
                q = 2;
            }
            if (absValue(a[1][2])>absValue(a[p][q])) {
                p = 1;
                q = 2;
            }
            const Scalar off = a[p][q];
            if (absValue(off) <= roundoff)break;
            const Scalar tau = (a[q][q]-a[p][p])/(2*off);
            const Scalar t = (tau >= 0?1:-1)/(absValue(tau)+std::sqrt(1+tau*tau));
            const Scalar c = 1/std::sqrt(1+t*t), sn = t*c;
            a[p][p] -= t*off;
            a[q][q] += t*off;
            a[p][q] = a[q][p] = 0;
            for (int k = 0; k<3; ++k)if (k != p && k != q) {
                const Scalar kp = a[k][p], kq = a[k][q];
                a[k][p] = a[p][k] = c*kp-sn*kq;
                a[k][q] = a[q][k] = sn*kp+c*kq;
            }
            for (int k = 0; k<3; ++k) {
                const Scalar kp = vectors[k][p], kq = vectors[k][q];
                vectors[k][p] = c*kp-sn*kq;
                vectors[k][q] = sn*kp+c*kq;
            }
        }
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)if(!finite(a[i][j])||!finite(vectors[i][j]))
            return {undefinedValue(),undefinedValue(),undefinedValue()};
        const Scalar largest = maxValue(a[0][0], maxValue(a[1][1], a[2][2]));
        const Scalar cutoff = 64*std::numeric_limits<Scalar>::epsilon()*largest;
        Vec3 solution{};
        for (int i = 0; i<3; ++i)if (a[i][i]>cutoff) {
            const Vec3 direction = {vectors[0][i], vectors[1][i], vectors[2][i]};
            solution += direction*(dot(direction, rhs)/a[i][i]);
        }
        return solution;
    }
    CHMT_HD inline Scalar barthJespersen(Scalar value, Scalar lower, Scalar upper, Scalar increment) {
        if (increment>0)return minValue(1, maxValue(0, (upper-value)/increment));
        if (increment<0)return minValue(1, maxValue(0, (lower-value)/increment));
        return 1;
    }

CHMT_HD inline Scalar limiter(Scalar value,Scalar lower,Scalar upper,Scalar increment){
    const Scalar scale=maxValue(absValue(value),maxValue(absValue(lower),absValue(upper)));
    const Scalar roundoff=64*std::numeric_limits<Scalar>::epsilon()*scale;
    if(upper-lower<=roundoff&&absValue(increment)<=roundoff)return 1;
    return barthJespersen(value,lower,upper,increment);
}
}}
#endif
