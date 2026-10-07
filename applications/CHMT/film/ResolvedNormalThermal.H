#ifndef CHMT_FILM_RESOLVEDNORMALTHERMAL_H
#define CHMT_FILM_RESOLVEDNORMALTHERMAL_H
#include "gpu/DeviceViews.H"
#include "film/FilmMath.H"
namespace chmt {
CHMT_HD inline bool recoverNormalEnthalpy(Real mass, Real H, Real volume, Real pressure,
    const PhysicsConfig& p, Real& temperature, Real& liquidFraction) {
    const int c=p.material.phaseCondensed;
    if (c<0 || c>=Nc || mass<=0 || volume<=0 || !finite(H) || !finite(pressure)) return false;
    const CondensedThermo& solid=p.condensed[c];
    if (!validCondensed(solid) || !validCondensed(p.liquid)
        || !finite(solid.conductivity) || solid.conductivity<0
        || !finite(p.liquid.conductivity) || p.liquid.conductivity<0
        || !closeEnough(solid.rho,p.liquid.rho,0,p.tolerances.relativeMass)
        || !closeEnough(mass,volume*solid.rho,p.tolerances.absoluteMass,p.tolerances.relativeMass)) return false;
    const Real Tm=p.meltTemperature;
    if (Tm<solid.Tmin || Tm>solid.Tmax || Tm<p.liquid.Tmin || Tm>p.liquid.Tmax) return false;
    const Real hs=condensedE(solid,Tm)+pressure/solid.rho;
    const Real hl=liquidH(p,Tm,pressure);
    if (!(hl>hs)) return false;
    const Real h=H/mass;
    Real T=Tm, fraction=0;
    if (h<hs) {
        if (!invertCaloric(solid.e0+pressure/solid.rho,solid.cp0,solid.cp1,h,
            solid.Tmin,Tm,p.tolerances,T)) return false;
    } else if (h<=hl) {
        fraction=(h-hs)/(hl-hs);
    } else {
        fraction=1;
        if (!invertCaloric(p.liquid.e0+(pressure-p.liquidReferencePressure)/p.liquid.rho,
            p.liquid.cp0,p.liquid.cp1,h,Tm,p.liquid.Tmax,p.tolerances,T)) return false;
    }
    temperature=T; liquidFraction=fraction; return true;
}
// Initial finite-volume cut-cell diagnostic. This preserves supplied phase
// inventory; it does not substitute an analytic future temperature or front.
CHMT_HD inline bool recoverNormalSensibleAverage(Real mass, Real H, Real fraction,
    Real pressure, const PhysicsConfig& p, Real& temperature) {
    const int c=p.material.phaseCondensed;
    if (c<0 || c>=Nc || mass<=0 || fraction<0 || fraction>1) return false;
    const CondensedThermo& s=p.condensed[c];
    const Real a0=(1-fraction)*(s.e0+pressure/s.rho)
        +fraction*(p.liquid.e0+(pressure-p.liquidReferencePressure)/p.liquid.rho);
    return invertCaloric(a0,(1-fraction)*s.cp0+fraction*p.liquid.cp0,
        (1-fraction)*s.cp1+fraction*p.liquid.cp1,H/mass,
        maxValue(s.Tmin,p.liquid.Tmin),minValue(s.Tmax,p.liquid.Tmax),p.tolerances,temperature);
}
CHMT_HD inline Real normalConductivePower(Real Ta, Real Tb, Real da, Real db,
    Real ka, Real kb, Real area) {
    return area*seriesConductance(ka,kb,da,db)*(Ta-Tb);
}
int launchResolvedNormalThermalRhs(NormalThermalView, const PhysicsConfig&, Real stageDt,
    DeviceStatus*, void* stream);
int launchNormalThermalValidate(NormalThermalView, const PhysicsConfig&, DeviceStatus*, void* stream);
} // namespace chmt
#endif
