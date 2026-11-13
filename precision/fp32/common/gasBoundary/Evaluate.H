#pragma once
#include "Definition.H"
namespace ugkp { namespace boundary {
template<class R> struct PrimitiveT {R rho,ux,uy,uz,p,T;};
using Primitive=PrimitiveT<double>;
// Eliminate boundary psi=1/(R*T) and rho=p/(R*T) algebraically. This
// satisfies the selected OpenFOAM total-condition equations simultaneously,
// without a lagged host psi field or an extra boundary iteration.
template<class R>
UGKP_RESIDENT_BOUNDARY_HD inline bool totalState(const Face& f,const R* values,
    const PrimitiveT<R>& owner,R gasConstant,R nx,R ny,R nz,R meshNormal,PrimitiveT<R>& out) {
    out=owner;
    const R un=owner.ux*nx+owner.uy*ny+owner.uz*nz;
    const bool inflow=un-meshNormal<0;
    if(f.uMode==int(Velocity::Fixed)||(f.uMode==int(Velocity::InletOutlet)&&inflow)) {
        out.ux=values[Ux];out.uy=values[Uy];out.uz=values[Uz];
    } else if(f.uMode==int(Velocity::PressureInletOutlet)&&inflow) {
        const R tangentNormal=values[Ux]*nx+values[Uy]*ny+values[Uz]*nz;
        out.ux=un*nx+values[Ux]-tangentNormal*nx;
        out.uy=un*ny+values[Uy]-tangentNormal*ny;
        out.uz=un*nz+values[Uz]-tangentNormal*nz;
    }
    const R kinetic=inflow?R(.5)*(out.ux*out.ux+out.uy*out.uy+out.uz*out.uz):0;
    // OpenFOAM totalPressure subtracts the prescribed tangentialVelocity's
    // kinetic energy (the supplied vector, before normal projection). Total
    // temperature still uses the complete boundary velocity.
    const R pressureKinetic=kinetic-(inflow&&f.uMode==int(Velocity::PressureInletOutlet)
        ?R(.5)*(values[Ux]*values[Ux]+values[Uy]*values[Uy]+values[Uz]*values[Uz]):0);
    if(f.tMode==int(Scalar::Fixed))out.T=values[Temperature];
    if(f.tMode==int(Scalar::Total))
        out.T=values[Temperature]-kinetic*(R(f.temperatureGamma)-1)/(R(f.temperatureGamma)*gasConstant);
    if(!(out.T>0)||!(gasConstant>0))return false;
    if(f.pMode==int(Scalar::Fixed))out.p=values[Pressure];
    if(f.pMode==int(Scalar::Total)) {
        const R psi=1/(gasConstant*out.T);
        if(f.compressiblePressure&&R(f.pressureGamma)>1) {
            const R g=(R(f.pressureGamma)-1)/R(f.pressureGamma);
            out.p=values[Pressure]/std::pow(1+psi*g*pressureKinetic,1/g);
        } else out.p=values[Pressure]/(1+psi*pressureKinetic);
    } else if(f.rhoMode&& !f.pMode)out.p=values[Density]*gasConstant*out.T;
    out.rho=out.p/(gasConstant*out.T);
    return out.p>0&&out.rho>0&&std::isfinite(out.p)&&std::isfinite(out.T)
        &&std::isfinite(out.ux)&&std::isfinite(out.uy)&&std::isfinite(out.uz);
}
}}
