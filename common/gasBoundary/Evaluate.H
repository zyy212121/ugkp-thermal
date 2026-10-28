#pragma once
#include "Definition.H"
namespace ugkp { namespace boundary {
struct Primitive {double rho,ux,uy,uz,p,T;};
// Eliminate boundary psi=1/(R*T) and rho=p/(R*T) algebraically. This
// satisfies the selected OpenFOAM total-condition equations simultaneously,
// without a lagged host psi field or an extra boundary iteration.
UGKP_RESIDENT_BOUNDARY_HD inline bool totalState(const Face& f,const double* values,
    const Primitive& owner,double R,double nx,double ny,double nz,double meshNormal,Primitive& out) {
    out=owner;
    const double un=owner.ux*nx+owner.uy*ny+owner.uz*nz;
    const bool inflow=un-meshNormal<0;
    if(f.uMode==int(Velocity::Fixed)||(f.uMode==int(Velocity::InletOutlet)&&inflow)) {
        out.ux=values[Ux];out.uy=values[Uy];out.uz=values[Uz];
    } else if(f.uMode==int(Velocity::PressureInletOutlet)&&inflow) {
        const double tangentNormal=values[Ux]*nx+values[Uy]*ny+values[Uz]*nz;
        out.ux=un*nx+values[Ux]-tangentNormal*nx;
        out.uy=un*ny+values[Uy]-tangentNormal*ny;
        out.uz=un*nz+values[Uz]-tangentNormal*nz;
    }
    const double kinetic=inflow?.5*(out.ux*out.ux+out.uy*out.uy+out.uz*out.uz):0;
    if(f.tMode==int(Scalar::Fixed))out.T=values[Temperature];
    if(f.tMode==int(Scalar::Total))
        out.T=values[Temperature]-kinetic*(f.temperatureGamma-1)/(f.temperatureGamma*R);
    if(!(out.T>0)||!(R>0))return false;
    if(f.pMode==int(Scalar::Fixed))out.p=values[Pressure];
    if(f.pMode==int(Scalar::Total)) {
        const double psi=1/(R*out.T);
        if(f.compressiblePressure&&f.pressureGamma>1) {
            const double g=(f.pressureGamma-1)/f.pressureGamma;
            out.p=values[Pressure]/::pow(1+psi*g*kinetic,1/g);
        } else out.p=values[Pressure]/(1+psi*kinetic);
    } else if(f.rhoMode&& !f.pMode)out.p=values[Density]*R*out.T;
    out.rho=out.p/(R*out.T);
    return out.p>0&&out.rho>0&&std::isfinite(out.p)&&std::isfinite(out.T)
        &&std::isfinite(out.ux)&&std::isfinite(out.uy)&&std::isfinite(out.uz);
}
}}
