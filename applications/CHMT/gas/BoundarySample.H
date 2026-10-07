#ifndef CHMT_GAS_BOUNDARYSAMPLE_H
#define CHMT_GAS_BOUNDARYSAMPLE_H
#include "gpu/DeviceViews.H"
#include "gas/AleFlux.H"
namespace chmt {
    // Recover a positive EOS trace for SST at the current endpoint. Thermal
    // zeroGradient follows the current owner; it never freezes imported T/rho.
    CHMT_HD inline bool wallThermodynamicTrace(const GasPrimitive& owner,
        const GasPrimitive& prescribed,bool fixed,const PhysicsConfig& p,GasPrimitive& trace) {
        GasPrimitive w=owner;
        if(fixed)w.temperature=prescribed.temperature;
        Real R=0;for(int s=0;s<Ns;++s)R+=w.Y[s]*p.species[s].R;
        if(!(R>0&&w.temperature>0&&w.pressure>0))return false;
        w.rho=w.pressure/(R*w.temperature);w.velocity=prescribed.velocity;
        return recoverGas(conservativeGas(w,1,p),1,p,trace);
    }
    // Only the normal heat flux is prescribed. The stationary adiabatic wall
    // must have zero conductive energy even when stored T differs from owner T.
    CHMT_HD inline GasQ impermeableWallDiffusiveFlux(const GasPrimitive& owner,
        const GasPrimitive& prescribed,Vec3 wall,GasGradient gradient,Vec3 area,
        bool fixed,bool noSlip,const PhysicsConfig& p,Real nut) {
        GasPrimitive state=owner;state.velocity=wall;
        if(fixed)state.temperature=prescribed.temperature;
        for(int s=0;s<Ns;++s)gradient.Y[s]={};
        if(!fixed)gradient.temperature={};
        if(!noSlip)for(int axis=0;axis<3;++axis)gradient.velocity[axis]={};
        return diffusiveFlux(state,gradient,area,p,noSlip?nut:0,0,noSlip&&p.enableSst);
    }

    CHMT_HD inline GasPrimitive boundarySample(int c, int f, GasView gas, GeometryView geometry, Real dt) {
            GasPrimitive w = gas.primitive[c];
            auto kind = geometry.boundaryKind[f];
            const GasPrimitive prescribed = geometry.boundaryPrimitive?geometry.boundaryPrimitive[f]:GasPrimitive{};
            if (kind == BoundaryKind::Inlet)return prescribed;
            if (kind == BoundaryKind::Interface) {
                // Coupler supplies the actual blowing-wall gas state. These
                // are mirrored samples for gradients, not a second BC flux.
                w.rho = 2*prescribed.rho-w.rho;
                w.velocity = prescribed.velocity*2-w.velocity;
                w.temperature = 2*prescribed.temperature-w.temperature;
                w.pressure = 2*prescribed.pressure-w.pressure;
                for (int species = 0; species < Ns; ++species) {
                    w.Y[species] = 2*prescribed.Y[species]-w.Y[species];
                }
                return w;
            }
            if (kind == BoundaryKind::Outlet) {
                if (prescribed.pressure>0)w.pressure = prescribed.pressure;
                return w;
            }
            if (kind == BoundaryKind::NoSlip || kind == BoundaryKind::Slip) {
                const Vec3 n = normalized(geometry.areaVector[f]);
                const Real wn = geometry.sweptVolume[f]/(dt*mag(geometry.areaVector[f]));
                Vec3 wall = prescribed.velocity+n*(wn-dot(prescribed.velocity, n));
                w.velocity = kind == BoundaryKind::Slip?w.velocity-n*(2*dot(w.velocity-wall, n)):wall*2-w.velocity;
                if (fixedTemperature(geometry.thermalBoundary,f))w.temperature = 2*prescribed.temperature-w.temperature;
            }
            return w;
        }
}
#endif
