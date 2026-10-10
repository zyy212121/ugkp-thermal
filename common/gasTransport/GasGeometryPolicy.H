#pragma once
#include <cmath>
#include <limits>
#include "GasModel.H"

namespace ugkwp
{

// Non-owning geometry for ONE attempted Euler stage. The caller must bind new
// metrics after a rejected/reduced interval; full-window sweeps cannot be reused.
// Cell volumes are geometric volumes [m^3]. Face sweeps [m^3] are positive in the
// owner-outward direction, matching the area vector used by the flux operator.
// Evaluation areas/centres remain in the ordinary shared gas geometry arrays.
template<class Real>
struct GasGeometryState
{
    bool enabled = false;
    const Real* oldVolume = nullptr;
    const Real* newVolume = nullptr;
    const Real* faceSweptVolume = nullptr;
    Real interval = Real(0);
    // Enabled views must explicitly bind configured GCL tolerances. Invalid
    // sentinels prevent silently replacing caller settings with epsilon defaults.
    Real absoluteGeometryTolerance = Real(-1); // swept volume [m^3]
    Real relativeGeometryTolerance = Real(-1); // dimensionless
};

template<class Real>
struct GasFaceFrame
{
    Real nx = Real(0), ny = Real(0), nz = Real(0);
    Real area = Real(0);
    Real wx = Real(0), wy = Real(0), wz = Real(0);
    Real meshVolumeRate = Real(0); // signed sweep / stage interval [m^3/s]
};

enum class GasConservativeStorage : unsigned char { Density, Integral };

namespace gasGeometryDetail
{
template<class Real>
UGKWP_GAS_HD inline bool finite(Real value) { return std::isfinite(value); }
template<class Real>
UGKWP_GAS_HD inline Real absolute(Real value) { return value < Real(0) ? -value : value; }
template<class Real>
UGKWP_GAS_HD inline Real maximum(Real a, Real b) { return a > b ? a : b; }
template<class Real>
UGKWP_GAS_HD inline Real tolerance() { return Real(128)*std::numeric_limits<Real>::epsilon(); }
template<class Real>
UGKWP_GAS_HD inline bool equal(Real a, Real b, Real relativeTolerance = tolerance<Real>())
{
    return finite(a) && finite(b)
        && absolute(a-b) <= relativeTolerance*maximum(absolute(a),absolute(b));
}
template<class Real>
UGKWP_GAS_HD inline bool validFrame(const GasFaceFrame<Real>& f)
{
    if (!finite(f.nx) || !finite(f.ny) || !finite(f.nz)
        || !finite(f.area) || f.area <= Real(0)
        || !finite(f.wx) || !finite(f.wy) || !finite(f.wz)
        || !finite(f.meshVolumeRate)) return false;
    const Real normalSquared = f.nx*f.nx+f.ny*f.ny+f.nz*f.nz;
    const Real normalRate = (f.wx*f.nx+f.wy*f.ny+f.wz*f.nz)*f.area;
    // A tangential frame may nearly cancel in w dot n. Its roundoff scale is
    // the individual products, not the small residual normal mesh rate.
    const Real rateScale = maximum(absolute(f.meshVolumeRate), f.area*
        (absolute(f.wx*f.nx)+absolute(f.wy*f.ny)+absolute(f.wz*f.nz)));
    return equal(normalSquared,Real(1)) && finite(normalRate) && finite(rateScale)
        && absolute(normalRate-f.meshVolumeRate) <= tolerance<Real>()*rateScale;
}
template<class Real, int Ns>
UGKWP_GAS_HD inline bool validConservative(const Real* state)
{
    static_assert(Ns >= 0, "species count must not be negative");
    if (!state || !finite(state[0]) || state[0] <= Real(0)) return false;
    for (int k=1;k<5+Ns;++k) if (!finite(state[k])) return false;
    Real speciesSum = Real(0);
    for (int k=0;k<Ns;++k)
    {
        if (state[5+k] < Real(0)) return false;
        speciesSum += state[5+k];
    }
    // Total energy can legitimately be negative because of formation offsets.
    // Temperature/pressure/table admissibility is checked by shared EOS recovery.
    return Ns == 0 || equal(speciesSum,state[0]);
}
UGKWP_GAS_HD inline bool validStorage(GasConservativeStorage storage)
{
    return storage == GasConservativeStorage::Density
        || storage == GasConservativeStorage::Integral;
}
} // namespace gasGeometryDetail

// Canonical frame uses normal motion only. Optional tangential velocity is a
// coordinate-frame choice; it must not contribute to normal swept volume.
// Reversing both area and sweep preserves physical w and reverses face flux.
// Every fallible helper here publishes outputs only after full validation.
template<class Real>
UGKWP_GAS_HD inline bool makeGasFaceFrame
(
    Real areaX, Real areaY, Real areaZ, Real signedSweep, Real interval,
    GasFaceFrame<Real>& output,
    Real tangentX = Real(0), Real tangentY = Real(0), Real tangentZ = Real(0)
)
{
    using namespace gasGeometryDetail;
    if (!finite(areaX) || !finite(areaY) || !finite(areaZ)
        || !finite(signedSweep) || !finite(interval) || interval <= Real(0)
        || !finite(tangentX) || !finite(tangentY) || !finite(tangentZ)) return false;
    const Real scale = maximum(absolute(areaX),maximum(absolute(areaY),absolute(areaZ)));
    if (scale <= Real(0)) return false;
    const Real ax=areaX/scale, ay=areaY/scale, az=areaZ/scale;
    const Real normalMagnitude = std::sqrt(ax*ax+ay*ay+az*az);
    GasFaceFrame<Real> candidate;
    candidate.nx=ax/normalMagnitude;
    candidate.ny=ay/normalMagnitude;
    candidate.nz=az/normalMagnitude;
    candidate.area=scale*normalMagnitude;
    candidate.meshVolumeRate=signedSweep/interval;
    const Real tangentNormal=tangentX*candidate.nx+tangentY*candidate.ny+tangentZ*candidate.nz;
    const Real tangentScale=absolute(tangentX)+absolute(tangentY)+absolute(tangentZ);
    if (!finite(tangentNormal) || !finite(tangentScale)
        || absolute(tangentNormal) > tolerance<Real>()*tangentScale) return false;
    const Real normalSpeed=candidate.meshVolumeRate/candidate.area;
    candidate.wx=normalSpeed*candidate.nx+tangentX;
    candidate.wy=normalSpeed*candidate.ny+tangentY;
    candidate.wz=normalSpeed*candidate.nz+tangentZ;
    if (!validFrame(candidate)) return false;
    output=candidate;
    return true;
}

// Primitive/Conservative use the existing shared Riemann POD interface:
// {rho,ux,uy,uz,p} and {q[5]}. Templates keep this policy precision-independent
// and avoid making geometry a new Riemann solver or thermodynamic owner.
// Conserved inputs are densities, irrespective of application storage units.
// E' = E - w dot (rho*u) + rho*|w|^2/2; actual formation energy is untouched.
template<class Primitive, class Conservative, class Real>
UGKWP_GAS_HD inline bool shiftGasStateToFaceFrame
(
    const Primitive& primitive, const Conservative& conserved,
    const GasFaceFrame<Real>& frame,
    Primitive& primitiveOutput, Conservative& conservativeOutput
)
{
    using namespace gasGeometryDetail;
    if (!validFrame(frame) || !validConservative<Real,0>(conserved.q)
        || !finite(primitive.rho) || primitive.rho <= Real(0)
        || !finite(primitive.p) || primitive.p <= Real(0)
        || !finite(primitive.ux) || !finite(primitive.uy) || !finite(primitive.uz)
        || !equal(Real(primitive.rho),conserved.q[0])
        || !equal(Real(primitive.rho*primitive.ux),conserved.q[1])
        || !equal(Real(primitive.rho*primitive.uy),conserved.q[2])
        || !equal(Real(primitive.rho*primitive.uz),conserved.q[3])) return false;
    Primitive p=primitive;
    Conservative q=conserved;
    // Preserve legacy fixed-grid values, including signed zeros, bit for bit.
    if (frame.wx != Real(0) || frame.wy != Real(0) || frame.wz != Real(0))
    {
        p.ux-=frame.wx; p.uy-=frame.wy; p.uz-=frame.wz;
        q.q[1]-=conserved.q[0]*frame.wx;
        q.q[2]-=conserved.q[0]*frame.wy;
        q.q[3]-=conserved.q[0]*frame.wz;
        q.q[4]+=(Real(.5)*conserved.q[0]*frame.wx-conserved.q[1])*frame.wx
               +(Real(.5)*conserved.q[0]*frame.wy-conserved.q[2])*frame.wy
               +(Real(.5)*conserved.q[0]*frame.wz-conserved.q[3])*frame.wz;
    }
    if (!finite(p.ux) || !finite(p.uy) || !finite(p.uz)
        || !validConservative<Real,0>(q.q)) return false;
    primitiveOutput=p;
    conservativeOutput=q;
    return true;
}

// Transform the SELECTED compatible shared scheme's moving-frame flux back.
// Fm=Fm', Fp=Fp'+w*Fm', FE=FE'+w dot Fp'+|w|^2*Fm'/2.
// This is already inertial Euler flux minus mesh transport. Do not subtract
// meshVolumeRate*U again. Units remain exactly those of input (per-area or
// area-integrated); scale by face area once at the usual shared flux owner.
// maxSignalSpeed remains the relative-frame normal spectral radius for CFL.
template<class FluxResult, class Real>
UGKWP_GAS_HD inline bool restoreGasFluxFromFaceFrame
(
    const FluxResult& movingFlux, const GasFaceFrame<Real>& frame,
    FluxResult& output
)
{
    using namespace gasGeometryDetail;
    if (!movingFlux.valid || !validFrame(frame)
        || !finite(movingFlux.maxSignalSpeed) || movingFlux.maxSignalSpeed < Real(0)) return false;
    for (int k=0;k<5;++k) if (!finite(movingFlux.flux[k])) return false;
    FluxResult candidate=movingFlux;
    if (frame.wx != Real(0) || frame.wy != Real(0) || frame.wz != Real(0))
    {
        candidate.flux[1]+=frame.wx*movingFlux.flux[0];
        candidate.flux[2]+=frame.wy*movingFlux.flux[0];
        candidate.flux[3]+=frame.wz*movingFlux.flux[0];
        candidate.flux[4]+=frame.wx*movingFlux.flux[1]
            +frame.wy*movingFlux.flux[2]+frame.wz*movingFlux.flux[3]
            +Real(.5)*(frame.wx*frame.wx+frame.wy*frame.wy+frame.wz*frame.wz)*movingFlux.flux[0];
    }
    for (int k=0;k<5;++k) if (!finite(candidate.flux[k])) return false;
    output=candidate;
    return true;
}

// Integral Euler GCL: V_new - V_old = sum(owner-outward swept face volumes).
// Threshold is absoluteVolumeTolerance + relativeTolerance*max(Vold,Vnew),
// matching CHMT's configured geometric-volume comparison. There is no implicit
// epsilon floor and no sweep-magnitude inflation. Defaults are solely for
// standalone math callers; production callers pass both configured values.
// Closure of evaluation area vectors is a separate mesh validation obligation.
template<class Real>
UGKWP_GAS_HD inline bool validateGasEulerGeometry
(
    Real oldVolume, Real newVolume, Real sumOutwardSweeps,
    Real absoluteVolumeTolerance=Real(0),
    Real relativeTolerance=gasGeometryDetail::tolerance<Real>()
)
{
    using namespace gasGeometryDetail;
    if (!finite(oldVolume) || !finite(newVolume) || !finite(sumOutwardSweeps)
        || oldVolume <= Real(0) || newVolume <= Real(0)
        || !finite(absoluteVolumeTolerance) || absoluteVolumeTolerance < Real(0)
        || !finite(relativeTolerance) || relativeTolerance < Real(0)) return false;
    const Real threshold=absoluteVolumeTolerance+relativeTolerance*maximum(oldVolume,newVolume);
    return finite(threshold)
        && absolute((newVolume-oldVolume)-sumOutwardSweeps) <= threshold;
}

// Existing time selectors: 1 Euler, 2 SSPRK2, 3 SSPRK3. Moving higher-order RK
// requires stage-specific base/evaluation/temporary/blended volumes and metric
// weights. No such contract is implied by stage timestamps or these pointers.
UGKWP_GAS_HD inline bool validateGasGeometryTimeIntegrator(bool moving, int selector)
{
    return selector >= 1 && selector <= 3 && (!moving || selector == 1);
}

// q = [mass, momentum xyz, total energy, Ns species inventories] for Integral;
// every component divided by V for Density. Never reinterpret rhoY as kg.
template<class Real, int Ns>
UGKWP_GAS_HD inline bool convertGasConservativeStorage
(
    const Real* input, Real volume, GasConservativeStorage source,
    GasConservativeStorage destination, Real* output
)
{
    using namespace gasGeometryDetail;
    if (!output || !validStorage(source) || !validStorage(destination)
        || !finite(volume) || volume <= Real(0) || !validConservative<Real,Ns>(input)) return false;
    Real candidate[5+Ns];
    for (int k=0;k<5+Ns;++k)
        candidate[k] = source == destination ? input[k]
            : destination == GasConservativeStorage::Integral ? input[k]*volume : input[k]/volume;
    if (!validConservative<Real,Ns>(candidate)) return false;
    for (int k=0;k<5+Ns;++k) output[k]=candidate[k];
    return true;
}

// Pure conservative Euler cell update. outwardFluxSum is the signed outward
// sum of area-integrated physical ALE fluxes [inventory/s], already including
// mesh transport. Storage stays in the selected representation. Whole-trial
// rejection, EOS admissibility and accepted-state commit remain caller-owned.
template<class Real, int Ns>
UGKWP_GAS_HD inline bool advanceGasEulerGeometry
(
    const Real* base, GasConservativeStorage storage,
    Real oldVolume, Real newVolume, Real sumOutwardSweeps, Real interval,
    const Real* outwardFluxSum, Real* output,
    Real absoluteVolumeTolerance=Real(0),
    Real relativeTolerance=gasGeometryDetail::tolerance<Real>()
)
{
    using namespace gasGeometryDetail;
    if (!outwardFluxSum || !output || !finite(interval) || interval <= Real(0)
        || !validateGasEulerGeometry(oldVolume,newVolume,sumOutwardSweeps,
            absoluteVolumeTolerance,relativeTolerance)) return false;
    Real inventory[5+Ns];
    if (!convertGasConservativeStorage<Real,Ns>
        (base,oldVolume,storage,GasConservativeStorage::Integral,inventory)) return false;
    for (int k=0;k<5+Ns;++k)
    {
        if (!finite(outwardFluxSum[k])) return false;
        inventory[k]-=interval*outwardFluxSum[k];
    }
    return convertGasConservativeStorage<Real,Ns>
        (inventory,newVolume,GasConservativeStorage::Integral,storage,output);
}

} // namespace ugkwp
