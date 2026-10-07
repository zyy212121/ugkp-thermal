#include "gas/GasKernels.H"
#include "gas/SstKernels.H"
#include "gas/AleFlux.H"
#include "gas/KernelSupport.cuh"
#include "coupling/ExchangeLedger.H"
namespace chmt {
    namespace {
        __global__ void recoverKernel(GasView gas, GeometryView geometry, PhysicsConfig physics, bool endpoint) {
            int c = blockIdx.x*blockDim.x+threadIdx.x;
            if (c >= gas.nCells || gasStopped(gas.status))return;
            if (gas.voidFraction && gas.voidFraction[c] != 1) {
                gasDeviceError(gas.status, ErrorCode::Unsupported, c, gas.voidFraction[c], ErrorLocation::Cell);
                return;
            }
            DeviceStatus local;
            const Real volume = endpoint?geometry.newVolume[c]:geometry.evaluationVolume[c];
            if (!recoverGas(gas.q[c], volume, physics, gas.primitive[c], &local, c)) {
                gasDeviceError(gas.status, static_cast<ErrorCode>(local.code), c, local.value,
                    ErrorLocation::Cell);
                return;
            }
            if (!endpoint) {
                for (int k = geometry.cellFaceOffsets[c]; k < geometry.cellFaceOffsets[c+1]; ++k) {
                    const int face = geometry.cellFaces[k];
                    const auto kind = geometry.boundaryKind[face];
                    const bool explicitWall = kind == BoundaryKind::Interface
                        || (physics.enableSst && kind == BoundaryKind::NoSlip);
                    if (!explicitWall)continue;
                    GasPrimitive wall;
                    if (!geometry.boundaryPrimitive
                        || !explicitBoundaryPrimitive(geometry.boundaryPrimitive[face], physics, wall)) {
                        gasDeviceError(gas.status, ErrorCode::PropertyRange, face, 0, ErrorLocation::Face);
                        return;
                    }
                }
            }
        }
        __device__ GasPrimitive boundarySample(int c, int f, GasView gas, GeometryView geometry, Real dt) {
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
                if (prescribed.temperature>0)w.temperature = 2*prescribed.temperature-w.temperature;
            }
            return w;
        }
        __global__ void gradientKernel(GasView gas, GeometryView geometry, PhysicsConfig physics, Real dt) {
            int c = blockIdx.x*blockDim.x+threadIdx.x;
            if (c >= gas.nCells || gasStopped(gas.status))return;
            GasGradient result{};
            const GasPrimitive centre = gas.primitive[c];
            Symmetric3 matrix;
            Vec3 sums[6+Ns]{};
            for (int k = geometry.cellFaceOffsets[c]; k<geometry.cellFaceOffsets[c+1]; ++k) {
                int f = geometry.cellFaces[k];
                if (geometry.boundaryKind[f] == BoundaryKind::Empty)continue;
                int other = neighbourCell(c, f, geometry);
                Vec3 dx;
                GasPrimitive w;
                if (other >= 0) {
                    dx = neighbourDisplacement(c, f, other, geometry);
                    w = gas.primitive[other];
                } else {
                    dx = boundaryStencilDisplacement(geometry.boundaryKind[f],
                        geometry.faceCentre[f]-geometry.cellCentre[c]);
                    w = boundarySample(c, f, gas, geometry, dt);
                }
                Symmetric3 local;
                Vec3 unused;
                addLeastSquares(local, unused, dx, 0);
                matrix.xx += local.xx;
                matrix.xy += local.xy;
                matrix.xz += local.xz;
                matrix.yy += local.yy;
                matrix.yz += local.yz;
                matrix.zz += local.zz;
                const Real d2 = dot(dx, dx);
                if (!(d2>0)) {
                    gasDeviceError(gas.status, ErrorCode::InvalidInput, f, d2, ErrorLocation::Face);
                    return;
                }
                for (int j = 0; j<6+Ns; ++j) {
                    Real value = primitiveComponent(w, j);
                    sums[j] += dx*((value-primitiveComponent(centre, j))/d2);
                }
            }
            for (int j = 0; j<6+Ns; ++j)gradientComponent(result, j) = solveLeastSquares(matrix, sums[j]);
            // Physical gradients remain unlimited for viscosity and SST. Each face
            // applies a common neighbour-bound/admissibility factor for MUSCL only.
            (void)physics;
            gas.gradient[c] = result;
        }
        __device__ Real reconstructionScale(int c, int face, GasView gas, GeometryView geometry,
            const PhysicsConfig& physics, Real dt) {
            if (physics.spatialOrder != 2)return 0;
            if (physics.reconstruction == ReconstructionMode::SmoothVerification)return 1;
            Real scale = 1;
            const auto& w = gas.primitive[c];
            Vec3 dx = geometry.faceCentre[face]-geometry.cellCentre[c];
            for (int j = 0; j<6+Ns; ++j) {
                Real lo = primitiveComponent(w, j), hi = lo;
                for (int k = geometry.cellFaceOffsets[c]; k<geometry.cellFaceOffsets[c+1]; ++k) {
                    int f = geometry.cellFaces[k];
                    if (geometry.boundaryKind[f] == BoundaryKind::Empty)continue;
                    int other = neighbourCell(c, f, geometry);
                    GasPrimitive n = other >= 0?gas.primitive[other]:boundarySample(c, f, gas, geometry, dt);
                    Real v = primitiveComponent(n, j);
                    lo = minValue(lo, v);
                    hi = maxValue(hi, v);
                }
                scale = minValue(scale, barthJespersen(primitiveComponent(w, j), lo, hi,
                    dot(gradientComponent(gas.gradient[c], j), dx)));
            }
            return scale;
        }
        __device__ bool faceState(int c, int f, Vec3 position, GasView gas, GeometryView geometry,
            const PhysicsConfig& physics, Real dt, GasPrimitive& w) {
            Real theta = 0;
            return reconstructAdmissible(gas.primitive[c], gas.gradient[c], position-geometry.cellCentre[c],
                reconstructionScale(c, f, gas, geometry, physics, dt), physics, w, theta);
        }
        __device__ GasGradient faceGradient(int l, int r, int f, GasView gas, GeometryView geometry,
            const GasPrimitive& target) {
            GasGradient grad = gas.gradient[l];
            Vec3 dx;
            if (r >= 0) {
                dx = neighbourDisplacement(l, f, r, geometry);
                for (int j = 0; j<6+Ns; ++j)gradientComponent(grad, j) = (gradientComponent(grad,
                    j)+gradientComponent(gas.gradient[r], j))*.5;
            } else dx = geometry.faceCentre[f]-geometry.cellCentre[l];
            for (int j = 0; j<6+Ns; ++j) {
                Real difference = primitiveComponent(r >= 0?gas.primitive[r]:target,
                    j)-primitiveComponent(gas.primitive[l], j);
                gradientComponent(grad, j) = correctedGradient(gradientComponent(grad, j), dx, difference);
            }
            return grad;
        }
        __global__ void faceKernel(GasView gas, GeometryView geometry, PacketView packets,
            PhysicsConfig physics, Real dt) {
            int f = blockIdx.x*blockDim.x+threadIdx.x;
            if (f >= gas.nFaces || gasStopped(gas.status))return;
            auto kind = geometry.boundaryKind[f];
            int l = geometry.owner[f], r = geometry.neighbour[f];
            if (kind == BoundaryKind::Periodic) {
                int partner = geometry.periodicPartner[f];
                if (f>partner)return;
                r = geometry.owner[partner];
            }
            if (kind == BoundaryKind::Empty) {
                gas.faceFlux[f] = GasQ{};
                return;
            }
            if (kind == BoundaryKind::Interface) {
                GasQ total{};
                int found = 0;
                for (int k = 0; k<packets.count; ++k) {
                    const auto& p = packets.packets[k];
                    if (!(requiredConsumers(p.kind)&ConsumeGas) || p.kind == ExchangeKind::ParticleGas
                        || p.face != geometry.faceIds[f])continue;
                    if (p.gasCell != l || p.geometry != geometry.geometryVersion || (p.consumerMask&ConsumeGas)) {
                        gasDeviceError(gas.status, ErrorCode::Packet, f, p.gasCell, ErrorLocation::Face);
                        return;
                    }
                    DeviceStatus local;
                    if (!validatePacketMath(p, physics.tolerances, &local)) {
                        gasDeviceError(gas.status, ErrorCode::Packet, f, local.value, ErrorLocation::Face);
                        return;
                    }
                    total += packetDelta(p).gas;
                    found++;
                }
                if (!found) {
                    gasDeviceError(gas.status, ErrorCode::Packet, f, 0, ErrorLocation::Face);
                    return;
                }
                gas.faceFlux[f] = total*(-1/dt);
                return;
            }
            GasPrimitive left, right;
            const Vec3 A = geometry.areaVector[f];
            const Real area = mag(A), meshRate = geometry.sweptVolume[f]/dt;
            if (!(area>0)) {
                gasDeviceError(gas.status, ErrorCode::InvalidInput, f, area, ErrorLocation::Face);
                return;
            }
            if (!faceState(l, f, geometry.faceCentre[f], gas, geometry, physics, dt, left)) {
                gasDeviceError(gas.status, ErrorCode::PropertyRange, f, 0, ErrorLocation::Face);
                return;
            }
            GasQ flux;
            Real nut = physics.enableSst?gas.sst.primitive[l].eddyViscosity:0,
                k = physics.enableSst?gas.sst.primitive[l].k:0;
            if (kind == BoundaryKind::Slip || kind == BoundaryKind::NoSlip) {
                const Vec3 n = A/area;
                const GasPrimitive prescribed = geometry.boundaryPrimitive?geometry.boundaryPrimitive[f]:GasPrimitive{};
                const Vec3 wall = prescribed.velocity+n*(meshRate/area-dot(prescribed.velocity, n));
                // Riemann pressure includes normal velocity mismatch, but the accepted
                // impermeable-wall mass/species flux is identically zero.
                const Real relative = dot(left.velocity, n)-meshRate/area;
                const Real pressure = left.pressure+left.rho*relative*(left.soundSpeed+absValue(relative));
                if (!(pressure>0)) {
                    gasDeviceError(gas.status, ErrorCode::PropertyRange, f, pressure, ErrorLocation::Face);
                    return;
                }
                flux.momentum = A*pressure;
                flux.energy = pressure*meshRate;
                if (kind == BoundaryKind::NoSlip) {
                    right = gas.primitive[l];
                    right.velocity = wall;
                    if (prescribed.temperature>0)right.temperature = prescribed.temperature;
                    GasGradient grad = faceGradient(l, -1, f, gas, geometry, right);
                    for (int s = 0; s<Ns; ++s)grad.Y[s] = {};
                    if (prescribed.temperature <= 0)grad.temperature = {};
                    GasQ viscous = diffusiveFlux(right, grad, A, physics, nut, 0, physics.enableSst);
                    flux += viscous;
                }
            } else {
                if (r >= 0) {
                    Vec3 pos = geometry.faceCentre[f];
                    if (kind == BoundaryKind::Periodic)pos = geometry.faceCentre[geometry.periodicPartner[f]];
                    int rf = kind == BoundaryKind::Periodic?geometry.periodicPartner[f]:f;
                    if (!faceState(r, rf, pos, gas, geometry, physics, dt, right)) {
                        gasDeviceError(gas.status, ErrorCode::PropertyRange, f, 0, ErrorLocation::Face);
                        return;
                    }
                    if (physics.enableSst) {
                        nut = .5*(nut+gas.sst.primitive[r].eddyViscosity);
                        k = .5*(k+gas.sst.primitive[r].k);
                    }
                } else if (kind == BoundaryKind::Inlet) {
                    if (!geometry.boundaryPrimitive) {
                        gasDeviceError(gas.status, ErrorCode::InvalidInput, f, 0, ErrorLocation::Face);
                        return;
                    }
                    right = geometry.boundaryPrimitive[f];
                } else if (kind == BoundaryKind::Outlet) {
                    right = left;
                    if (geometry.boundaryPrimitive && geometry.boundaryPrimitive[f].pressure>0) {
                        Real R = 0;
                        for (int s = 0; s<Ns; ++s)R += right.Y[s]*physics.species[s].R;
                        right.rho = geometry.boundaryPrimitive[f].pressure/(R*right.temperature);
                    }
                } else {
                    gasDeviceError(gas.status, ErrorCode::Unsupported, f, static_cast<int>(kind),
                        ErrorLocation::Face);
                    return;
                }
                if (!aleRusanov(left, right, A, meshRate, physics, flux)) {
                    gasDeviceError(gas.status, ErrorCode::PropertyRange, f, 0, ErrorLocation::Face);
                    return;
                }
                GasPrimitive mean = left;
                mean.rho = .5*(left.rho+right.rho);
                mean.velocity = (left.velocity+right.velocity)*.5;
                mean.temperature = .5*(left.temperature+right.temperature);
                for (int s = 0; s<Ns; ++s)mean.Y[s] = .5*(left.Y[s]+right.Y[s]);
                GasGradient grad = faceGradient(l, r, f, gas, geometry, right);
                if (kind == BoundaryKind::Outlet)grad = GasGradient{};
                flux += diffusiveFlux(mean, grad, A, physics, nut, k);
            }
            if (!finite(flux.energy) || !finite(flux.mass) || !finite(flux.momentum)) {
                gasDeviceError(gas.status, ErrorCode::Nonfinite, f, flux.energy, ErrorLocation::Face);
                return;
            }
            gas.faceFlux[f] = flux;
            if (kind == BoundaryKind::Periodic)gas.faceFlux[geometry.periodicPartner[f]] = flux*(-1);
        }
        __global__ void residualKernel(GasView gas, GeometryView geometry, PacketView packets,
            PhysicsConfig physics, Real dt) {
            int c = blockIdx.x*blockDim.x+threadIdx.x;
            if (c >= gas.nCells || gasStopped(gas.status))return;
            GasQ rhs{};
            for (int k = geometry.cellFaceOffsets[c]; k < geometry.cellFaceOffsets[c+1]; ++k) {
                rhs += gas.faceFlux[geometry.cellFaces[k]]*(-geometry.cellFaceSigns[k]);
            }
            rhs.momentum += physics.gravity*gas.q[c].mass;
            rhs.energy += dot(physics.gravity, gas.q[c].momentum);
            // Particle packets are volumetric exchanges; interface packets have already
            // replaced the boundary flux and MUST NOT be injected a second time here.
            for (int k = 0; k<packets.count; ++k) {
                const auto& p = packets.packets[k];
                if (p.kind != ExchangeKind::ParticleGas || p.gasCell != c)continue;
                DeviceStatus local;
                if (p.geometry != geometry.geometryVersion || (p.consumerMask&ConsumeGas)
                    || !validatePacketMath(p, physics.tolerances, &local)) {
                    gasDeviceError(gas.status, ErrorCode::Packet, c, p.mass, ErrorLocation::Cell);
                    return;
                }
                rhs += packetDelta(p).gas*(1/dt);
            }
            gas.rhs[c] = rhs;
        }
        __global__ void updateKernel(const GasQ* base, GasView gas, GeometryView geometry, Real dt,
            DeviceStatus* status) {
            int c = blockIdx.x*blockDim.x+threadIdx.x;
            if (c >= gas.nCells || gasStopped(status))return;
            GasQ q = base[c]+gas.rhs[c]*dt;
            if (!finite(q.mass) || q.mass <= 0 || !finite(q.energy) || !finite(q.momentum)
                || !(geometry.newVolume[c]>0)) {
                gasDeviceError(status, ErrorCode::Inventory, c, q.mass, ErrorLocation::Cell);
                return;
            }
            for (int s = 0; s<Ns; ++s) {
                if (!finite(q.species[s]) || q.species[s]<0) {
                    gasDeviceError(status, ErrorCode::Inventory, c, q.species[s], ErrorLocation::Cell);
                    return;
                }
            }
            // Composition closure and caloric bounds use configured tolerances in
            // launchGasValidate; this signature intentionally has no PhysicsConfig.
            gas.q[c] = q;
        }
        bool validViews(GasView v, GeometryView g) {
            return v.nCells >= 0 && v.nFaces >= 0 && (v.nCells>0 || v.nFaces == 0) && v.nCells == g.nCells
                && v.nFaces == g.nFaces && v.status && (v.nCells == 0 || (v.q && v.rhs && v.primitive
                && v.gradient && g.cellCentre && g.evaluationVolume && g.newVolume && g.cellFaceOffsets
                && g.cellFaces && g.cellFaceSigns)) && (v.nFaces == 0 || (v.faceFlux && g.owner && g.neighbour
                && g.periodicPartner && g.faceCentre && g.areaVector && g.sweptVolume && g.boundaryKind
                && g.faceIds));
        }
    }
    int launchGasMeanPrepare(GasView v, GeometryView g, const PhysicsConfig& p, double dt, void* stream) {
        if (!validViews(v, g) || !finite(dt) || dt <= 0 || (p.enableParticles && v.nCells
            && !v.voidFraction))return static_cast<int>(cudaErrorInvalidValue);
        if (!v.nCells)return 0;
        auto s = static_cast<cudaStream_t>(stream);
        recoverKernel<<<(v.nCells+127)/128, 128, 0, s>>>(v, g, p, false);
        if (int e = launchError())return e;
        gradientKernel<<<(v.nCells+127)/128, 128, 0, s>>>(v, g, p, dt);
        if (int e = launchError())return e;
        return 0;
    }
    int launchGasPrepare(GasView v, GeometryView g, const PhysicsConfig& p, double dt, void* stream) {
        if (int e=launchGasMeanPrepare(v,g,p,dt,stream))return e;
        return p.enableSst?launchSstPrepare(v, g, p, stream):0;
    }
    int launchGasRhs(GasView v, GeometryView g, PacketView packets, const PhysicsConfig& p, double dt,
        void* stream) {
        if (packets.count<0 || (packets.count && !packets.packets))return static_cast<int>(cudaErrorInvalidValue);
        // Re-preparing is idempotent: no inventory, wall-omega projection or budget
        // changes occur. It also prevents stale reconstruction after a corrector.
        if (int e = launchGasPrepare(v, g, p, dt, stream))return e;
        if (!v.nCells)return 0;
        auto s = static_cast<cudaStream_t>(stream);
        if (v.nFaces) {
            faceKernel<<<(v.nFaces+127)/128, 128, 0, s>>>(v, g, packets, p, dt);
            if (int e = launchError())return e;
        }
        residualKernel<<<(v.nCells+127)/128, 128, 0, s>>>(v, g, packets, p, dt);
        if (int e = launchError())return e;
        return p.enableSst?launchSstRhs(v, g, p, dt, stream):0;
    }
    int launchGasUpdate(const GasQ* base, GasView v, const GeometryView& g, double dt, DeviceStatus* status,
        void* stream) {
        if (v.nCells<0 || v.nCells != g.nCells || !finite(dt) || dt <= 0 || !status || (v.nCells && (!base
            || !v.q || !v.rhs || !g.newVolume)))return static_cast<int>(cudaErrorInvalidValue);
        if (!v.nCells)return 0;
        updateKernel<<<(v.nCells+127)/128, 128, 0, static_cast<cudaStream_t>(stream)>>>(base, v, g, dt, status);
        return launchError();
    }
    int launchGasValidate(GasView v, GeometryView g, const PhysicsConfig& p, void* stream) {
        if (!validViews(v, g))return static_cast<int>(cudaErrorInvalidValue);
        if (!v.nCells)return 0;
        recoverKernel<<<(v.nCells+127)/128, 128, 0, static_cast<cudaStream_t>(stream)>>>(v, g, p, true);
        return launchError();
    }
}
// namespace chmt
