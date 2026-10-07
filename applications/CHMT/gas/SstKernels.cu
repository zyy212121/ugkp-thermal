#include "gas/SstKernels.H"
#include "gas/SstMath.H"
#include "gas/KernelSupport.cuh"
namespace chmt {
    namespace {
        bool validSst(GasView v, GeometryView g) {
            const auto s = v.sst;
            return v.status && s.nCells == v.nCells && s.nFaces == v.nFaces && (v.nCells == 0 || (s.q && s.rhs
                && s.primitive && s.gradientK && s.gradientOmega && s.production && s.dissipation
                && s.omegaConstraint && (s.wallDistance || g.wallDistance) && v.primitive && v.gradient
                && g.evaluationVolume && g.cellFaceOffsets && g.cellFaces && g.cellFaceSigns))
                && (v.nFaces == 0 || (s.faceFlux && v.faceFlux && g.areaVector && g.faceCentre && g.owner
                && g.neighbour && g.periodicPartner && g.boundaryKind));
        }
        __device__ Real distanceFor(int c, GasView v, GeometryView g) {
            return v.sst.wallDistance?v.sst.wallDistance[c]:g.wallDistance[c];
        }
        __device__ bool isWall(BoundaryKind k) {
            return k == BoundaryKind::NoSlip || k == BoundaryKind::Interface;
        }
        __device__ SstPrimitive boundarySst(int c, int f, GasView v, GeometryView g) {
            auto s = v.sst.primitive[c];
            auto kind = g.boundaryKind[f];
            if (isWall(kind)) {
                s = lowReSstWallTrace(s);
            } else if (kind == BoundaryKind::Inlet && v.sst.boundary) {
                s = v.sst.boundary[f];
            }
            return s;
        }
        __global__ void recoverSst(GasView v, GeometryView g, PhysicsConfig p) {
            int c = blockIdx.x*blockDim.x+threadIdx.x;
            if (c >= v.nCells || gasStopped(v.status))return;
            const Real mass = v.q[c].mass;
            const auto q = v.sst.q[c];
            const Real distance = distanceFor(c, v, g);
            if (!(mass>0) || !finite(q.rhoK) || !finite(q.rhoOmega) || q.rhoK<0 || q.rhoOmega <= 0
                || !(distance>0) || !finite(distance)) {
                gasDeviceError(v.status, ErrorCode::Inventory, c, q.rhoOmega, ErrorLocation::Cell);
                return;
            }
            SstPrimitive s;
            s.k = q.rhoK/mass;
            s.omega = q.rhoOmega/mass;
            if (s.k<p.sst.minimumK || s.omega<p.sst.minimumOmega) {
                gasDeviceError(v.status, ErrorCode::Inventory, c, s.omega, ErrorLocation::Cell);
                return;
            }
            v.sst.primitive[c] = s;
        }
        __global__ void sstGradients(GasView v, GeometryView g) {
            int c = blockIdx.x*blockDim.x+threadIdx.x;
            if (c >= v.nCells || gasStopped(v.status))return;
            Symmetric3 matrix;
            Vec3 bk{}, bo{};
            auto s = v.sst.primitive[c];
            for (int k = g.cellFaceOffsets[c]; k<g.cellFaceOffsets[c+1]; ++k) {
                int f = g.cellFaces[k];
                if (g.boundaryKind[f] == BoundaryKind::Empty)continue;
                int other = neighbourCell(c, f, g);
                Vec3 dx;
                SstPrimitive n;
                if (other >= 0) {
                    dx = neighbourDisplacement(c, f, other, g);
                    n = v.sst.primitive[other];
                } else {
                    dx = boundaryStencilDisplacement(g.boundaryKind[f], g.faceCentre[f]-g.cellCentre[c]);
                    n = boundarySst(c, f, v, g);
                    if (g.boundaryKind[f] != BoundaryKind::Inlet) {
                        n.k = 2*n.k-s.k;
                        n.omega = 2*n.omega-s.omega;
                    }
                }
                if (!finite(n.k) || !finite(n.omega)) {
                    gasDeviceError(v.status, ErrorCode::PropertyRange, f, n.omega, ErrorLocation::Face);
                    return;
                }
                Symmetric3 unused;
                addLeastSquares(matrix, bk, dx, n.k-s.k);
                addLeastSquares(unused, bo, dx, n.omega-s.omega);
            }
            v.sst.gradientK[c] = solveLeastSquares(matrix, bk);
            v.sst.gradientOmega[c] = solveLeastSquares(matrix, bo);
        }
        __global__ void algebraSst(GasView v, GeometryView g, PhysicsConfig p) {
            int c = blockIdx.x*blockDim.x+threadIdx.x;
            if (c >= v.nCells || gasStopped(v.status))return;
            if (!sstAlgebra(v.sst.q[c], v.primitive[c].rho, g.evaluationVolume[c], v.primitive[c],
                v.gradient[c], v.sst.gradientK[c], v.sst.gradientOmega[c], distanceFor(c, v, g), p,
                v.sst.primitive[c]))gasDeviceError(v.status, ErrorCode::PropertyRange, c, distanceFor(c, v, g),
                ErrorLocation::Cell);
        }
        __device__ SstPrimitive reconstructedSst(int c, int f, GasView v, GeometryView g, const PhysicsConfig& p) {
            auto s = v.sst.primitive[c];
            if (p.spatialOrder != 2)return s;
            Vec3 dx = g.faceCentre[f]-g.cellCentre[c];
            Real dk = dot(v.sst.gradientK[c], dx), dw = dot(v.sst.gradientOmega[c], dx), theta = 1;
            if (p.reconstruction == ReconstructionMode::LimitedLinear) {
                Real lk = s.k, hk = s.k, lw = s.omega, hw = s.omega;
                for (int i = g.cellFaceOffsets[c]; i<g.cellFaceOffsets[c+1]; ++i) {
                    int face = g.cellFaces[i];
                    if (g.boundaryKind[face] == BoundaryKind::Empty)continue;
                    int other = neighbourCell(c, face, g);
                    auto n = other >= 0?v.sst.primitive[other]:boundarySst(c, face, v, g);
                    lk = minValue(lk, n.k);
                    hk = maxValue(hk, n.k);
                    lw = minValue(lw, n.omega);
                    hw = maxValue(hw, n.omega);
                }
                theta = minValue(barthJespersen(s.k, lk, hk, dk), barthJespersen(s.omega, lw, hw, dw));
            }
            if (dk<0)theta = minValue(theta, s.k/(-dk));
            if (dw<0)theta = minValue(theta, .999999999999*s.omega/(-dw));
            s.k += theta*dk;
            s.omega += theta*dw;
            return s;
        }
        __global__ void fluxSst(GasView v, GeometryView g, PhysicsConfig p) {
            int f = blockIdx.x*blockDim.x+threadIdx.x;
            if (f >= v.nFaces || gasStopped(v.status))return;
            auto kind = g.boundaryKind[f];
            int l = g.owner[f], r = g.neighbour[f];
            if (kind == BoundaryKind::Periodic) {
                int pair = g.periodicPartner[f];
                if (f>pair)return;
                r = g.owner[pair];
            }
            if (kind == BoundaryKind::Empty || kind == BoundaryKind::Slip) {
                v.sst.faceFlux[f] = {};
                return;
            }
            auto left = reconstructedSst(l, f, v, g, p), right = left;
            Vec3 gradK = v.sst.gradientK[l], gradO = v.sst.gradientOmega[l], dx;
            Real rhoL = v.primitive[l].rho, rhoR = rhoL;
            if (r >= 0) {
                int rf = kind == BoundaryKind::Periodic?g.periodicPartner[f]:f;
                right = reconstructedSst(r, rf, v, g, p);
                dx = neighbourDisplacement(l, f, r, g);
                rhoR = v.primitive[r].rho;
                gradK = (gradK+v.sst.gradientK[r])*.5;
                gradO = (gradO+v.sst.gradientOmega[r])*.5;
                gradK = correctedGradient(gradK, dx, v.sst.primitive[r].k-v.sst.primitive[l].k);
                gradO = correctedGradient(gradO, dx, v.sst.primitive[r].omega-v.sst.primitive[l].omega);
            } else {
                if (kind == BoundaryKind::Inlet && !v.sst.boundary) {
                    gasDeviceError(v.status, ErrorCode::InvalidInput, f, 0, ErrorLocation::Face);
                    return;
                }
                right = boundarySst(l, f, v, g);
                dx = g.faceCentre[f]-g.cellCentre[l];
                gradK = correctedGradient(gradK, dx, right.k-v.sst.primitive[l].k);
                gradO = correctedGradient(gradO, dx, right.omega-v.sst.primitive[l].omega);
                if (kind == BoundaryKind::Outlet)gradK = gradO = {};
                if (isWall(kind)) {
                    gradO = lowReWallOmegaGradient(gradO, g.areaVector[f]);
                }
            }
            if (!finite(right.k) || !finite(right.omega) || right.k<0 || right.omega <= 0) {
                gasDeviceError(v.status, ErrorCode::PropertyRange, f, right.omega, ErrorLocation::Face);
                return;
            }
            auto flux = sstTransportFlux(v.faceFlux[f].mass, left, right, gradK, gradO, g.areaVector[f], rhoL,
                rhoR, p, isWall(kind));
            if (!finite(flux.rhoK) || !finite(flux.rhoOmega)) {
                gasDeviceError(v.status, ErrorCode::Nonfinite, f, flux.rhoOmega, ErrorLocation::Face);
                return;
            }
            v.sst.faceFlux[f] = flux;
            if (kind == BoundaryKind::Periodic)v.sst.faceFlux[g.periodicPartner[f]] = {-flux.rhoK, -flux.rhoOmega};
        }
        __global__ void rhsSst(GasView v, GeometryView g, PhysicsConfig p) {
            int c = blockIdx.x*blockDim.x+threadIdx.x;
            if (c >= v.nCells || gasStopped(v.status))return;
            SstQ rhs{};
            for (int k = g.cellFaceOffsets[c]; k<g.cellFaceOffsets[c+1]; ++k) {
                auto f = v.sst.faceFlux[g.cellFaces[k]];
                rhs.rhoK -= g.cellFaceSigns[k]*f.rhoK;
                rhs.rhoOmega -= g.cellFaceSigns[k]*f.rhoOmega;
            }
            auto source = sstSources(v.primitive[c], v.sst.primitive[c], v.gradient[c], v.sst.gradientK[c],
                v.sst.gradientOmega[c], g.evaluationVolume[c], p);
            rhs.rhoK += source.productionK-source.destructionK*v.sst.q[c].rhoK;
            rhs.rhoOmega += source.productionOmega-source.destructionOmega*v.sst.q[c].rhoOmega;
            v.sst.rhs[c] = rhs;
            v.sst.production[c] = source.productionK;
            v.sst.dissipation[c] = source.destructionK*v.sst.q[c].rhoK;
        }
        __global__ void updateSst(const SstQ* base, GasView v, GeometryView g, PhysicsConfig p, Real dt,
            DeviceStatus* status) {
            int c = blockIdx.x*blockDim.x+threadIdx.x;
            if (c >= v.nCells || gasStopped(status))return;
            // Mean primitive/gradients still refer to RHS evaluation geometry here.
            auto source = sstSources(v.primitive[c], v.sst.primitive[c], v.gradient[c], v.sst.gradientK[c],
                v.sst.gradientOmega[c], g.evaluationVolume[c], p);
            Real lossK = 0, lossO = 0;
            SstQ result;
            result.rhoK = exponentialDestruction(base[c].rhoK,
                v.sst.rhs[c].rhoK+source.destructionK*v.sst.q[c].rhoK, source.destructionK, dt, lossK);
            result.rhoOmega = exponentialDestruction(base[c].rhoOmega,
                v.sst.rhs[c].rhoOmega+source.destructionOmega*v.sst.q[c].rhoOmega, source.destructionOmega, dt,
                lossO);
            bool wallConstraintPending = false;
            for (int entry = g.cellFaceOffsets[c]; entry < g.cellFaceOffsets[c+1]; ++entry) {
                const int face = g.cellFaces[entry];
                wallConstraintPending = wallConstraintPending || (isWall(g.boundaryKind[face])
                    && g.neighbour[face] < 0 && g.owner[face] == c);
            }
            if (!sstStageAdmissible(result, v.q[c].mass, p, wallConstraintPending)
                || !finite(lossK) || lossK < 0) {
                gasDeviceError(status, ErrorCode::Inventory, c, result.rhoOmega, ErrorLocation::Cell);
                return;
            }
            v.sst.q[c] = result;
            // Preserve the actual provisional inventory. Endpoint projection
            // accounts for its full delta; a skipped projection must fail validation.
            v.sst.omegaConstraint[c] = wallConstraintPending ? undefinedValue() : 0;
            v.sst.production[c] = source.productionK;
            v.sst.dissipation[c] = lossK/dt;
        }
        __global__ void validateSstEndpoint(GasView view, PhysicsConfig physics, DeviceStatus* status) {
            const int cell = blockIdx.x*blockDim.x+threadIdx.x;
            if (cell >= view.nCells || gasStopped(status))return;
            if (!sstEndpointAdmissible(view.sst.q[cell], view.q[cell].mass, physics,
                view.sst.omegaConstraint[cell])) {
                gasDeviceError(status, ErrorCode::Inventory, cell, view.sst.q[cell].rhoOmega,
                    ErrorLocation::Cell);
            }
        }
        __global__ void constrainWallCells(GasView v, GeometryView endpoint, PhysicsConfig p,
            DeviceStatus* status) {
            const int cell = blockIdx.x*blockDim.x+threadIdx.x;
            if (cell >= v.nCells || gasStopped(status))return;
            Real targets = 0;
            int count = 0;
            for (int k = endpoint.cellFaceOffsets[cell]; k<endpoint.cellFaceOffsets[cell+1]; ++k) {
                const int face = endpoint.cellFaces[k];
                if (!isWall(endpoint.boundaryKind[face]))continue;
                if (endpoint.neighbour[face] >= 0 || endpoint.owner[face] != cell || !endpoint.boundaryPrimitive) {
                    gasDeviceError(status, ErrorCode::InvalidInput, face, 0, ErrorLocation::Face);
                    return;
                }
                Real target = 0;
                if (!wallOmegaTarget(endpoint.boundaryPrimitive[face], endpoint.wallDistance[cell], p, target)) {
                    gasDeviceError(status, ErrorCode::PropertyRange, face, endpoint.wallDistance[cell],
                        ErrorLocation::Face);
                    return;
                }
                targets += target;
                ++count;
            }
            SstQ constrained;
            Real delta = 0;
            if (!applyWallOmegaConstraint(v.q[cell], v.sst.q[cell], targets, count, p, constrained, delta)) {
                gasDeviceError(status, ErrorCode::Inventory, cell, targets, ErrorLocation::Cell);
                return;
            }
            v.sst.q[cell] = constrained;
            v.sst.omegaConstraint[cell] = delta;
        }
    }
    int launchSstPrepare(GasView v, GeometryView g, const PhysicsConfig& p, void* stream) {
        if (!p.enableSst)return 0;
        if (!validSst(v, g) || p.gasViscosity <= 0)return static_cast<int>(cudaErrorInvalidValue);
        if (!v.nCells)return 0;
        auto s = static_cast<cudaStream_t>(stream);
        recoverSst<<<(v.nCells+127)/128, 128, 0, s>>>(v, g, p);
        if (int e = launchError())return e;
        sstGradients<<<(v.nCells+127)/128, 128, 0, s>>>(v, g);
        if (int e = launchError())return e;
        algebraSst<<<(v.nCells+127)/128, 128, 0, s>>>(v, g, p);
        return launchError();
    }
    int launchSstRhs(GasView v, GeometryView g, const PhysicsConfig& p, double dt, void* stream) {
        if (!p.enableSst)return 0;
        if (!validSst(v, g) || !finite(dt) || dt <= 0)return static_cast<int>(cudaErrorInvalidValue);
        if (!v.nCells)return 0;
        auto s = static_cast<cudaStream_t>(stream);
        if (v.nFaces) {
            fluxSst<<<(v.nFaces+127)/128, 128, 0, s>>>(v, g, p);
            if (int e = launchError())return e;
        }
        rhsSst<<<(v.nCells+127)/128, 128, 0, s>>>(v, g, p);
        return launchError();
    }
    int launchSstUpdate(const SstQ* base, GasView v, GeometryView g, const PhysicsConfig& p, double dt,
        DeviceStatus* status, void* stream) {
        if (!p.enableSst)return 0;
        if (!validSst(v, g) || !base || !status || !finite(dt)
            || dt <= 0)return static_cast<int>(cudaErrorInvalidValue);
        if (!v.nCells)return 0;
        updateSst<<<(v.nCells+127)/128, 128, 0, static_cast<cudaStream_t>(stream)>>>(base, v, g, p, dt, status);
        return launchError();
    }
    int launchSstValidate(GasView view, const PhysicsConfig& physics, DeviceStatus* status, void* stream) {
        if (!physics.enableSst)return 0;
        if (view.nCells < 0 || view.sst.nCells != view.nCells || !status
            || (view.nCells && (!view.q || !view.sst.q || !view.sst.omegaConstraint))) {
            return static_cast<int>(cudaErrorInvalidValue);
        }
        if (!view.nCells)return 0;
        validateSstEndpoint<<<(view.nCells+127)/128, 128, 0, static_cast<cudaStream_t>(stream)>>>(
            view, physics, status);
        return launchError();
    }
    int launchSstConstrainWalls(GasView v, GeometryView endpoint, const PhysicsConfig& p, DeviceStatus* status,
        void* stream) {
        if (!p.enableSst)return 0;
        if (v.nCells != endpoint.nCells || v.nFaces != endpoint.nFaces || v.sst.nCells != v.nCells || !status
            || v.nCells<0)return static_cast<int>(cudaErrorInvalidValue);
        if (!v.nCells)return 0;
        if (!v.q || !v.sst.q || !v.sst.omegaConstraint || !endpoint.wallDistance || !endpoint.cellFaceOffsets
            || !endpoint.cellFaces || !endpoint.boundaryKind || !endpoint.owner
            || !endpoint.neighbour)return static_cast<int>(cudaErrorInvalidValue);
        constrainWallCells<<<(v.nCells+127)/128, 128, 0, static_cast<cudaStream_t>(stream)>>>(v, endpoint, p,
            status);
        return launchError();
    }
}
// namespace chmt
