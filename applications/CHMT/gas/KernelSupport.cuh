#ifndef CHMT_GAS_KERNELSUPPORT_CUH
#define CHMT_GAS_KERNELSUPPORT_CUH
#include "gpu/DeviceViews.H"
#include <cuda_runtime.h>
namespace chmt {
    __device__ inline void gasDeviceError(DeviceStatus* s, ErrorCode code, int index, Real value,
        ErrorLocation location) {
        if (s && atomicCAS(&s->code, 0, static_cast<int>(code)) == 0) {
            s->index = index;
            s->value = value;
            s->location = location;
        }
    }
    __device__ inline bool gasStopped(const DeviceStatus* s) {
        return s && s->code != 0;
    }
    __device__ inline int neighbourCell(int cell, int face, GeometryView g) {
        if (g.neighbour[face] >= 0)return cell == g.owner[face]?g.neighbour[face]:g.owner[face];
        if (g.boundaryKind[face] == BoundaryKind::Periodic)return g.owner[g.periodicPartner[face]];
        return -1;
    }
    __device__ inline Vec3 neighbourDisplacement(int cell, int face, int other, GeometryView g) {
        Vec3 dx = g.cellCentre[other]-g.cellCentre[cell];
        if (g.boundaryKind[face] == BoundaryKind::Periodic) {
            dx += g.faceCentre[face]-g.faceCentre[g.periodicPartner[face]];
        }
        return dx;
    }
    inline int launchError() {
        return static_cast<int>(cudaPeekAtLastError());
    }
}
#endif
