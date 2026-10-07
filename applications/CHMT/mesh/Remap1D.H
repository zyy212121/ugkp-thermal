#ifndef CHMT_MESH_REMAP1D_H
#define CHMT_MESH_REMAP1D_H
#include "core/Types.H"
namespace chmt {
    // Piecewise-constant overlap of integral inventories; not a 3-D remesher.
    CHMT_HD inline bool remapCell1D(const Real* oldEdges, int nOld, const GasQ* oldQ, const Real* newEdges,
        int nNew, int cell, GasQ& out) {
        if (!oldEdges || !newEdges || !oldQ || nOld<1 || nNew<1 || cell<0 || cell >= nNew)return false;
        for (int i = 0; i <= nOld; ++i)if (!finite(oldEdges[i]) || (i
            && oldEdges[i] <= oldEdges[i-1]))return false;
        for (int i = 0; i <= nNew; ++i)if (!finite(newEdges[i]) || (i
            && newEdges[i] <= newEdges[i-1]))return false;
        const Real length = oldEdges[nOld]-oldEdges[0], tol = 64*std::numeric_limits<Real>::epsilon()*length;
        if (absValue(oldEdges[0]-newEdges[0])>tol || absValue(oldEdges[nOld]-newEdges[nNew])>tol)return false;
        GasQ result{};
        Real covered = 0;
        for (int i = 0; i<nOld; ++i) {
            const auto& q = oldQ[i];
            if (!finite(q.mass) || q.mass <= 0 || !finite(q.energy) || !finite(q.momentum))return false;
            Real sum = 0;
            for (int s = 0; s<Ns; ++s) {
                if (!finite(q.species[s]) || q.species[s]<0)return false;
                sum += q.species[s];
            }
            if (absValue(sum-q.mass)>128*std::numeric_limits<Real>::epsilon()*q.mass)return false;
            const Real overlap = maxValue(0, minValue(oldEdges[i+1], newEdges[cell+1])-maxValue(oldEdges[i],
                newEdges[cell]));
            result += q*(overlap/(oldEdges[i+1]-oldEdges[i]));
            covered += overlap;
        }
        if (absValue(covered-(newEdges[cell+1]-newEdges[cell]))>tol)return false;
        out = result;
        return true;
    }
    int launchConservativeRemap1D(const double*, int, const GasQ*, const double*, int, GasQ*, DeviceStatus*,
        void*);
}
#endif
