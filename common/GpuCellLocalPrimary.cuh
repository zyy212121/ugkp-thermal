#pragma once
// Shared primary fields with optional compile-time thermal extension.
__device__ __forceinline__ void copyCellLocalParticle
(DeviceState& s, const int i, const int c, const int dst)
{
            s.compactPx[dst] = s.px[i];
            s.compactPy[dst] = s.py[i];
            s.compactPz[dst] = s.pz[i];
            s.compactPux[dst] = s.pux[i];
            s.compactPuy[dst] = s.puy[i];
            s.compactPuz[dst] = s.puz[i];
            s.compactPT[dst] = s.pT[i];
            s.compactPTheta[dst] = s.pTheta[i];
#ifdef GPU_PARTICLE_EXTRA_FIELDS
    GPU_PARTICLE_EXTRA_FIELDS::copyContactAge(s, i, dst);
#endif
            s.compactPd[dst] = s.pd[i];
            s.compactPm[dst] = s.pm[i];
            s.compactPCellId[dst] = c;
            s.compactPStatus[dst] = 1;
        #ifdef GPU_PARTICLE_EXTRA_FIELDS
    GPU_PARTICLE_EXTRA_FIELDS::copy(s, i, dst);
#endif
    s.compactPRng[dst] = s.pRng[i];
            s.compactPOrigId[dst] = s.pOrigId[i];
#ifdef GPU_PARTICLE_EXTRA_FIELDS
    GPU_PARTICLE_EXTRA_FIELDS::publish(s, dst);
#endif
}

#ifdef GPU_PARTICLE_EXTRA_FIELDS
#undef GPU_PARTICLE_EXTRA_FIELDS
#endif
