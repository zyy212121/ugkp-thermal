#pragma once
#if defined(__CUDACC__)
#define UGKP_WALL_CONTROL_HD __host__ __device__
#else
#define UGKP_WALL_CONTROL_HD
#endif
namespace ugkwp { namespace gaswall {
// Mathematical iteration controls only. Defaults preserve the existing solver.
template<class R> struct WallIterationControls {
    int maximumLineSearchIterations=40;
    R lineSearchReduction=R(.5);
    int seedMaximumIterations=200;
    R seedRelativeTolerance=R(1e-5),seedRelaxation=R(.5);
    int profileBracketIterations=16;
    R profileBracketGrowth=R(2);
    int profileBisectionIterations=70;
    R jacobianRefreshReduction=R(.5);
};
template<class D,class S> UGKP_WALL_CONTROL_HD inline WallIterationControls<D>
castWallIterationControls(const WallIterationControls<S>& s) {
    WallIterationControls<D> d;
    d.maximumLineSearchIterations=s.maximumLineSearchIterations;
    d.lineSearchReduction=D(s.lineSearchReduction);
    d.seedMaximumIterations=s.seedMaximumIterations;
    d.seedRelativeTolerance=D(s.seedRelativeTolerance);d.seedRelaxation=D(s.seedRelaxation);
    d.profileBracketIterations=s.profileBracketIterations;
    d.profileBracketGrowth=D(s.profileBracketGrowth);
    d.profileBisectionIterations=s.profileBisectionIterations;
    d.jacobianRefreshReduction=D(s.jacobianRefreshReduction);
    return d;
}
template<class R> UGKP_WALL_CONTROL_HD inline R relaxWallSeed(R oldValue,R candidate,R relaxation) {
    // Keep the previous arithmetic exactly for the default half relaxation.
    return relaxation==R(.5)?(oldValue+candidate)/R(2)
        :(R(1)-relaxation)*oldValue+relaxation*candidate;
}
}}
#undef UGKP_WALL_CONTROL_HD
