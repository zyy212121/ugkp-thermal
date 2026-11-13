#pragma once
#include "FoamSolverScheme.H"
namespace ugkp { namespace solverControls {
template<class Config> inline void readWallLayerSolverControls(Config& c,
    const Foam::Time& time,const Foam::word& region=Foam::word::null) {
    const auto& scheme=startupScheme(time);
    Foam::label nodes=c.nodes;
    scheme.read("wallLayer","nodes",nodes,"effectiveDefault",region);c.nodes=nodes;
    scheme.read("wallLayer","stretch",c.stretch,"effectiveDefault",region);
    Foam::label iterations=c.maxIterations;
    scheme.read("wallLayer","maxIterations",iterations,"precisionDefault",region);c.maxIterations=iterations;
#define UGKP_WALL_OUTER(Key) scheme.read("wallLayer",#Key,c.Key,"precisionDefault",region)
    UGKP_WALL_OUTER(relativeTolerance);UGKP_WALL_OUTER(absoluteTolerance);
#undef UGKP_WALL_OUTER
#define UGKP_WALL_INNER(Key) scheme.read("wallLayer",#Key,c.numerics.Key,"effectiveDefault",region)
    UGKP_WALL_INNER(maximumLineSearchIterations);UGKP_WALL_INNER(lineSearchReduction);
    UGKP_WALL_INNER(seedMaximumIterations);UGKP_WALL_INNER(seedRelativeTolerance);UGKP_WALL_INNER(seedRelaxation);
    UGKP_WALL_INNER(profileBracketIterations);UGKP_WALL_INNER(profileBracketGrowth);
    UGKP_WALL_INNER(profileBisectionIterations);UGKP_WALL_INNER(jacobianRefreshReduction);
#undef UGKP_WALL_INNER
}
}}
