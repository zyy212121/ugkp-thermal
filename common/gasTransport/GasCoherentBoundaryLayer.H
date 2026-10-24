#pragma once
#include "GasBoundaryLayerWorkspace.H"
#include "../gasWall/WallAcceptedHistory.H"
#include "../gasWall/WallCoherentSolver.H"
namespace ugkwp { namespace gaswall { namespace coherent {
// Numerical continuation belongs to a physical wall, unlike pooled evaluation
// scratch. This is not the accepted macro history: a last numerical iterate may
// remain useful even when its publication carries an advisory residual.
template<int M> struct PersistentWallState {
    Cache<M> cache;
    WallHistoryFrame<2> frame;
    double state[M][6]{};
};
struct SolveStats {
    int iterations=0,residualEvaluations=0,factorizations=0;
    double physicalResidual=std::numeric_limits<double>::infinity(),seconds=0;
};
}} // gaswall::coherent
inline std::size_t gasCoherentWallScratchBytes(int nodes) {
    switch(gasBoundaryLayerWorkspaceCapacity(nodes)) {
    case 24:return sizeof(gaswall::coherent::Scratch<24>);
    case 48:return sizeof(gaswall::coherent::Scratch<48>);
    case 128:return sizeof(gaswall::coherent::Scratch<128>);
    default:return 0;
    }
}
inline std::size_t gasCoherentWallStateBytes(int nodes) {
    switch(gasBoundaryLayerWorkspaceCapacity(nodes)) {
    case 24:return sizeof(gaswall::coherent::PersistentWallState<24>);
    case 48:return sizeof(gaswall::coherent::PersistentWallState<48>);
    case 128:return sizeof(gaswall::coherent::PersistentWallState<128>);
    default:return 0;
    }
}
UGKWP_GAS_HD inline bool sameCoherentCoordinates(
    const gaswall::WallHistoryFrame<2>& a,const gaswall::WallHistoryFrame<2>& b) {
    if(a.temperature!=b.temperature||a.dependent!=b.dependent||a.independent[0]!=b.independent[0])return false;
    for(int j=0;j<2;++j){if(a.wallU[j]!=b.wallU[j])return false;for(int d=0;d<3;++d)if(a.tangent[j][d]!=b.tangent[j][d])return false;}
    return true;
}
// accepted, when supplied, is already in the current frame. Geometry changes
// discard only the numerical initial guess/Jacobian, never the physical step.
template<int M,class Team>
UGKWP_GAS_HD inline bool loadCoherentWallState(
    gaswall::coherent::PersistentWallState<M>& wall,
    const gaswall::WallHistoryFrame<2>& frame,
    gaswall::WallWorkspace<double,2,M>& work,const double* accepted,const Team& team,
    bool* usedNumerical=nullptr) {
    const bool warm=team.broadcast(team.leader()&&wall.cache.haveState
        &&gaswall::compatibleHistory(wall.frame,frame));
    if(usedNumerical)*usedNumerical=warm;
    if(warm) {
        for(int i=team.rank();i<frame.nodes;i+=team.size())
            gaswall::transformHistoryRow(wall.state[i],work.state[i],wall.frame,frame);
        if(team.leader()&&!sameCoherentCoordinates(wall.frame,frame))wall.cache.factors.valid=false;
    } else {
        if(accepted)for(int i=team.rank();i<frame.nodes;i+=team.size())for(int j=0;j<6;++j)work.state[i][j]=accepted[i*6+j];
        if(team.leader()){wall.cache.haveState=accepted!=nullptr;wall.cache.factors.valid=false;}
    }
    team.sync();return warm||accepted!=nullptr;
}
template<int M,class Team>
UGKWP_GAS_HD inline void saveCoherentWallState(
    gaswall::coherent::PersistentWallState<M>& wall,
    const gaswall::WallHistoryFrame<2>& frame,
    const gaswall::WallWorkspace<double,2,M>& work,const Team& team) {
    for(int i=team.rank();i<frame.nodes;i+=team.size())for(int j=0;j<6;++j)wall.state[i][j]=work.state[i][j];
    team.sync();if(team.leader())wall.frame=frame;team.sync();
}

// Shared callable adapter. Preparation and advisory publication semantics are
// the same as the existing reference route; only the numerical operator/cache
// changes. accepted is immutable and has already been transformed by its owner.
template<int M,class Team=gaswall::detail::SerialWallTeam>
UGKWP_GAS_HD inline bool evaluateGasCoherentWallWorkspace(
    const gaswall::WallInput<double,2>& original,
    const gaswall::WallModelConfig<double>& originalConfig,
    gaswall::WallWorkspace<double,2,M>& work,
    gaswall::coherent::Scratch<M>& scratch,
    gaswall::coherent::PersistentWallState<M>& wall,const double* accepted,
    gaswall::WallOutput<double,2>& output,gaswall::WallStatus& status,
    gaswall::WallReferenceStats& stats,const Team& team=Team{}) {
    using namespace gaswall;
    auto in=original;in.model.thermo.diagnostics=in.diagnostics;
    auto config=originalConfig;config.coherentDiscretization=true;
    if(team.leader()){stats={};status={};output.residual=std::numeric_limits<double>::infinity();}
    team.sync();
    if(!detail::prepareReactingWallInput<M>(in,config,status,team))return false;
    detail::LayerContext<double,2> context;
    if(!team.all(detail::makeContext(in,config,context))){if(team.leader())detail::fail(status,WallCode::InvalidThermodynamics);return false;}
    const auto frame=historyFrame(in,config,context);
    bool numerical=false;
    const bool warm=loadCoherentWallState(wall,frame,work,accepted,team,&numerical);
    const auto centralBefore=scratch.centralDirections;
    team.sync();coherent::SolveStats measured;
    coherent::Engine<M,Team> engine(scratch,team);
    const bool published=engine.solve(in,config,work,wall.cache,output,measured);
    saveCoherentWallState(wall,frame,work,team);
    if(team.leader()){
        stats.residual=measured.physicalResidual;stats.iterations=measured.iterations;
        stats.residualEvaluations=measured.residualEvaluations;stats.jacobianEvaluations=measured.factorizations;
        stats.centralDirections=int(scratch.centralDirections-centralBefore);
        stats.usedHistory=warm&&!numerical;stats.usedNumericalState=numerical;stats.converged=stats.residual<=config.absoluteTolerance+config.relativeTolerance;
        status.iteration=stats.iterations;status.residual=stats.residual;
        if(!published)detail::fail(status,WallCode::InadmissibleState);
        if(published&&!stats.converged)detail::wallWarning(in,ugkp::diagnostics::Code::AccuracyResidual,2422,stats.residual,config.absoluteTolerance+config.relativeTolerance);
    }
    team.sync();return published;
}
} // ugkwp
