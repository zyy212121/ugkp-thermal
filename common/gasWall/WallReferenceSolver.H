#pragma once
#include "ReactingWallLayer.H"

namespace ugkwp { namespace gaswall {
// Reference solves borrow accepted history; only the macro-step commit may
// publish a new history. Scratch is worker-owned, history is wall-owned.
template<int N,int M> struct WallReferenceScratch {
    WallReal best[M][N+4]{};
};
struct WallReferenceStats {
    double residual=std::numeric_limits<double>::infinity();
    int iterations=0,residualEvaluations=0,jacobianEvaluations=0,centralDirections=0;
    bool usedHistory=false,converged=false;
    bool usedNumericalState=false; // separate from accepted macro history
};
struct WallReferenceCounters {
 unsigned long long solves=0,failures=0,iterations=0,residuals=0,jacobians=0,central=0,history=0;
 unsigned long long numericalHistory=0;
 double maxResidual=0;
};
namespace detail {
template<class R,int N,int M,class TableReal>
UGKWP_GAS_HD inline bool assembleReferenceCentralNode(
 const WallInput<R,N,TableReal>& in,const WallModelConfig<R>& c,
 const LayerContext<R,N>& ctx,WallWorkspace<R,N,M>& w,int node,bool& storageAvailable)
{
    constexpr int B=N+4;
    // Keep the simplex-tangent species columns, including absent species.
    if(!assembleJacobianNode(in,c,ctx,w,node))return false;
    auto silent=in;silent.diagnostics=nullptr;silent.model.thermo.diagnostics=nullptr;
    PerturbedLayerState<R,N> state(w.state,node);
    CachedLayerRows<R,N,decltype(state)> cache(state,node);
    const int columns[5]={0,1,2,N+2,N+3};
    for(int q=0;q<5;++q){
        const int col=columns[q];const R old=state.row[col];
        const R scale=col==2?R(100):col==N+2?::sqrt(max(R(1),in.matching.k)):
            col==N+3?::sqrt(std::numeric_limits<R>::epsilon())/::sqrt(in.matching.omega):R(1);
        const R h=::cbrt(std::numeric_limits<R>::epsilon())*max(abs(old),scale);
        if(col==N+3&&old<=h)continue;
        R plus[3][B]{},minus[3][B]{};const int lo=node>0?node-1:0,hi=node+1<c.nodes?node+1:c.nodes-1;
        state.row[col]=old+h;if(!ctx.trialStorageComplete&&!layerTrialStorageAvailable(in,in.temperature+state.row[2],node)){storageAvailable=false;return false;}cache.invalidate(node);bool valid=true;
        for(int i=lo;i<=hi;++i)valid=layerRow(silent,c,ctx,cache,w.y,i,plus[i-lo])&&valid;
        state.row[col]=old-h;if(!ctx.trialStorageComplete&&!layerTrialStorageAvailable(in,in.temperature+state.row[2],node)){storageAvailable=false;return false;}cache.invalidate(node);
        for(int i=lo;i<=hi;++i)valid=layerRow(silent,c,ctx,cache,w.y,i,minus[i-lo])&&valid;
        state.row[col]=old;
        if(!valid)continue; // Preserve the feasible forward column at a boundary.
        for(int i=lo;i<=hi;++i)for(int row=0;row<B;++row){
            R* block=node<i?w.lower[i][row]:node==i?w.diagonal[i][row]:w.upper[i][row];
            block[col]=(plus[i-lo][row]-minus[i-lo][row])/(R(2)*h);
        }
    }
    return true;
}
template<class R,int N,int M,class TableReal,class Team>
UGKWP_GAS_HD inline bool referenceCentralStep(
 const WallInput<R,N,TableReal>& in,const WallModelConfig<R>& c,
 const LayerContext<R,N>& ctx,WallWorkspace<R,N,M>& w,const Team& team,bool& storageAvailable)
{
    constexpr int B=N+4;
    for(int i=team.rank();i<c.nodes;i+=team.size())for(int row=0;row<B;++row){
        w.delta[i][row]=-w.residual[i][row];
        for(int col=0;col<B;++col)w.lower[i][row][col]=w.diagonal[i][row][col]=w.upper[i][row][col]=0;
    }
    team.sync();bool valid=true;
    for(int i=team.rank();i<c.nodes;i+=team.size())
        if(!assembleReferenceCentralNode(in,c,ctx,w,i,storageAvailable)){valid=false;break;}
    storageAvailable=team.all(storageAvailable);
    if(!team.all(valid))return false;
    bool solved=false;if(team.leader())solved=solveLayerNewtonSystem(c,w);
    return team.broadcast(solved);
}
template<class R,int N,int M,class Team>
UGKWP_GAS_HD inline R referenceMerit(const WallModelConfig<R>& c,
 const WallWorkspace<R,N,M>& w,const Team& team)
{
    // Deterministic order matches the CPU reference, independent of warp width.
    R value=0;if(team.leader())for(int i=0;i<c.nodes;++i)for(int j=0;j<N+4;++j)
        value+=w.residual[i][j]*w.residual[i][j];
    return team.broadcast(value);
}
}

// accepted is an optional, immutable transformed profile in the CURRENT wall
// frame/species coordinates and on the same node geometry. The adapter owns
// coordinate conversion and geometry invalidation. No endpoint lift or warm
// Picard pass is applied here. A nonconverged publication remains advisory, as
// in the existing BVP API; stats distinguish it from a converged equation solve.
template<class R,int N,int M,class TableReal,class Team=detail::SerialWallTeam>
UGKWP_GAS_HD inline bool evaluateWallReference(
 const WallInput<R,N,TableReal>& original,const WallModelConfig<R>& c,
 WallWorkspace<R,N,M>& w,WallReferenceScratch<N,M>& scratch,const R* accepted,
 WallOutput<R,N>& output,WallStatus& status,WallReferenceStats& stats,const Team& team=Team{})
{
    using namespace detail;constexpr int B=N+4;
    if(team.leader()){stats={};status={};output.residual=std::numeric_limits<R>::infinity();}
    team.sync();auto in=original;in.model.thermo.diagnostics=in.diagnostics;
    if(!prepareReactingWallInput<M>(in,c,status,team))return false;
    LayerContext<R,N> ctx;
    if(!team.all(makeContext(in,c,ctx))){if(team.leader())fail(status,WallCode::InvalidThermodynamics);return false;}
    for(int i=team.rank();i<c.nodes;i+=team.size()){
        const R f=::pow(R(i)/R(c.nodes-1),c.stretch);w.y[i]=in.matchingDistance*f;
        if(accepted){for(int j=0;j<B;++j)w.state[i][j]=accepted[i*B+j];}
        else{
            w.state[i][0]=f*ctx.edgeU[0];w.state[i][1]=f*ctx.edgeU[1];
            w.state[i][2]=f*(in.matching.temperature-in.temperature);
            for(int j=0;j<N-1;++j)w.state[i][3+j]=in.matching.massFraction[ctx.independent[j]];
            w.state[i][N+2]=c.enableSst?::sqrt(in.matching.k)*f:R(0);
            w.state[i][N+3]=c.enableSst?f/::sqrt(in.matching.omega):R(1);
        }
    }
    team.sync();if(!accepted&&!seedSstLayer(in,c,ctx,w,team)){if(team.leader()){fail(status,WallCode::InadmissibleState);wallFatal(in,ugkp::diagnostics::Code::NonFinite,2420);}return false;}
    if(team.leader())stats.usedHistory=accepted!=nullptr;
    R norm=0;
    auto measure=[&](const auto& input,const auto& state){
        if(team.leader())++stats.residualEvaluations;
        return allResidual(input,c,ctx,state,w.y,w.residual,norm,team);
    };
    if(!measure(in,w.state)){if(team.leader())fail(status,WallCode::InadmissibleState);return false;}
    auto silent=in;silent.diagnostics=nullptr;silent.model.thermo.diagnostics=nullptr;
    bool storageAvailable=true;
    auto trialResidual=[&](R step){
        bool feasible=true;
        for(int i=team.rank();i<c.nodes;i+=team.size()){
            for(int j=0;j<B;++j)w.trial[i][j]=w.state[i][j]+step*w.delta[i][j];
            feasible=feasibleLayerComposition<R,N>(w.trial[i])&&feasible;
        }
        team.sync();if(team.leader())++stats.residualEvaluations;
        return allResidual(silent,c,ctx,w.trial,w.y,w.residual,norm,team,
            ctx.trialStorageComplete?nullptr:&in,&storageAvailable,feasible,w.upper);
    };
    int iteration=0;const R tolerance=c.absoluteTolerance+c.relativeTolerance;
    for(;norm>tolerance&&iteration<c.maxIterations;++iteration){
        if(team.leader())++stats.jacobianEvaluations;
        if(!newtonStep(in,c,ctx,w,team)){if(team.leader())wallFatal(in,ugkp::diagnostics::Code::ZeroDenominator,2421);return false;}
        const R merit=referenceMerit(c,w,team);R bestMerit=merit,alpha=1;
        bool improved=false;
        for(int line=0;line<40;++line){
            const bool feasible=trialResidual(alpha);
            if(!storageAvailable)return false;
            const R nextMerit=referenceMerit(c,w,team);
            if(feasible&&gasFinite(nextMerit)&&nextMerit<merit){
                improved=true;bestMerit=nextMerit;break;
            }
            alpha/=R(2);
        }
        if(improved&&alpha>::sqrt(std::numeric_limits<R>::epsilon())){
            for(int i=team.rank();i<c.nodes;i+=team.size())for(int j=0;j<B;++j)w.state[i][j]=w.trial[i][j];
            team.sync();continue;
        }
        // Preserve the best forward trial while re-evaluating the original
        // residual. This replaces CPU old-state/old-residual stack copies.
        for(int i=team.rank();i<c.nodes;i+=team.size())for(int j=0;j<B;++j)
            scratch.best[i][j]=improved?w.trial[i][j]:w.state[i][j];
        team.sync();if(!measure(in,w.state)){if(team.leader())fail(status,WallCode::InadmissibleState);return false;}
        if(team.leader()){++stats.jacobianEvaluations;++stats.centralDirections;}
        if(referenceCentralStep(in,c,ctx,w,team,storageAvailable)){
            R a=1;
            for(int line=0;line<40;++line){
                const bool feasible=trialResidual(a);
                if(!storageAvailable)return false;
                const R candidate=referenceMerit(c,w,team);
                if(feasible&&gasFinite(candidate)&&candidate<bestMerit){
                    for(int i=team.rank();i<c.nodes;i+=team.size())for(int j=0;j<B;++j)scratch.best[i][j]=w.trial[i][j];
                    improved=true;break;
                }
                a/=R(2);
            }
        }
        if(!storageAvailable)return false;
        team.sync();for(int i=team.rank();i<c.nodes;i+=team.size())for(int j=0;j<B;++j)w.state[i][j]=scratch.best[i][j];
        team.sync();if(!measure(in,w.state)){if(team.leader())fail(status,WallCode::InadmissibleState);return false;}
        if(!improved)break;
    }
    if(!measure(in,w.state)){if(team.leader())fail(status,WallCode::InadmissibleState);return false;}
    bool published=false;
    if(team.leader()){
        stats.residual=norm;stats.iterations=iteration;stats.converged=norm<=tolerance;
        status.iteration=iteration;status.residual=norm;
        published=publishLayer(in,c,ctx,w,iteration,norm,output,false);
        if(!published){fail(status,WallCode::InadmissibleState);wallFatal(in,ugkp::diagnostics::Code::NonFinite,10022);}
        if(published&&!stats.converged)wallWarning(in,ugkp::diagnostics::Code::AccuracyResidual,2422,norm,tolerance);
    }
    return team.broadcast(published);
}
}}
