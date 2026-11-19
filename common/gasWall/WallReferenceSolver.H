#pragma once
#include "ReactingWallLayer.H"

namespace ugkwp { namespace gaswall {


template<int N,int M> struct WallReferenceScratch {
    WallReal best[M][N+4]{};
};
struct WallReferenceStats {
    WallReal residual=std::numeric_limits<WallReal>::infinity();
    int iterations=0,residualEvaluations=0,jacobianEvaluations=0,centralDirections=0;
    bool usedHistory=false,converged=false;
    bool usedNumericalState=false;
};
struct WallReferenceCounters {
 unsigned long long solves=0,failures=0,iterations=0,residuals=0,jacobians=0,central=0,history=0;
 unsigned long long numericalHistory=0;
 WallReal maxResidual=0;
};
namespace detail {
template<class R,int N,int M,class Team>
UGKWP_GAS_HD inline R referenceMerit(const WallModelConfig<R>& c,
 const WallWorkspace<R,N,M>& w,const Team& team)
{

    R value=0;if(team.leader())for(int i=0;i<c.nodes;++i)for(int j=0;j<N+4;++j)
        value+=w.residual[i][j]*w.residual[i][j];
    return team.broadcast(value);
}
}






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
        const R f=std::pow(R(i)/R(c.nodes-1),c.stretch);w.y[i]=in.matchingDistance*f;
        if(accepted){for(int j=0;j<B;++j)w.state[i][j]=accepted[i*B+j];}
        else{
            w.state[i][0]=f*ctx.edgeU[0];w.state[i][1]=f*ctx.edgeU[1];
            w.state[i][2]=f*(in.matching.temperature-in.temperature);
            for(int j=0;j<N-1;++j)w.state[i][3+j]=in.matching.massFraction[ctx.independent[j]];
            w.state[i][N+2]=c.enableSst?std::sqrt(in.matching.k)*f:R(0);
            w.state[i][N+3]=c.enableSst?f/std::sqrt(in.matching.omega):R(1);
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
    int iteration=0,searchesRemaining=c.maxIterations>0?c.maxIterations:1;const R tolerance=c.absoluteTolerance+c.relativeTolerance;
    for(;norm>tolerance&&iteration<c.maxIterations&&searchesRemaining>0;++iteration){
        if(team.leader())++stats.jacobianEvaluations;
        if(!newtonStep(in,c,ctx,w,team)){if(team.leader())wallFatal(in,ugkp::diagnostics::Code::ZeroDenominator,2421);return false;}
        const R merit=referenceMerit(c,w,team);R bestMerit=0,alpha=1;
        bool available=false,improved=false;


        for(int line=0;line<c.numerics.maximumLineSearchIterations&&searchesRemaining>0;++line){
            --searchesRemaining;
            const bool feasible=trialResidual(alpha);
            if(!storageAvailable)return false;
            if(feasible){
                const R nextMerit=referenceMerit(c,w,team);
                if(gasFinite(nextMerit) && nextMerit<merit){
                    for(int i=team.rank();i<c.nodes;i+=team.size())
                        for(int j=0;j<B;++j)w.state[i][j]=w.trial[i][j];
                    team.sync();improved=true;break;
                }
                if(gasFinite(nextMerit) && (!available || nextMerit<bestMerit)){
                    bestMerit=nextMerit;available=true;
                    for(int i=team.rank();i<c.nodes;i+=team.size())
                        for(int j=0;j<B;++j)scratch.best[i][j]=w.trial[i][j];
                    team.sync();
                }
            }
            alpha*=c.numerics.lineSearchReduction;
        }
        if(!improved && available){
            for(int i=team.rank();i<c.nodes;i+=team.size())
                for(int j=0;j<B;++j)w.state[i][j]=scratch.best[i][j];
            team.sync();
        }
        if(!improved || in.diagnostics){
            if(!measure(in,w.state)){if(team.leader())fail(status,WallCode::InadmissibleState);return false;}
        }
        if(!improved){++iteration;break;}
    }
    bool published=false;
    if(team.leader()){
        stats.residual=norm;stats.iterations=iteration;stats.converged=norm<=tolerance;
        status.iteration=iteration;status.residual=norm;
        if(!stats.converged){
            status.code=WallCode::NonConvergence;
            wallWarning(in,ugkp::diagnostics::Code::AccuracyResidual,2422,norm,tolerance);
        }
        published=publishLayer(in,c,ctx,w,iteration,norm,output,false);
        if(!published){fail(status,WallCode::InadmissibleState);wallFatal(in,ugkp::diagnostics::Code::NonFinite,10022);}
    }
    return team.broadcast(published);
}
}}
