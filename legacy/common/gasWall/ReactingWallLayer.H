#pragma once
#include <cstdio>
#include "../../../common/gasWall/WallLayerTransport.H"
namespace ugkwp { namespace gaswall {
template<class R,int N,int M,class TableReal,class Team=detail::SerialWallTeam> UGKWP_GAS_HD inline bool evaluateReactingWallLayerImpl(
 const WallInput<R,N,TableReal>& originalInput,const WallModelConfig<R>& c,WallWorkspace<R,N,M>& w,WallOutput<R,N>& output,WallStatus& status,const Team& team=Team{},const R* accepted=nullptr){
    using namespace detail;
    auto in=originalInput;in.model.thermo.diagnostics=in.diagnostics;
    if(!prepareReactingWallInput<M>(in,c,status,team))return false;
    LayerContext<R,N> ctx;if(!team.all(makeContext(in,c,ctx))){if(team.leader())fail(status,WallCode::InvalidThermodynamics);return false;}
    for(int i=team.rank();i<c.nodes;i+=team.size()){const R eta=R(i)/R(c.nodes-1),f=::pow(eta,c.stretch);w.y[i]=in.matchingDistance*f;
        if(accepted){for(int j=0;j<N+4;++j)w.state[i][j]=accepted[i*(N+4)+j];continue;}
        w.state[i][0]=f*ctx.edgeU[0];w.state[i][1]=f*ctx.edgeU[1];
        w.state[i][2]=f*(in.matching.temperature-in.temperature);
        for(int j=0;j<N-1;++j)w.state[i][3+j]=in.matching.massFraction[ctx.independent[j]];
        w.state[i][N+2]=c.enableSst?::sqrt(in.matching.k)*f:R(0);w.state[i][N+3]=c.enableSst?f/::sqrt(in.matching.omega):R(1);


}


    team.sync();
    // Seed composition is constant until Newton. Preserve the scalar warning
    // multiplicity for a non-simplex supplied composition; ordinary feasible
    // composition can share the node coefficient work without extra warnings.
    R seedIndependentSum=0;
    for(int j=0;j<N-1;++j)seedIndependentSum+=in.matching.massFraction[ctx.independent[j]];
    bool seeded=accepted!=nullptr;
    if(accepted){}
    else if(seedIndependentSum<=R(1))seeded=seedSstLayer(in,c,ctx,w,team);
    else if(team.leader())seeded=seedSstLayer(in,c,ctx,w,SerialWallTeam{});
    // Use the positive Picard solution directly, without a shear-based k
    // injection or physical floor. A tiny positive matching k can support a
    // finite interior turbulent branch; assess the accepted equation residual
    // separately from that initializer's branch choice.
    seeded=team.broadcast(seeded);
    if(!seeded){if(team.leader()){fail(status,WallCode::InadmissibleState);wallFatal(in,ugkp::diagnostics::Code::NonFinite,2420);}return false;}
    team.sync();
    R norm=0;
    const bool measured=allResidual(in,c,ctx,w.state,w.y,w.residual,norm,team);
    if(!measured){if(team.leader())fail(status,WallCode::InadmissibleState);return false;}
    const R tolerance=c.absoluteTolerance+c.relativeTolerance;
    const R initialNorm=norm;
    R lastAlpha=0;
    bool stalledUpdate=false,newtonAttempted=false;
    int iteration=0;
    for(;norm>tolerance&&iteration<c.maxIterations;++iteration){
        if(!newtonStep(in,c,ctx,w,team)){
            if(team.leader())wallFatal(in,ugkp::diagnostics::Code::ZeroDenominator,2421);
            return false;
        }
        R alpha=0;newtonAttempted=true;
        if(!applyLayerNewtonUpdate(in,c,ctx,w,norm,alpha,team))return false;
        lastAlpha=alpha;
        if(alpha==0){stalledUpdate=true;break;}
    }
    if(team.leader()){status.iteration=iteration;status.residual=double(norm);}
    // A finite, admissible iterate is not a solved wall boundary condition.
    // In particular, its extrapolated matching flux can be arbitrarily large.
    // Keep output untouched on failure so it cannot become a ready closure.
    if(!gasFinite(norm)||norm>tolerance){
        if(team.leader()){
            // Failure-only observation: no additional residual evaluation or
            // alteration of the accepted state, iteration budget or tolerance.
            int maximumNode=0,maximumComponent=0;
            R maximumResidual=abs(w.residual[0][0]);
            for(int i=0;i<c.nodes;++i)for(int j=0;j<N+4;++j){
                const R value=abs(w.residual[i][j]);
                if((gasFinite(maximumResidual)&&!gasFinite(value))
                    ||value>maximumResidual){
                    maximumResidual=value;maximumNode=i;maximumComponent=j;
                }
            }
            const char* reason=!gasFinite(norm)?"nonfinite":
                (stalledUpdate?"lineSearchStalled":
                 (iteration>=c.maxIterations?"iterationBudget":"other"));
            const R lastDelta=newtonAttempted?w.delta[maximumNode][maximumComponent]:R(0);
            printf("WALL_REACTING_NONCONVERGENCE face=%d nodes=%d iteration=%d newtonAttempts=%d maxIterations=%d reason=%s initialResidual=%.17g finalResidual=%.17g tolerance=%.17g relativeTolerance=%.17g absoluteTolerance=%.17g maximumNode=%d maximumComponent=%d signedResidual=%.17g componentScale=%.17g stateValue=%.17g lastNewtonDelta=%.17g lastAcceptedAlpha=%.17g\n",
                int(in.diagnosticEntity),c.nodes,iteration,iteration+(stalledUpdate?1:0),c.maxIterations,reason,
                double(initialNorm),double(norm),double(tolerance),double(c.relativeTolerance),double(c.absoluteTolerance),
                maximumNode,maximumComponent,double(w.residual[maximumNode][maximumComponent]),
                double(ctx.scale[maximumComponent]),double(w.state[maximumNode][maximumComponent]),
                double(lastDelta),double(lastAlpha));
            fail(status,WallCode::NonConvergence);
            using namespace ugkp::diagnostics;
            if(in.diagnostics)recordFatal(in.diagnostics,makeRecord(
                gasFinite(norm)?Code::InvalidPhysicalState:Code::NonFinite,2422,23,in.diagnosticNode,
                EntityKind::Face,in.diagnosticEntity,-1,double(norm),double(tolerance),0,Quantity::Accuracy));
        }
        return false;
    }
    const bool terminalProfile=false;
    bool published=false;
    if(team.leader()){published=publishLayer(in,c,ctx,w,iteration,norm,output,terminalProfile);
        if(!published){fail(status,WallCode::InadmissibleState);wallFatal(in,ugkp::diagnostics::Code::NonFinite,10022);}}
    return team.broadcast(published);
}
// Keep the original <R,N,M> entry-point spelling while deducing borrowed tables.
template<class R,int N,int M,class TableReal=R> UGKWP_GAS_HD inline typename std::enable_if<std::is_same<R,WallReal>::value,bool>::type evaluateReactingWallLayer(
 const WallInput<R,N,TableReal>& input,const WallModelConfig<R>& config,WallWorkspace<R,N,M>& workspace,
 WallOutput<R,N>& output,WallStatus& status){
 return evaluateReactingWallLayerImpl(input,config,workspace,output,status);
}
template<class S,int N,int M> UGKWP_GAS_HD inline typename std::enable_if<!std::is_same<S,WallReal>::value,bool>::type evaluateReactingWallLayer(
 const WallInput<S,N>& input,const WallModelConfig<S>& config,WallWorkspace<S,N,M>& workspace,
 WallOutput<S,N>& output,WallStatus& status){
 const auto promoted=detail::wallComputationInput(input);const auto controls=detail::wallComputationConfig(config);
 WallOutput<WallReal,N> result;
 if(!evaluateReactingWallLayerImpl(promoted,controls,workspace,result,status))return false;
 if(!detail::publishWallPrecision(input,result,output))return detail::fail(status,WallCode::InadmissibleState);
 return true;
}
}} // namespace

#undef UGKWP_WALL_POINT_INLINE
#undef UGKWP_WALL_DEVICE_LEAF
