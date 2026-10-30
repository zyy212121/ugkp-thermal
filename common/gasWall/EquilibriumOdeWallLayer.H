#pragma once
#include "ReactingWallLayer.H"

namespace ugkwp { namespace gaswall {
// Local, steady ODE transport with the published semilocal EWM2 viscosity.
// The species/enthalpy/normal-advection equations and wall flux publication
// are shared with ReactingWallLayer. No local k/omega inventory is invented.
// The wall-stress scalar closes the constitutive law; it is not an imposed
// target stress or an inventory constraint. Both transport and this scalar
// consistency equation must converge before anything is published.
template<class R,int N,int M,class TableReal,class Team=detail::SerialWallTeam>
UGKWP_GAS_HD inline bool evaluateEquilibriumOdeWallLayerImpl(
 const WallInput<R,N,TableReal>& originalInput,const WallModelConfig<R>& controls,
 WallWorkspace<R,N,M>& w,WallOutput<R,N>& output,WallStatus& status,
 const Team& team=Team{},const R* accepted=nullptr,R acceptedStress=0){
    using namespace detail;
    auto c=controls;c.enableSst=false;c.coherentDiscretization=false;
    auto in=originalInput;in.model.thermo.diagnostics=in.diagnostics;
    in.integrateOwnerKSource=false;in.volumeSstClosure=false;
    if(!prepareReactingWallInput<M>(in,c,status,team))return false;
    const bool parameters=gasFinite(c.equilibriumKappa)&&c.equilibriumKappa>=0
        &&gasFinite(c.equilibriumAPlus)&&c.equilibriumAPlus>0
        &&wallMolecularViscosity(in,in.matching.temperature,in.matching.massFraction)>0&&in.matchingDistance>0
        &&c.maxIterations>0&&c.relativeTolerance>=0&&c.absoluteTolerance>=0;
    if(!team.all(parameters)){
        if(team.leader())fail(status,WallCode::InvalidInput);
        return false;
    }
    LayerContext<R,N> ctx;
    if(!team.all(makeContext(in,c,ctx))){
        if(team.leader())fail(status,WallCode::InvalidThermodynamics);
        return false;
    }
    const R speed=::sqrt(ctx.edgeU[0]*ctx.edgeU[0]+ctx.edgeU[1]*ctx.edgeU[1]);
    const R laminarStress=wallMolecularViscosity(in,in.matching.temperature,in.matching.massFraction)*speed/in.matchingDistance;
    ctx.equilibriumWallStress=accepted?acceptedStress:laminarStress;
    const bool localStress=c.localStressCorrection&&ctx.mass!=0;
    for(int d=0;d<2;++d)ctx.equilibriumWallTraction[d]=speed>0
        ?ctx.equilibriumWallStress*ctx.edgeU[d]/speed:R(0);
    for(int i=team.rank();i<c.nodes;i+=team.size()){
        const R fraction=::pow(R(i)/R(c.nodes-1),c.stretch);
        w.y[i]=in.matchingDistance*fraction;
        if(accepted){for(int j=0;j<N+4;++j)w.state[i][j]=accepted[i*(N+4)+j];continue;}
        for(int d=0;d<2;++d)w.state[i][d]=fraction*ctx.edgeU[d];
        w.state[i][2]=fraction*(in.matching.temperature-in.temperature);
        for(int j=0;j<N-1;++j)w.state[i][3+j]=in.matching.massFraction[ctx.independent[j]];
        w.state[i][N+2]=0;w.state[i][N+3]=1;
    }
    team.sync();
    const R tolerance=c.absoluteTolerance+c.relativeTolerance;
    R norm=std::numeric_limits<R>::infinity();
    R previousStress=0,previousDefect=0,lower=0,upper=0;
    bool previous=false,hasLower=false,hasUpper=false;
    int updates=0;
    for(int outer=0;outer<c.maxIterations;++outer){
        if(!allResidual(in,c,ctx,w.state,w.y,w.residual,norm,team)){
            if(team.leader())fail(status,WallCode::InadmissibleState);
            return false;
        }
        while(norm>tolerance&&updates<c.maxIterations){
            if(!newtonStep(in,c,ctx,w,team)){
                if(team.leader())fail(status,WallCode::SingularJacobian);
                return false;
            }
            R alpha=0;
            if(!applyLayerNewtonUpdate(in,c,ctx,w,norm,alpha,team)){
                if(team.leader())fail(status,WallCode::InadmissibleState);
                return false;
            }
            ++updates;
            if(alpha==0)break;
        }
        if(!gasFinite(norm)||norm>tolerance)break;
        R measuredStress=0,measuredTraction[2]{};bool measured=true;
        if(team.leader()){
            LayerFlux<R,N> first;
            measured=intervalFlux(in,c,ctx,w.state[0],w.state[1],w.y[0],w.y[1],first);
            if(measured)measuredStress=::sqrt(first.momentum[0]*first.momentum[0]
                                            +first.momentum[1]*first.momentum[1]);
            if(measured)for(int d=0;d<2;++d)measuredTraction[d]=-first.momentum[d];
            measured=measured&&gasFinite(measuredStress);
        }
        if(!team.broadcast(measured)){
            if(team.leader())fail(status,WallCode::InadmissibleState);
            return false;
        }
        measuredStress=team.broadcast(measuredStress);
        if(localStress)for(int d=0;d<2;++d)measuredTraction[d]=team.broadcast(measuredTraction[d]);
        const R stress=ctx.equilibriumWallStress,defect=measuredStress-stress;
        const R scale=max(laminarStress,max(stress,measuredStress));
        const R stressResidual=c.equilibriumKappa==0?R(0):(scale>0?abs(defect)/scale:R(0));
        norm=max(norm,stressResidual);
        if(localStress&&c.equilibriumKappa!=0){
            const R vectorDefect=::hypot(measuredTraction[0]-ctx.equilibriumWallTraction[0],
                measuredTraction[1]-ctx.equilibriumWallTraction[1]);
            norm=max(norm,scale>0?vectorDefect/scale:vectorDefect);
        }
        if(norm<=tolerance){
            bool published=false;
            if(team.leader()){
                status.code=WallCode::Success;status.iteration=updates;status.residual=double(norm);
                published=publishLayer(in,c,ctx,w,updates,norm,output);
                if(!published)fail(status,WallCode::InadmissibleState);
            }
            return team.broadcast(published);
        }
        if(updates>=c.maxIterations)break;
        // A safeguarded secant update accelerates the scalar fixed point.
        // Bounds are obtained only from evaluated signs, never prescribed
        // by a target wall law, macro inventory, or fitting factor.
        if(defect>0){lower=stress;hasLower=true;}
        else {upper=stress;hasUpper=true;}
        R next=measuredStress;
        if(previous&&defect!=previousDefect){
            const R secant=stress-defect*(stress-previousStress)/(defect-previousDefect);
            if(gasFinite(secant)&&secant>=0&&(!hasLower||secant>lower)
                &&(!hasUpper||secant<upper))next=secant;
        }
        if(hasLower&&hasUpper&&(!(next>lower)||!(next<upper)))next=(lower+upper)/R(2);
        if(!gasFinite(next)||next<0)break;
        previous=true;previousStress=stress;previousDefect=defect;
        ctx.equilibriumWallStress=next;
        if(localStress)for(int d=0;d<2;++d)ctx.equilibriumWallTraction[d]=measuredStress>0
            ?next*measuredTraction[d]/measuredStress:R(0);
        ++updates;
    }
    if(team.leader()){
        fail(status,WallCode::NonConvergence);
        status.iteration=updates;status.residual=double(norm);
        // Preserve the numerical failure before the application adapter adds
        // its generic unavailable-output diagnostic.
        using namespace ugkp::diagnostics;
        if(in.diagnostics)recordFatal(in.diagnostics,makeRecord(
            gasFinite(norm)?Code::InvalidPhysicalState:Code::NonFinite,2422,23,in.diagnosticNode,
            EntityKind::Face,in.diagnosticEntity,-1,double(norm),double(tolerance),0,Quantity::Accuracy));
    }
    return false;
}

template<class R,int N,int M,class TableReal=R>
UGKWP_GAS_HD inline typename std::enable_if<std::is_same<R,WallReal>::value,bool>::type
evaluateEquilibriumOdeWallLayer(const WallInput<R,N,TableReal>& in,
 const WallModelConfig<R>& c,WallWorkspace<R,N,M>& w,WallOutput<R,N>& out,WallStatus& status){
    return evaluateEquilibriumOdeWallLayerImpl(in,c,w,out,status);
}
template<class S,int N,int M>
UGKWP_GAS_HD inline typename std::enable_if<!std::is_same<S,WallReal>::value,bool>::type
evaluateEquilibriumOdeWallLayer(const WallInput<S,N>& in,
 const WallModelConfig<S>& c,WallWorkspace<S,N,M>& w,WallOutput<S,N>& out,WallStatus& status){
    const auto promoted=detail::wallComputationInput(in);
    const auto controls=detail::wallComputationConfig(c);
    WallOutput<WallReal,N> result;
    if(!evaluateEquilibriumOdeWallLayerImpl(promoted,controls,w,result,status))return false;
    if(!detail::publishWallPrecision(in,result,out))return detail::fail(status,WallCode::InadmissibleState);
    return true;
}
}}
