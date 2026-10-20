// Unfused cc392dd4 publication traversal, with current total-k flux semantics.
// Tests compare every published field with the fused implementation.
#pragma once
namespace ugkwp { namespace gaswall { namespace detail {
template<class R,int N,int M,class TableReal> UGKWP_GAS_HD inline bool referencePublishLayer(const WallInput<R,N,TableReal>& in,const WallModelConfig<R>& c,
 const LayerContext<R,N>& ctx,WallWorkspace<R,N,M>& w,int iterations,R norm,WallOutput<R,N>& output,const bool terminalProfile=false){
    WallOutput<R,N> out;LayerPoint<R,N> wall;if(!pointState(in,c,ctx,w.state[0],wall,terminalProfile))return false;
    if(!trace(in,wall.Y,out,terminalProfile))return false;
    LayerFlux<R,N> first,last;if(!intervalFlux(in,c,ctx,w.state[0],w.state[1],w.y[0],w.y[1],first,terminalProfile)
        ||!intervalFlux(in,c,ctx,w.state[c.nodes-2],w.state[c.nodes-1],w.y[c.nodes-2],w.y[c.nodes-1],last,terminalProfile))return false;
    for(int d=0;d<2;++d){const R tau=-first.momentum[d];for(int a=0;a<3;++a)out.traction[a]+=tau*ctx.tangent[d][a];}
    R wallEnergy=ctx.mass*wall.normalVelocity*wall.normalVelocity/R(2);
    for(int j=0;j<N;++j)wallEnergy+=in.massFlux[j]*(ctx.wallEnthalpy[j]-ctx.energyReference);
    out.conductiveHeatFlux=first.energy-wallEnergy;
    R resistance=0,advectiveIntegral=0,viscousIntegral=0;bool heatAuditAvailable=true;
    for(int i=0;i+1<c.nodes;++i){LayerFlux<R,N> f;
        if(!intervalFlux(in,c,ctx,w.state[i],w.state[i+1],w.y[i],w.y[i+1],f,terminalProfile))return false;
        if(!(f.conductivity>0)){heatAuditAvailable=false;continue;}
        const R r=(w.y[i+1]-w.y[i])/f.conductivity;
        if(!gasFinite(r)||r>std::numeric_limits<R>::max()-resistance){heatAuditAvailable=false;continue;}
        const R total=resistance+r,a=resistance/total,b=r/total;
        advectiveIntegral=a*advectiveIntegral+b*f.advectiveEnergy;
        viscousIntegral=a*viscousIntegral+b*f.stressWork;resistance=total;
    }
    out.heatConduction=heatAuditAvailable&&resistance>0?(in.temperature-in.matching.temperature)/resistance:std::numeric_limits<R>::quiet_NaN();
    out.heatViscous=heatAuditAvailable?viscousIntegral:std::numeric_limits<R>::quiet_NaN();
    out.heatTransport=heatAuditAvailable?advectiveIntegral-wallEnergy:std::numeric_limits<R>::quiet_NaN();
    out.heatBalanceResidual=out.heatConduction+out.heatViscous+out.heatTransport-out.conductiveHeatFlux;
    // The matching endpoint also has a half control volume. Its source is
    // included consistently in both the diagnostic integral and edge flux.
    for(int i=0;i<c.nodes;++i){LayerPoint<R,N> p;
        // Chemistry publication needs only this node's state and reaction.
        // Neighbor reconstruction/SST sources serve the optional diagnostic.
        if(in.integrateOwnerKSource){
            if(!evaluatePoint(in,c,ctx,w.state,w.y,i,p,terminalProfile))return false;
        }else if(!pointState(in,c,ctx,w.state[i],p,terminalProfile)||!pointReaction(in,p,terminalProfile))return false;
        if(in.integrateOwnerKSource){w.seed[i][0]=p.sourceK;
            w.seed[i][1]=i>0?p.rho*in.model.sstCoefficients.betaStar*p.k*p.omega:R(0);
            w.seed[i][2]=p.rho*p.k;}
        const R width=i==0?w.y[1]/R(2):(i==c.nodes-1?(w.y[i]-w.y[i-1])/R(2):(w.y[i+1]-w.y[i-1])/R(2));
        for(int j=0;j<N;++j)out.reactionIntegral[j]+=width*p.source[j];
        if(i==c.nodes-1)for(int j=0;j<N;++j)out.matchingSpeciesFlux[j]=last.species[j]+width*p.source[j];}
    for(int j=0;j<N;++j){out.speciesBalanceResidual[j]=out.matchingSpeciesFlux[j]-in.massFlux[j]-out.reactionIntegral[j];
        out.chemistryMismatch[j]=in.coarseReactionIntegral?out.reactionIntegral[j]-in.coarseReactionIntegral[j]:std::numeric_limits<R>::quiet_NaN();}
    out.chemistryMismatchAvailable=in.coarseReactionIntegral!=nullptr;
    out.wallKFlux=first.k;
    if(c.enableSst){const auto& q=in.quadrature;
        const R slope=::sqrt(wall.rho*in.model.sstCoefficients.beta1/(R(6)*in.model.viscosity));
        const R ell=w.state[0][N+3]/slope;
        const R power=(R(1)+::sqrt(R(1)+R(24)*in.model.sstCoefficients.betaStar/in.model.sstCoefficients.beta1))/R(2);
        const R advection=ctx.mass/(R(2)*in.model.viscosity),firstK=w.state[1][N+2]*w.state[1][N+2];
        out.wallKFlux=-in.model.viscosity*firstK*nearWallKDerivativeAtWall(w.y[1],ell,power,advection);
        const R matchingHalf=(w.y[c.nodes-1]-w.y[c.nodes-2])/R(2);
        const R matchingKFlux=last.k+(in.volumeSstClosure?matchingHalf*w.seed[c.nodes-1][0]:R(0));
        if(in.volumeSstClosure){out.kFluxCount=c.nodes+1;
            out.kFluxHeight[0]=0;out.kTransportFlux[0]=out.wallKFlux;w.seed[0][3]=out.wallKFlux;
            for(int i=1;i<c.nodes;++i){LayerFlux<R,N> f;LayerPoint<R,N> a,b;
                if(!intervalFlux(in,c,ctx,w.state[i-1],w.state[i],w.y[i-1],w.y[i],f)
                    ||!pointState(in,c,ctx,w.state[i-1],a)||!pointState(in,c,ctx,w.state[i],b))return false;
                out.kFluxHeight[i]=(w.y[i-1]+w.y[i])/R(2);
                out.kTransportFlux[i]=f.k;w.seed[i][3]=f.k;
                out.omegaDiffusiveFlux[i]=f.omega-ctx.mass*(a.omega+b.omega)/R(2);
            }
            // Matching data are imposed at the macro interface, not at the
            // final dual face. Include its remaining half control volume.
            const int end=c.nodes-1;LayerPoint<R,N> edge,previous;
            if(!evaluatePoint(in,c,ctx,w.state,w.y,end,edge)
                ||!pointState(in,c,ctx,w.state[end-1],previous))return false;
            R average;
            if(end==1){
                // The minimum two-node grid starts at z_wall=0 for a smooth
                // wall. Integrate the existing linear-z asymptote only over
                // its positive outer half; omega at the wall is not sampled.
                const R a=R(2)/(previous.z+edge.z),b=R(1)/edge.z;
                average=a*b*(a*a+a*b+b*b)/R(3);
            }else average=powerIntegral(previous.omega*previous.omega,edge.omega*edge.omega,
                w.y[end-1],w.y[end],(w.y[end-1]+w.y[end])/R(2),w.y[end])/matchingHalf;
            const R omegaSource=edge.sourceOmega+edge.rho*sstBeta(edge.f1,in.model.sstCoefficients)*(edge.omega*edge.omega-average);
            out.kFluxHeight[c.nodes]=w.y[end];
            out.kTransportFlux[c.nodes]=matchingKFlux;
            out.omegaDiffusiveFlux[c.nodes]=last.omega+matchingHalf*omegaSource-ctx.mass*edge.omega;
        }
        if(in.integrateOwnerKSource){
        if(q.count>0&&(!q.distance||!q.volumeWeight))return wallFatal(in,ugkp::diagnostics::Code::MissingOperator,10017);
        for(int qi=0;qi<q.count;++qi){const R y=q.distance[qi],weight=q.volumeWeight[qi];
            if(!gasFinite(y)||!gasFinite(weight))return wallFatal(in,ugkp::diagnostics::Code::NonFinite,2419,y);
            int l=0;while(l+1<c.nodes-1&&w.y[l+1]<y)++l;
            const R fraction=(y-w.y[l])/(w.y[l+1]-w.y[l]);
            int control=0;
            if(in.volumeSstClosure){
                while(control+1<c.nodes&&out.kFluxHeight[control+1]<y)++control;
                if(control+1<out.kFluxCount)out.ownerKBalanceResidual-=weight*((control+1==c.nodes?matchingKFlux:w.seed[control+1][3])-w.seed[control][3])/(out.kFluxHeight[control+1]-out.kFluxHeight[control]);
                // The physical SST residual is never replaced by a flux
                // identity. Interior dual volumes use the solved equation's
                // actual nodal source, including an unconverged residual.
                if(control>0)out.integratedKSource+=weight*w.seed[control][0];
            }

            // Reconstruct the already-discretized full source with positive
            // nodal weights. Recomputing it with interval secant gradients
            // destroys consistency when local production/destruction cancel.
            if(l>0||fraction==R(1)){
                if(!in.volumeSstClosure)out.integratedKSource+=weight*((R(1)-fraction)*w.seed[l][0]+fraction*w.seed[l+1][0]);
                out.ownerKDestruction+=weight*((R(1)-fraction)*w.seed[l][1]+fraction*w.seed[l+1][1]);
                out.ownerKContent+=weight*((R(1)-fraction)*w.seed[l][2]+fraction*w.seed[l+1][2]);continue;}
            R state[N+4];for(int j=0;j<N+4;++j)state[j]=w.state[l][j]+fraction*(w.state[l+1][j]-w.state[l][j]);
            if(l==0)state[N+2]=::sqrt(firstK*nearWallKShape(y,w.y[1],ell,power,advection));
            LayerPoint<R,N> p,a,b;if(!pointState(in,c,ctx,state,p,terminalProfile)||!pointState(in,c,ctx,w.state[l],a,terminalProfile)||!pointState(in,c,ctx,w.state[l+1],b,terminalProfile))return false;
            if(l==0){const R dy=w.y[1];
                const R dk=firstK*nearWallKDerivative(y,dy,ell,power,advection);
                applySstGradients(in,y,(b.u[0]-a.u[0])/dy,(b.u[1]-a.u[1])/dy,((b.normalCoefficient+a.normalCoefficient)/R(2)*(b.theta-a.theta)+(b.T+a.T)/R(2)*(b.normalCoefficient-a.normalCoefficient))/dy,dk,-R(2)*(b.z-a.z)/(dy*p.z*p.z*p.z),p);}
            else pointSst(in,c,y,a,b,w.y[l+1]-w.y[l],p);
            if(!in.volumeSstClosure||control==0)out.integratedKSource+=weight*p.sourceK;
            out.ownerKDestruction+=weight*p.rho*in.model.sstCoefficients.betaStar*p.k*p.omega;
            out.ownerKContent+=weight*p.rho*p.k;}
        out.ownerKBalanceResidual+=out.integratedKSource;
        }else out.integratedKSource=std::numeric_limits<R>::quiet_NaN();
        int l=0;while(l+1<c.nodes-1&&w.y[l+1]<in.ownerDistance)++l;
        const R fraction=(in.ownerDistance-w.y[l])/(w.y[l+1]-w.y[l]);const R z=w.state[l][N+3]+fraction*(w.state[l+1][N+3]-w.state[l][N+3]);
        out.ownerOmega=R(1)/(z*z);
        if(in.volumeSstClosure&&out.ownerKContent>0)
            out.ownerOmega=out.ownerKDestruction/(in.model.sstCoefficients.betaStar*out.ownerKContent);

        }


    if(!gasFinite(out.conductiveHeatFlux))return wallFatal(in,ugkp::diagnostics::Code::NonFinite,2426,out.conductiveHeatFlux);
    if(!gasFinite(out.wallKFlux))return wallFatal(in,ugkp::diagnostics::Code::NonFinite,31300,out.wallKFlux);
    if(!gasFinite(out.ownerOmega))return wallFatal(in,ugkp::diagnostics::Code::NonFinite,31301,out.ownerOmega);
    if(in.volumeSstClosure&&(!gasFinite(out.ownerKContent)||!gasFinite(out.ownerKDestruction)||!gasFinite(out.integratedKSource)))
        return wallFatal(in,ugkp::diagnostics::Code::NonFinite,2419,!gasFinite(out.ownerKContent)?out.ownerKContent:(!gasFinite(out.ownerKDestruction)?out.ownerKDestruction:out.integratedKSource));
    // Standalone optional integrals remain audits; production closure values
    // above participate in the physical operator and must remain available.
    if(in.integrateOwnerKSource&&!gasFinite(out.integratedKSource))
        wallWarning(in,ugkp::diagnostics::Code::AccuracyResidual,2429,out.integratedKSource,std::numeric_limits<R>::max());
    for(int d=0;d<3;++d)if(!gasFinite(out.traction[d]))return wallFatal(in,ugkp::diagnostics::Code::NonFinite,10018,out.traction[d],0,d);
    for(int j=0;j<N;++j){
        // Guard the physical fluxes, not overflow in their diagnostic difference.
        if(!gasFinite(out.wallSpeciesFlux[j]))return wallFatal(in,ugkp::diagnostics::Code::NonFinite,10019,out.wallSpeciesFlux[j],0,j);
        if(!gasFinite(out.matchingSpeciesFlux[j]))return wallFatal(in,ugkp::diagnostics::Code::NonFinite,31302,out.matchingSpeciesFlux[j],0,j);
        if(!gasFinite(out.reactionIntegral[j]))return wallFatal(in,ugkp::diagnostics::Code::NonFinite,31303,out.reactionIntegral[j],0,j);
        if(!gasFinite(out.speciesBalanceResidual[j])||abs(out.speciesBalanceResidual[j])>c.absoluteTolerance+c.relativeTolerance)
            wallWarning(in,ugkp::diagnostics::Code::ConservationResidual,2427,out.speciesBalanceResidual[j],c.absoluteTolerance+c.relativeTolerance,0,j);
    }
    out.iterations=iterations;out.residual=norm;output=out;return true;
}
}}}
