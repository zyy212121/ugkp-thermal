// Uncached scalar oracle for seed property reuse. Updated for the consistent
// face/sink discretization; the residual and captured-profile tests independently
// validate that discretization. This oracle only tests cache transparency.
namespace ugkwp { namespace gaswall { namespace detail {
template<class R,int N,int M,class TableReal,class Team=SerialWallTeam> UGKWP_GAS_HD inline bool seedSstLayerReference(const WallInput<R,N,TableReal>& in,
 const WallModelConfig<R>& c,const LayerContext<R,N>& ctx,WallWorkspace<R,N,M>& w,const Team& team=Team{}){
    if(!c.enableSst)return true;
    const int K=N+2,Z=N+3;const auto& coefficients=in.model.sstCoefficients;
    R wallRho=0,slope=0;bool wallValid=false;
    if(team.leader()){
        LayerPoint<R,N> wall;wallValid=pointState(in,c,ctx,w.state[0],wall);
        if(wallValid){wallRho=wall.rho;slope=::sqrt(wallRho*coefficients.beta1/(R(6)*in.model.viscosity));}
    }
    if(!team.broadcast(wallValid))return false;
    slope=team.broadcast(slope);
    for(int i=1+team.rank();i<c.nodes;i+=team.size()){const R omega=R(1)/(slope*slope*w.y[i]*w.y[i])+in.matching.omega;
        w.state[i][Z]=R(1)/::sqrt(omega);}
    team.sync();
    if(team.leader())w.state[c.nodes-1][Z]=R(1)/::sqrt(in.matching.omega);
    team.sync();
    for(int phase=ctx.mass>0?0:1;phase<2;++phase){
    for(int iteration=0;iteration<200;++iteration){
        bool valid=true;
        for(int i=team.rank();i<c.nodes;i+=team.size()){LayerPoint<R,N> p;if(!evaluatePoint(in,c,ctx,w.state,w.y,i,p)){valid=false;break;}
            w.seed[i][0]=in.model.viscosity+p.rho*p.nut;
            w.seed[i][1]=in.model.viscosity+p.rho*sstAlphaK(p.f1,coefficients)*p.nut;
            w.seed[i][2]=in.model.viscosity+p.rho*sstAlphaOmega(p.f1,coefficients)*p.nut;
            const R kRest=p.sourceK+p.rho*coefficients.betaStar*p.k*p.omega;
            w.seed[i][3]=max(R(0),kRest);
            w.seed[i][4]=p.rho*coefficients.betaStar*p.omega+(p.k>0?max(R(0),-kRest)/p.k:R(0));
            const R omegaRest=p.sourceOmega+p.rho*sstBeta(p.f1,coefficients)*p.omega*p.omega;
            w.seed[i][5]=max(R(0),omegaRest);
            w.seed[i][6]=p.rho*sstBeta(p.f1,coefficients)*p.omega+(p.omega>0?max(R(0),-omegaRest)/p.omega:R(0));
            // The omega equation integrates destruction over its dual cell;
            // retaining pointwise omega^2 here changes its near-wall balance.
            if(phase==1&&i>1&&i+1<c.nodes){
                const R za=w.state[i-1][Z],zb=w.state[i+1][Z];
                const R average=omegaSquareAverage(R(1)/(za*za),p.omega,R(1)/(zb*zb),w.y[i-1],w.y[i],w.y[i+1]);
                w.seed[i][6]+=p.rho*sstBeta(p.f1,coefficients)*(average/p.omega-p.omega);
            }
            w.seed[i][7]=p.omega;}
        if(!team.all(valid))return false;
        for(int face=team.rank();face<c.nodes-1;face+=team.size()){
            LayerPoint<R,N> a,b;
            if(!loadBasePoint(in,c,ctx,w.state[face],a)||!loadBasePoint(in,c,ctx,w.state[face+1],b)){valid=false;break;}
            const auto aa=a,bb=b;const R ya=w.y[face],yb=w.y[face+1],dy=yb-ya;
            pointSst(in,c,ya,aa,bb,dy,a);pointSst(in,c,yb,aa,bb,dy,b);
            auto* faceD=w.diagonal[face][0];
            faceD[0]=transportConductance(in.model.viscosity+a.rho*a.nut,in.model.viscosity+b.rho*b.nut);
            faceD[1]=(R(2)*in.model.viscosity+a.rho*sstAlphaK(a.f1,coefficients)*a.nut+b.rho*sstAlphaK(b.f1,coefficients)*b.nut)/R(2);
            if(face==0){const R ell=a.z/slope;
              const R power=(R(1)+::sqrt(R(1)+R(24)*coefficients.betaStar/coefficients.beta1))/R(2);
              faceD[1]*=dy*nearWallKDerivative(dy/R(2),dy,ell,power,ctx.mass/(R(2)*in.model.viscosity));}
            faceD[2]=(R(2)*in.model.viscosity+a.rho*sstAlphaOmega(a.f1,coefficients)*a.nut+b.rho*sstAlphaOmega(b.f1,coefficients)*b.nut)/R(2);
            if(a.omega>0&&b.omega>0&&a.omega!=b.omega)
                faceD[2]*=powerGradient(a.omega,b.omega,ya,yb,(ya+yb)/R(2))*dy/(b.omega-a.omega);
        }
        if(!team.all(valid))return false;
        // Boundary physics is uniform for this wall: evaluate it once, then
        // broadcast only the value needed by the independent omega recurrence.
        R wallZ=0,firstZ=0,firstOmega=0;bool boundaryValid=false;
        if(team.leader()){
            LayerFlux<R,N> first;boundaryValid=intervalFlux(in,c,ctx,w.state[0],w.state[1],w.y[0],w.y[1],first);
            if(boundaryValid){
                if(ctx.mass>0){R tau2=0;for(int d=0;d<2;++d){const R tau=-first.momentum[d];tau2+=tau*tau;}
                    const R ut=::sqrt(::sqrt(tau2)/wallRho);boundaryValid=ut>0;
                    if(boundaryValid){const R vp=ctx.mass/(wallRho*ut);
                        wallZ=::sqrt(in.model.viscosity/wallRho/(ut*ut)*vp*(R(1)+R(5)*vp)/R(25));}}
                if(boundaryValid){firstZ=wallZ+w.y[1]*slope;firstOmega=R(1)/(firstZ*firstZ);}
            }
        }
        if(!team.broadcast(boundaryValid))return false;
        firstOmega=team.broadcast(firstOmega);
        if(!solveSeedSystemsWithTeam(c,w,ctx.mass,ctx.edgeU[0],ctx.edgeU[1],
            in.matching.k,firstOmega,in.matching.omega,team))return false;
        R change=0;bool updated=true;
        if(team.leader()){
        for(int i=1;i<c.nodes-1;++i){for(int d=0;d<2;++d){change=max(change,abs(w.delta[i][d]-w.state[i][d])/max(R(1),abs(ctx.edgeU[d])));w.state[i][d]=(w.state[i][d]+w.delta[i][d])/R(2);}
            if(w.delta[i][K]<0||w.delta[i][Z]<=0){updated=false;break;}
            const R oldK=w.state[i][K]*w.state[i][K],oldOmega=w.seed[i][7];
            const R kScale=max(oldK,max(w.delta[i][K],in.matching.k));
            if(kScale>0)change=max(change,abs(w.delta[i][K]-oldK)/kScale);
            change=max(change,abs(w.delta[i][Z]-oldOmega)/oldOmega);
            w.state[i][K]=::sqrt((oldK+w.delta[i][K])/R(2));w.state[i][Z]=R(1)/::sqrt((oldOmega+w.delta[i][Z])/R(2));
            if(!gasFinite(w.state[i][K])||!gasFinite(w.state[i][Z])){updated=false;break;}}
        if(updated){w.state[0][Z]=wallZ;if(c.nodes>2)w.state[1][Z]=firstZ;}
        }
        if(!team.broadcast(updated))return false;
        change=team.broadcast(change);
        if(change<R(1e-5))break;
    }
    }
    return true;
}

}}}
