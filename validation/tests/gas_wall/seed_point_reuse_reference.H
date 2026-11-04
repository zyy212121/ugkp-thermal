// Test-only uncached seed policy from abe202ad. Earlier 677adfe8 seed
// predates the directed cross diffusion and consistent face conductances.
namespace ugkwp { namespace gaswall { namespace detail {
template<class R,int N,int M,class TableReal,class Team=SerialWallTeam> UGKWP_GAS_HD inline bool seedSstLayerReference(const WallInput<R,N,TableReal>& in,
 const WallModelConfig<R>& c,const LayerContext<R,N>& ctx,WallWorkspace<R,N,M>& w,const Team& team=Team{}){
    if(!c.enableSst)return true;
    const int K=N+2,Z=N+3;const auto& coefficients=in.model.sstCoefficients;
    R wallRho=0,wallMu=0,slope=0;bool wallValid=false;
    if(team.leader()){
        LayerPoint<R,N> wall;wallValid=pointState(in,c,ctx,w.state[0],wall);
        if(wallValid){wallRho=wall.rho;wallMu=wall.mu;slope=::sqrt(wallRho*coefficients.beta1/(R(6)*wallMu));}
    }
    if(!team.broadcast(wallValid))return false;
    slope=team.broadcast(slope);wallMu=team.broadcast(wallMu);
    for(int i=1+team.rank();i<c.nodes;i+=team.size()){const R omega=R(1)/(slope*slope*w.y[i]*w.y[i])+in.matching.omega;
        w.state[i][Z]=R(1)/::sqrt(omega);}
    team.sync();
    if(team.leader())w.state[c.nodes-1][Z]=R(1)/::sqrt(in.matching.omega);
    team.sync();
    bool reuseThermo=false;
    // A finite blowing wall omega depends nonlinearly on the shear.
    // Establish that positive shear branch with the local destruction seed,
    // then continue on the same profile with dual-volume destruction. Starting
    // the latter from the bare linear profile can instead converge the Picard
    // surrogate to a nearly shear-free basin that Newton cannot leave.
    // With no blowing the singular-wall seed needs only the consistent phase.
    for(int phase=ctx.mass>0?0:1;phase<2;++phase){
    for(int iteration=0;iteration<200;++iteration){
        bool valid=true,reusable=true;
        const SeedThermoRows<R,N,decltype(w.state)> base(w.state,w.lower,false);
        for(int i=team.rank();i<c.nodes;i+=team.size()){LayerPoint<R,N> p;if(!evaluatePoint(in,c,ctx,base,w.y,i,p)){valid=false;break;}
            // Once readers use the cache, never rewrite a neighbour's data
            // in this sweep (also across the blowing continuation phases).
            if(iteration==0&&!reuseThermo){
                storeSeedThermo(p,w.lower[i]);
                R sum=0;for(int j=0;j<N-1;++j){const R v=w.state[i][3+j];reusable=reusable&&v>=0&&v<=1;sum+=v;}
                // Clipping warnings must retain their original multiplicity.
                reusable=reusable&&sum<=1;
            }
            w.seed[i][0]=p.mu+p.rho*p.nut;
            w.seed[i][1]=p.mu+p.rho*sstAlphaK(p.f1,coefficients)*p.nut;
            w.seed[i][2]=p.mu+p.rho*sstAlphaOmega(p.f1,coefficients)*p.nut;
            const R kRest=p.sourceK+p.rho*coefficients.betaStar*p.k*p.omega;
            w.seed[i][3]=max(R(0),kRest);
            w.seed[i][4]=p.rho*coefficients.betaStar*p.omega+(p.k>0?max(R(0),-kRest)/p.k:R(0));
            R omegaRest=p.sourceOmega+p.rho*sstBeta(p.f1,coefficients)*p.omega*p.omega;
            const bool integratedOmega=phase==1&&i>1&&i+1<c.nodes;
            R omegaSquareMean=0;
            if(integratedOmega){
                // Phase 1 seeds the same directed CD as the Newton residual.
                // Only k/omega enter this integral; reuse the current nodal
                // states without another thermodynamic or SST evaluation.
                LayerPoint<R,N> a,b;
                const R za=w.state[i-1][Z],zb=w.state[i+1][Z];
                a.k=w.state[i-1][K]*w.state[i-1][K];a.omega=R(1)/(za*za);
                b.k=w.state[i+1][K]*w.state[i+1][K];b.omega=R(1)/(zb*zb);
                const R dyL=w.y[i]-w.y[i-1],dyR=w.y[i+1]-w.y[i],width=(dyL+dyR)/R(2);
                const R crossIntegral=omegaDirectedCrossIntegral(a,p,b,w.y[i-1],w.y[i],w.y[i+1],width);
                omegaRest=p.sourceOmegaWithoutCrossDiffusion
                    +p.rho*sstBeta(p.f1,coefficients)*p.omega*p.omega
                    +(R(2)*p.rho*coefficients.alphaOmega2*(R(1)-p.f1))*(crossIntegral/width);
                omegaSquareMean=omegaLocalSquareMean(p.omega,w.y[i],
                    (w.y[i-1]+w.y[i])/R(2),(w.y[i]+w.y[i+1])/R(2));
            }
            w.seed[i][5]=max(R(0),omegaRest);
            w.seed[i][6]=p.rho*sstBeta(p.f1,coefficients)*p.omega+(p.omega>0?max(R(0),-omegaRest)/p.omega:R(0));

            // The omega equation integrates destruction over its dual cell;
            // retaining pointwise omega^2 here changes its near-wall balance.
            if(integratedOmega){
                w.seed[i][6]+=p.rho*sstBeta(p.f1,coefficients)*(omegaSquareMean/p.omega-p.omega);
            }
            w.seed[i][7]=p.omega;}

        if(!team.all(valid))return false;
        if(iteration==0)reuseThermo=team.all(reusable);
        const SeedThermoRows<R,N,decltype(w.state)> faceBase(w.state,w.lower,false);
        // Newton's diagonal scratch is idle during initialization. Freeze
        // the same face diffusivities used by the physical residual, including
        // the first k interval and the positive geometry-only omega conductance.
        // A nodal arithmetic approximation seeds a different discrete problem
        // and leaves Newton to repair its mesh-dependent near-wall defect.
        for(int face=team.rank();face<c.nodes-1;face+=team.size()){
            LayerPoint<R,N> a,b;
            if(!loadBasePoint(in,c,ctx,faceBase[face],a)||!loadBasePoint(in,c,ctx,faceBase[face+1],b)){valid=false;break;}
            const auto aa=a,bb=b;const R ya=w.y[face],yb=w.y[face+1],dy=yb-ya;
            pointSst(in,c,ya,aa,bb,dy,a);pointSst(in,c,yb,aa,bb,dy,b);
            auto* faceD=w.diagonal[face][0];
            faceD[0]=transportConductance(a.mu+a.rho*a.nut,b.mu+b.rho*b.nut);
            faceD[1]=((a.mu+b.mu)+a.rho*sstAlphaK(a.f1,coefficients)*a.nut+b.rho*sstAlphaK(b.f1,coefficients)*b.nut)/R(2);
            if(face==0){const R ell=a.z/slope;
              const R power=(R(1)+::sqrt(R(1)+R(24)*coefficients.betaStar/coefficients.beta1))/R(2);
              faceD[1]*=dy*nearWallKDerivative(dy/R(2),dy,ell,power,ctx.mass/(R(2)*a.mu));}
            faceD[2]=((a.mu+b.mu)+a.rho*sstAlphaOmega(a.f1,coefficients)*a.nut+b.rho*sstAlphaOmega(b.f1,coefficients)*b.nut)/R(2);
            faceD[2]*=omegaGradientFit(ya,yb);
        }
        if(!team.all(valid))return false;
        // Boundary physics is uniform for this wall: evaluate it once, then
        // broadcast only the value needed by the independent omega recurrence.
        R wallZ=0,firstZ=0,firstOmega=0;bool boundaryValid=false;
        if(team.leader()){
            LayerFlux<R,N> first;boundaryValid=base.ready?intervalFlux(in,c,ctx,base[0],base[1],w.y[0],w.y[1],first):intervalFlux(in,c,ctx,w.state[0],w.state[1],w.y[0],w.y[1],first);
            if(boundaryValid){
                if(ctx.mass>0){R tau2=0;for(int d=0;d<2;++d){const R tau=-first.momentum[d];tau2+=tau*tau;}
                    const R ut=::sqrt(::sqrt(tau2)/wallRho);boundaryValid=ut>0;
                    if(boundaryValid){const R vp=ctx.mass/(wallRho*ut);
                        wallZ=::sqrt(wallMu/wallRho/(ut*ut)*vp*(R(1)+R(5)*vp)/R(25));}}
                if(boundaryValid){firstZ=wallZ+w.y[1]*slope;firstOmega=R(1)/(firstZ*firstZ);}
            }
        }
        if(!team.broadcast(boundaryValid))return false;
        firstOmega=team.broadcast(firstOmega);
        // Keep the scalar recurrence callsites and their FMA arithmetic.
        // Node thermochemistry/SST work is parallel; velocity RHS recurrences
        // share a factorization and retain the scalar arithmetic/scratch row.
        bool systems=false;
        if(team.leader())systems=solveSeedSystemsWithTeam(c,w,ctx.mass,ctx.edgeU[0],ctx.edgeU[1],
            in.matching.k,firstOmega,in.matching.omega,SerialWallTeam{});
        if(!team.broadcast(systems))return false;
        R change=0;bool updated=true;
        if(team.leader()){
        for(int i=1;i<c.nodes-1;++i){for(int d=0;d<2;++d){change=max(change,abs(w.delta[i][d]-w.state[i][d])/max(R(1),abs(ctx.edgeU[d])));w.state[i][d]=(w.state[i][d]+w.delta[i][d])/R(2);}
            if(w.delta[i][K]<0||w.delta[i][Z]<=0){updated=false;break;}
            const R oldK=w.state[i][K]*w.state[i][K],oldOmega=w.seed[i][7];
            // A tiny k is not a converged k. With max(1,k_match), an
            // exponentially growing positive Picard mode was discarded after
            // 11-13 sweeps while its omega/F1 coupling remained unresolved.
            // Zero/zero has exactly zero change; no physical floor is added.
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
        if(iteration==199&&team.leader())wallWarning(in,ugkp::diagnostics::Code::AccuracyResidual,2428,change,1e-5);
    }
    }
    return true;
}

} } }
