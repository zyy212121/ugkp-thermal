#pragma once
#include "WallPrecision.H"
#include "WallExecution.H"
#include "WallLayerMath.H"
#include "WallThermoMath.H"
#include "EquilibriumOdeMath.H"
#include "../gasTransport/SpeciesDiffusion.H"
// Keep repeated SST/thermochemistry leaves out of each capacity-specialized
// device body. This bounds nvcc optimization memory; host calls remain inline.
#if defined(__CUDA_ARCH__)
#define UGKWP_WALL_DEVICE_LEAF __noinline__
#else
#define UGKWP_WALL_DEVICE_LEAF inline
#endif
namespace ugkwp { namespace gaswall { namespace detail {
// Internal velocity is tangential relative to the wall; temperature is T-Tw.
// Fluxes subtract the corresponding constant momentum/energy frame offsets.
template<class R,int N> struct LayerContext {
    R tangent[2][3]{},wallU[2]{},edgeU[2]{},mass=0,wallNormal=0,energyReference=0,wallEnthalpy[N]{},enthalpyJump[N]{},scale[N+4]{};
    int dependent=0,independent[N]{};
    bool trialStorageComplete=false;
    R equilibriumWallStress=0; // fixed constitutive parameter during one transport Newton solve
};
template<class R,int N> struct LayerPoint {
    R mu=0,T=0,theta=0,normalCoefficient=0,Y[N]{},rho=0,cp=0,h[N]{},u[2]{},normalVelocity=0,k=0,z=0,omega=0;
    R nut=0,f1=0,f2=0,sourceK=0,sourceOmega=0,sourceOmegaWithoutCrossDiffusion=0,source[N]{};
};
// The same closure is used by nodal diagnostics and every conservative
// interval flux, including final publication and numerical Jacobian probes.
template<class R,int N,class TableReal> UGKWP_GAS_HD inline bool applyEquilibriumViscosity(
 const WallInput<R,N,TableReal>& in,const WallModelConfig<R>& c,
 const LayerContext<R,N>& ctx,R y,LayerPoint<R,N>& p){
    (void)in;
    if(c.model!=BoundaryLayerModel::EquilibriumOde)return true;
    p.nut=equilibriumDynamicViscosity(p.rho,ctx.equilibriumWallStress,y,
        p.mu,c.equilibriumKappa,c.equilibriumAPlus)/p.rho;
    return gasFinite(p.nut)&&p.nut>=0;
}
// log(b/a) without overflowing b/a or losing a small positive-state jump.
template<class R> UGKWP_GAS_HD inline R wallPositiveLogRatio(R b,R a){
    const R difference=b-a;
    if(abs(difference)<=a/R(2))return ::log1p(difference/a);
    return ::log(b)-::log(a);
}
// Integral of k'_h * (log omega_h)' on part of one interpolation interval.
// k_h uses the same physical secant as pointSst. Away from the wall omega_h
// is the positive c0+c1/y^2 reconstruction used by its diffusion flux
// and destruction integral. The y=0 interval instead retains pointSst's
// linear-z asymptote, z=1/sqrt(omega); its outer half remains finite when
// z_wall=0. No encoded wall omega=0 is treated as a physical value.
template<class R,int N> UGKWP_GAS_HD inline R omegaCrossDiffusionIntegral(
 const LayerPoint<R,N>& a,const LayerPoint<R,N>& b,R ya,R yb,R lo,R hi){
    const R invalid=std::numeric_limits<R>::quiet_NaN();
    if(!(yb>ya&&ya>=0&&lo>=ya&&hi<=yb&&hi>=lo))return invalid;
    if(a.k==b.k||hi==lo)return R(0);
    const R slope=(b.k-a.k)/(yb-ya);
    if(ya>0){
        if(!(a.omega>0&&b.omega>0))return invalid;
        if(a.omega==b.omega)return R(0);
        const R omegaLo=omegaInverseSquareValue(a.omega,b.omega,ya,yb,lo);
        const R omegaHi=omegaInverseSquareValue(a.omega,b.omega,ya,yb,hi);
        return slope*wallPositiveLogRatio(omegaHi,omegaLo);
    }
    if(!(a.z>=0&&b.z>0))return invalid;
    const R tLo=(lo-ya)/(yb-ya),tHi=(hi-ya)/(yb-ya);
    const R zLo=(R(1)-tLo)*a.z+tLo*b.z,zHi=(R(1)-tHi)*a.z+tHi*b.z;
    // A wall-touching integral of this linear-k source diverges for z_wall=0.
    // No omega PDE row uses it: the first interior row is the wall asymptote,
    // and two-node matching publication integrates only [y1/2,y1].
    if(!(zLo>0&&zHi>0))return invalid;
    return -R(2)*slope*wallPositiveLogRatio(zHi,zLo);
}
// Nodal SST coefficients remain frozen over this control volume, as in the
// existing omega^2 integration. Both residual and matching publication use
// this source: the integrated cross diffusion replaces its nodal counterpart.
template<class R,int N> UGKWP_GAS_HD inline R omegaSourceWithCrossDiffusionIntegral(
 const LayerPoint<R,N>& p,const SstCoefficientsT<R>& coefficients,
 R width,R omegaSquareMean,R crossIntegral){
    return p.sourceOmegaWithoutCrossDiffusion
        +p.rho*sstBeta(p.f1,coefficients)*(p.omega*p.omega-omegaSquareMean)
        +(R(2)*p.rho*coefficients.alphaOmega2*(R(1)-p.f1))*(crossIntegral/width);
}
// In the omega residual, -C*k'*(log omega)' is directed transport with
// velocity sign -k' (C>=0). A centred half-interval source can overturn the
// diffusion off-diagonal at large local Peclet number. Use the incoming
// logarithmic difference with the physical centred k gradient. No source
// coefficient, matching value, physical timestep or state is adjusted.
template<class R,int N> UGKWP_GAS_HD inline R omegaDirectedCrossIntegral(
 const LayerPoint<R,N>& a,const LayerPoint<R,N>& p,const LayerPoint<R,N>& b,
 R ya,R y,R yb,R width){
    const R left=y-ya,right=yb-y;
    const R dk=centeredDerivative(a.k,p.k,b.k,left,right);
    if(dk<0)return width*dk*wallPositiveLogRatio(p.omega,a.omega)/left;
    if(dk>0)return width*dk*wallPositiveLogRatio(b.omega,p.omega)/right;
    return R(0);
}
// The prescribed matching endpoint has a half CV and a one-sided derivative.
template<class R,int N> UGKWP_GAS_HD inline R omegaMatchingCrossIntegral(
 const LayerPoint<R,N>& a,const LayerPoint<R,N>& b,R ya,R yb,R width){
    if(a.k==b.k||a.omega==b.omega)return R(0);
    const R h=yb-ya;
    return width*((b.k-a.k)/h)*wallPositiveLogRatio(b.omega,a.omega)/h;
}
// Local destruction keeps the scalar reaction diagonal. The positive,
// geometry-only factor integrates the singular A/y^2 wall asymptote exactly
// and tends to one with second-order error on a smooth stretched mesh.
template<class R> UGKWP_GAS_HD inline R omegaLocalSquareMean(R omega,R y,R lo,R hi){
    return omega*omega*omegaDestructionFit(y,lo,hi);
}
template<class R,int N,class TableReal> UGKWP_GAS_HD inline bool validatePointDynamics(
 const WallInput<R,N,TableReal>& in,const LayerPoint<R,N>& p){
    if(!gasFinite(p.k)||!gasFinite(p.z)||p.z<0)return wallFatal(in,(gasFinite(p.k)&&gasFinite(p.z)?ugkp::diagnostics::Code::InvalidPhysicalState:ugkp::diagnostics::Code::NonFinite),2411,p.z);
    return true;
}
// This small, frequently called primitive benefits from device inlining;
// retain out-of-line boundaries around the larger SST and Newton routines.
#if defined(__CUDA_ARCH__)
#define UGKWP_WALL_POINT_INLINE __forceinline__
#else
#define UGKWP_WALL_POINT_INLINE inline
#endif
template<class R,int N,class TableReal> UGKWP_GAS_HD UGKWP_WALL_POINT_INLINE bool pointState(const WallInput<R,N,TableReal>& in,
 const WallModelConfig<R>& c,const LayerContext<R,N>& ctx,const R* state,LayerPoint<R,N>& p,const bool terminalProfile=false){
    if(!in.model.thermo.species||!in.model.thermo.coefficients)return wallFatal(in,ugkp::diagnostics::Code::MissingOperator,2400);
    p.theta=state[2];p.T=in.temperature+p.theta;p.u[0]=state[0];p.u[1]=state[1];p.k=state[N+2]*state[N+2];p.z=state[N+3];
    (void)terminalProfile;
    if(!gasFinite(p.T)||p.T<=0)return wallFatal(in,gasFinite(p.T)?ugkp::diagnostics::Code::NegativeTemperature:ugkp::diagnostics::Code::NonFinite,2410,p.T,ugkp::diagnostics::Quantity::Temperature);
    if(!validatePointDynamics(in,p))return false;
    R sum=0,gasR=0;
    for(int j=0;j<N-1;++j){if(!wallFraction(in,state[3+j],p.Y[ctx.independent[j]],ctx.independent[j]))return false;sum+=p.Y[ctx.independent[j]];}
    if(!wallFraction(in,R(1)-sum,p.Y[ctx.dependent],ctx.dependent))return false;
    for(int j=0;j<N;++j){
        R cp;speciesCpH(j,p.T,in.model.thermo,cp,p.h[j]);
        if(!gasFinite(p.h[j])||!gasFinite(cp)||cp<=0)return wallFatal(in,(gasFinite(cp)&&gasFinite(p.h[j])?ugkp::diagnostics::Code::InvalidPhysicalState:ugkp::diagnostics::Code::NonFinite),10006,cp,0,j);
        if(in.model.thermo.species[j].molarMass==0)return wallFatal(in,ugkp::diagnostics::Code::ZeroDenominator,2413,0,0,j);
        p.cp+=p.Y[j]*cp;gasR+=p.Y[j]*universalGasConstant<R>()/in.model.thermo.species[j].molarMass;}
    if(gasR==0||in.pressure==0)return wallFatal(in,ugkp::diagnostics::Code::ZeroDenominator,10007,gasR);
    p.rho=in.pressure/(gasR*p.T);p.normalCoefficient=ctx.mass*gasR/in.pressure;p.normalVelocity=ctx.wallNormal+p.normalCoefficient*p.T;
    if(!gasFinite(p.rho)||p.rho<=0||!gasFinite(p.normalVelocity))return wallFatal(in,(gasFinite(p.rho)&&gasFinite(p.normalVelocity)?ugkp::diagnostics::Code::InvalidPhysicalState:ugkp::diagnostics::Code::NonFinite),2414,p.rho,ugkp::diagnostics::Quantity::Density);
    p.mu=wallMolecularViscosity(in,p.T,p.Y);
    if(in.model.molecularViscosity.model!=MolecularViscosityModel::Constant&&(!gasFinite(p.mu)||!(p.mu>0)))return wallFatal(in,ugkp::diagnostics::Code::InvalidPhysicalState,2702,p.mu);
    if(c.enableSst&&p.z>0)p.omega=R(1)/(p.z*p.z);
    return true;
}


// Jacobian rows only need a five-node window around one perturbed node.
// Keep pristine pointState results private to its assembling thread; reaction
// and the distinct node/interval SST gradients always operate on fresh copies.
template<class R,int N> struct LayerPointCache {
    int first;LayerPoint<R,N> points[5];unsigned char status[5]{};
    UGKWP_GAS_HD explicit LayerPointCache(int begin):first(begin){}
    UGKWP_GAS_HD void invalidate(int node){status[node-first]=0;}
};
template<class R,int N> struct CachedLayerRow {
    const R* values;LayerPointCache<R,N>* cache;int node;
    UGKWP_GAS_HD R operator[](int component)const{return values[component];}
};
template<class R,int N,class Rows> struct CachedLayerRows {
    const Rows& rows;mutable LayerPointCache<R,N> cache;
    // Only conservative channels enter a residual. Four intervals cover the
    // five-node stencil; unchanged outer intervals survive derivative columns.
    mutable R fluxes[4][N+5]{};mutable unsigned char fluxStatus[4]{};
    UGKWP_GAS_HD CachedLayerRows(const Rows& values,int node)
        :rows(values),cache(node>1?node-2:0){}
    UGKWP_GAS_HD CachedLayerRow<R,N> operator[](int node)const{return {rows[node],&cache,node};}
    UGKWP_GAS_HD void invalidate(int node,bool velocityOnly=false){
        const int point=node-cache.first;
        // A velocity-only probe cannot change T, Y, rho, Cp, h or the
        // mass-prescribed normal velocity. Keep their successfully decoded
        // values, but refresh BOTH velocity components (the previous probe
        // may have changed the other one). Failed/empty entries are refilled.
        if(velocityOnly&&cache.status[point]==1){
            cache.points[point].u[0]=rows[node][0];
            cache.points[point].u[1]=rows[node][1];
            if(!gasFinite(rows[node][0])||!gasFinite(rows[node][1]))cache.invalidate(node);
        }else cache.invalidate(node);
        // Even with invariant thermodynamics the incident momentum/energy
        // fluxes depend on the probe and must always be evaluated again.
        for(int face=node-1;face<=node;++face){const int slot=face-cache.first;
            if(slot>=0&&slot<4)fluxStatus[slot]=0;}
    }
};
template<class R,int N,class TableReal> UGKWP_GAS_HD inline bool loadBasePoint(
 const WallInput<R,N,TableReal>& in,const WallModelConfig<R>& c,const LayerContext<R,N>& ctx,
 const R* row,LayerPoint<R,N>& point,const bool terminalProfile=false){
    return pointState(in,c,ctx,row,point,terminalProfile);
}
template<class R,int N,class TableReal> UGKWP_GAS_HD inline bool loadBasePoint(
 const WallInput<R,N,TableReal>& in,const WallModelConfig<R>& c,const LayerContext<R,N>& ctx,
 const CachedLayerRow<R,N>& row,LayerPoint<R,N>& point,const bool terminalProfile=false){
    (void)terminalProfile;
    auto& cache=*row.cache;const int slot=row.node-cache.first;
    if(!cache.status[slot]){
        // pointState accumulates Cp, so every refill starts from a zero object.
        cache.points[slot]=LayerPoint<R,N>{};
        cache.status[slot]=pointState(in,c,ctx,row.values,cache.points[slot])?1:2;
    }
    if(cache.status[slot]!=1)return false;
    point=cache.points[slot];return true;
}

// A sequential seed sweep needs only the current node and its two neighbors.
// Cache lazily in first-use order, so an invalid node preserves partial writes.
// Each sweep owns a new cache: no entry survives an update of the iterate.
template<class R,int N> struct SeedLayerRow {
    const R* values;LayerPoint<R,N>* points;int* nodes;unsigned char* status;int node;
};
template<class R,int N,class Rows> struct SeedLayerRows {
    const Rows& rows;mutable LayerPoint<R,N> points[3];
    mutable int nodes[3]{-1,-1,-1};mutable unsigned char status[3]{};
    UGKWP_GAS_HD explicit SeedLayerRows(const Rows& values):rows(values){}
    UGKWP_GAS_HD SeedLayerRow<R,N> operator[](int node)const{return {rows[node],points,nodes,status,node};}
};
template<class R,int N,class TableReal> UGKWP_GAS_HD inline bool loadBasePoint(
 const WallInput<R,N,TableReal>& in,const WallModelConfig<R>& c,const LayerContext<R,N>& ctx,
 const SeedLayerRow<R,N>& row,LayerPoint<R,N>& point,const bool terminalProfile=false){
    (void)terminalProfile;
    const int slot=row.node%3;
    if(row.nodes[slot]!=row.node){
        row.nodes[slot]=row.node;row.points[slot]=LayerPoint<R,N>{};
        row.status[slot]=pointState(in,c,ctx,row.values,row.points[slot])?1:2;
    }
    if(row.status[slot]!=1)return false;
    // Reaction and node SST closure modify this copy, never the cached base.
    point=row.points[slot];return true;
}

// The seed changes u, sqrt(k), and z only. Its Jacobian scratch is idle;
// four rows per node retain just the invariant thermodynamic channels. Populate
// after the first ordinary sweep, so failures keep their original evaluation
// order. Reaction and all SST gradients/sources are still evaluated each sweep.
template<class R,int N> struct SeedThermoRow {
    SeedLayerRow<R,N> base;const R (*thermo)[N+4];bool ready;
};
template<class R,int N,class Rows> struct SeedThermoRows {
    SeedLayerRows<R,N,Rows> base;const R (*thermo)[N+4][N+4];bool ready;
    UGKWP_GAS_HD SeedThermoRows(const Rows& values,const R (*cache)[N+4][N+4],bool valid)
        :base(values),thermo(cache),ready(valid){}
    UGKWP_GAS_HD SeedThermoRow<R,N> operator[](int node)const{return {base[node],thermo[node],ready};}
};
template<class R,int N> UGKWP_GAS_HD inline void storeSeedThermo(const LayerPoint<R,N>& p,R (*cache)[N+4]){
    cache[0][0]=p.T;cache[0][1]=p.theta;cache[0][2]=p.rho;cache[0][3]=p.cp;cache[0][4]=p.normalCoefficient;
    cache[1][0]=p.normalVelocity;cache[1][1]=p.mu;
    for(int j=0;j<N;++j){cache[2][j]=p.Y[j];cache[3][j]=p.h[j];}
}
template<class R,int N,class TableReal> UGKWP_GAS_HD inline bool loadBasePoint(
 const WallInput<R,N,TableReal>& in,const WallModelConfig<R>& c,const LayerContext<R,N>& ctx,
 const SeedThermoRow<R,N>& row,LayerPoint<R,N>& point,const bool terminalProfile=false){
    if(!row.ready)return loadBasePoint(in,c,ctx,row.base,point,terminalProfile);
    const auto* x=row.base.values;const auto* cache=row.thermo;
    point.u[0]=x[0];point.u[1]=x[1];point.k=x[N+2]*x[N+2];point.z=x[N+3];
    // Keep the changing-state validation even though the thermodynamics is fixed.
    if(!validatePointDynamics(in,point))return false;
    point.T=cache[0][0];point.theta=cache[0][1];point.rho=cache[0][2];point.cp=cache[0][3];point.normalCoefficient=cache[0][4];
    point.normalVelocity=cache[1][0];point.mu=cache[1][1];
    for(int j=0;j<N;++j){point.Y[j]=cache[2][j];point.h[j]=cache[3][j];}
    if(c.enableSst&&point.z>0)point.omega=R(1)/(point.z*point.z);
    return true;
}

template<class R,int N,class TableReal> UGKWP_GAS_HD UGKWP_WALL_DEVICE_LEAF bool pointReaction(const WallInput<R,N,TableReal>& in,LayerPoint<R,N>& p,const bool terminalProfile=false){
    if(in.model.mode!=GasMode::MixtureChemistry)return true;
    if(!in.model.thermo.species)return wallFatal(in,ugkp::diagnostics::Code::MissingOperator,10008);
    R concentration[N];for(int j=0;j<N;++j)concentration[j]=p.rho*p.Y[j]/in.model.thermo.species[j].molarMass;
    const auto& mech=in.model.mechanism;
    if(mech.reactionCount>0&&!mech.reactions)return wallFatal(in,ugkp::diagnostics::Code::MissingOperator,10009);
    for(int r=0;r<mech.reactionCount;++r){R progress=std::numeric_limits<R>::quiet_NaN();bool available=false;
        const bool valid=gasChemistryDetail::progressRate<R,TableReal,N,false,true>(r,p.T,concentration,in.model.thermo,mech,progress,&available);
        if(!available||!valid)return wallFatal(in,ugkp::diagnostics::Code::NonFinite,2415,progress);
        const auto& rx=mech.reactions[r];
        for(int j=0;j<rx.reactantCount;++j){const auto& v=mech.reactants[rx.reactantOffset+j];p.source[v.species]-=v.coefficient*progress*in.model.thermo.species[v.species].molarMass;}
        for(int j=0;j<rx.productCount;++j){const auto& v=mech.products[rx.productOffset+j];p.source[v.species]+=v.coefficient*progress*in.model.thermo.species[v.species].molarMass;}}
    for(int j=0;j<N;++j)if(!gasFinite(p.source[j]))return wallFatal(in,ugkp::diagnostics::Code::NonFinite,10010,p.source[j],0,j);
    (void)terminalProfile;
    return true;
}
template<class R,int N,class TableReal> UGKWP_GAS_HD inline void applySstGradients(const WallInput<R,N,TableReal>& in,
 R y,R du0,R du1,R dv,R dk,R domega,LayerPoint<R,N>& p){
    const auto& a=in.model.sstCoefficients;
    const R cd=sstCrossDiffusion(p.omega,dk*domega,a);
    // Form the complete strain before either viscosity or production limiting.
    // Squared strain and limited production cannot be split into additive
    // normal/tangential pieces: their cross terms can cancel or reinforce.
    const auto& t=in.sstTangentialGradient;
    const R div=dv+t[0][0]+t[1][1];
    const R s2=R(2)*(t[0][0]*t[0][0]+t[1][1]*t[1][1]+dv*dv)
        +(t[0][1]+t[1][0])*(t[0][1]+t[1][0])
        +(du0+t[2][0])*(du0+t[2][0])+(du1+t[2][1])*(du1+t[2][1]);
    const R g=s2-R(2)/R(3)*div*div;
    p.f1=sstF1(p.k,p.omega,p.mu/p.rho,y,cd,a);
    p.f2=sstF2(p.k,p.omega,p.mu/p.rho,y,a);
    p.nut=sstNut(p.k,p.omega,s2,p.f2,a);
    p.sourceK=sstKSource(p.rho,p.k,p.omega,div,p.nut,g,a);
    // The same nodal SST expression as sstOmegaSource, factored once so the
    // CV integral need not subtract a potentially dominant nodal CD source.
    // Preserve the complete nodal source for the existing Picard initializer.
    const R gamma=sstGamma(p.f1,a),beta=sstBeta(p.f1,a);
    const R production=gamma*sstMinimum(g,sstOmegaProductionLimit(p.omega,s2,p.f2,a));
    const R withoutCross=production-(R(2)/R(3))*gamma*div*p.omega-beta*p.omega*p.omega;
    p.sourceOmegaWithoutCrossDiffusion=p.rho*withoutCross;
    p.sourceOmega=p.rho*(withoutCross+(R(1)-p.f1)*cd);
}
template<class R,int N,class TableReal> UGKWP_GAS_HD UGKWP_WALL_DEVICE_LEAF void pointSst(const WallInput<R,N,TableReal>& in,const WallModelConfig<R>& c,
 R y,const LayerPoint<R,N>& left,const LayerPoint<R,N>& right,R length,LayerPoint<R,N>& p,R leftSpacing=R(0)){
    if(!c.enableSst||p.z<=0||y<=0)return;
    const R hr=length-leftSpacing;const bool centered=leftSpacing>0&&hr>0;
    const R du0=centered?(c.coherentDiscretization?logShearDerivative(left.u[0],p.u[0],right.u[0],leftSpacing,hr,y):centeredDerivative(left.u[0],p.u[0],right.u[0],leftSpacing,hr)):(right.u[0]-left.u[0])/length;
    const R du1=centered?(c.coherentDiscretization?logShearDerivative(left.u[1],p.u[1],right.u[1],leftSpacing,hr,y):centeredDerivative(left.u[1],p.u[1],right.u[1],leftSpacing,hr)):(right.u[1]-left.u[1])/length;
    const R leftNormal=(p.normalCoefficient+left.normalCoefficient)/R(2)*(p.theta-left.theta)+(p.T+left.T)/R(2)*(p.normalCoefficient-left.normalCoefficient);
    const R rightNormal=(right.normalCoefficient+p.normalCoefficient)/R(2)*(right.theta-p.theta)+(right.T+p.T)/R(2)*(right.normalCoefficient-p.normalCoefficient);
    const R dv=centered?hr*leftNormal/(leftSpacing*length)+leftSpacing*rightNormal/(hr*length):(leftNormal+rightNormal)/length;
    const R dk=centered?centeredDerivative(left.k,p.k,right.k,leftSpacing,hr):(right.k-left.k)/length;
    const R dz=centered?centeredDerivative(left.z,p.z,right.z,leftSpacing,hr):(right.z-left.z)/length;
    // Discretize the physical omega gradient. Applying a discrete chain rule
    // to z causes a grid-locked cross-diffusion spike at steep matching layers.
    // Only the singular smooth-wall interval uses its analytic z asymptotic.
    const R domega=left.z>0&&right.z>0?(centered?centeredDerivative(left.omega,p.omega,right.omega,leftSpacing,hr):(right.omega-left.omega)/length):-R(2)*dz/(p.z*p.z*p.z);
    applySstGradients(in,y,du0,du1,dv,dk,domega,p);
}
template<class R,int N> struct LayerFlux {
    R species[N]{},momentum[2]{},energy=0,k=0,omega=0;
    R thermal=0,stressWork=0,advectiveEnergy=0,conductivity=0,kAdvection=0;
};
// Restore formation-inclusive absolute-frame energy for diagnostic probes.
// This is a constant linear combination of the solved conservative rows.
template<class R,int N> UGKWP_GAS_HD inline R physicalLayerEnergyFlux(const LayerContext<R,N>& ctx,const LayerFlux<R,N>& f){
    R energy=f.energy+ctx.mass*ctx.energyReference;
    for(int d=0;d<2;++d)energy+=ctx.wallU[d]*(f.momentum[d]+ctx.mass*ctx.wallU[d]/R(2));
    return energy;
}
template<class R,int N,class TableReal> UGKWP_GAS_HD inline bool layerFlux(const WallInput<R,N,TableReal>& in,const WallModelConfig<R>& c,
 const LayerContext<R,N>& ctx,const LayerPoint<R,N>& a,const LayerPoint<R,N>& b,R length,LayerFlux<R,N>& f,R ya=0,R yb=0,const bool terminalProfile=false){
    if(length==0)return wallFatal(in,ugkp::diagnostics::Code::ZeroDenominator,10011,length);
    const R rho=(a.rho+b.rho)/R(2),nut=(a.nut+b.nut)/R(2),T=(a.T+b.T)/R(2);
    R Y[N],grad[N],diff[N],diffEnthalpy=0;
    for(int j=0;j<N;++j){Y[j]=(a.Y[j]+b.Y[j])/R(2);grad[j]=(b.Y[j]-a.Y[j])/length;}
    bool diffusionAvailable=false;
    const bool diffusionValid=correctedSpeciesDiffusion(Y,grad,rho,in.model.diffusivity,nut,c.turbulentSchmidt,T,in.model.thermo,diff,diffEnthalpy,true,&diffusionAvailable);
    if(!diffusionAvailable||!diffusionValid)return wallFatal(in,ugkp::diagnostics::Code::NonFinite,2416,diffEnthalpy);
    const R muA=a.mu+a.rho*a.nut,muB=b.mu+b.rho*b.nut;
    const R mu=transportConductance(muA,muB),temperatureJump=b.theta-a.theta;
    R enthalpyJump=0;
    for(int j=0;j<N;++j)enthalpyJump+=Y[j]*speciesEnthalpyIncrement(j,a.T,temperatureJump,in.model.thermo,ctx.enthalpyJump[j]);
    // Integrate Cp exactly in T; formation offsets never enter the difference.
    const R cpSecant=temperatureJump!=0?enthalpyJump/temperatureJump:(a.cp+b.cp)/R(2);
    const R molecularMobility=in.model.molecularPrandtl>0?a.mu/in.model.molecularPrandtl:
        (cpSecant!=0?in.model.conductivity/cpSecant:R(0));
    const R mobilityA=molecularMobility+a.rho*a.nut/c.turbulentPrandtl;
    const R mobilityB=(in.model.molecularPrandtl>0?b.mu/in.model.molecularPrandtl:molecularMobility)+b.rho*b.nut/c.turbulentPrandtl;
    const R mobility=transportConductance(mobilityA,mobilityB);
    f.conductivity=mobility*cpSecant;f.thermal=-mobility*enthalpyJump/length;
    const R thermalWeight=thermalWorkVelocityWeight(muA,muB,mobilityA,mobilityB);
    R kinetic=R(.5)*((a.normalVelocity+b.normalVelocity)/R(2))*((a.normalVelocity+b.normalVelocity)/R(2));
    for(int j=0;j<N;++j){f.species[j]=ctx.mass*Y[j]+diff[j];f.advectiveEnergy+=f.species[j]*((ctx.wallEnthalpy[j]-ctx.energyReference)+speciesEnthalpyIncrement(j,in.temperature,(a.theta+b.theta)/R(2),in.model.thermo,ctx.enthalpyJump[j]));}
    for(int d=0;d<2;++d){const R u=(a.u[d]+b.u[d])/R(2),tau=mu*(b.u[d]-a.u[d])/length;
        f.momentum[d]=ctx.mass*u-tau;
        f.stressWork-=tau*(a.u[d]+thermalWeight*(b.u[d]-a.u[d]));kinetic+=R(.5)*u*u;}
    f.advectiveEnergy+=ctx.mass*kinetic;f.energy=f.advectiveEnergy+f.stressWork+f.thermal;
    const R dk=R(.5)*(a.mu+a.rho*sstAlphaK(a.f1,in.model.sstCoefficients)*a.nut+
        b.mu+b.rho*sstAlphaK(b.f1,in.model.sstCoefficients)*b.nut);
    R faceK=(a.k+b.k)/R(2),gradientK=(b.k-a.k)/length;
    if(c.enableSst&&ya==0&&yb>0){
        const R slope=::sqrt(a.rho*in.model.sstCoefficients.beta1/(R(6)*a.mu));
        const R ell=a.z/slope;
        const R power=(R(1)+::sqrt(R(1)+R(24)*in.model.sstCoefficients.betaStar/in.model.sstCoefficients.beta1))/R(2);
        const R advection=ctx.mass/(R(2)*a.mu);
        faceK=b.k*nearWallKShape(yb/R(2),yb,ell,power,advection);
        gradientK=b.k*nearWallKDerivative(yb/R(2),yb,ell,power,advection);
    }
    f.kAdvection=ctx.mass*faceK;f.k=f.kAdvection-dk*gradientK;


    if(c.enableSst){const R dw=R(.5)*(a.mu+a.rho*sstAlphaOmega(a.f1,in.model.sstCoefficients)*a.nut+
        b.mu+b.rho*sstAlphaOmega(b.f1,in.model.sstCoefficients)*b.nut);
        f.omega=ctx.mass*(a.omega+b.omega)/R(2)
            -dw*omegaGradientFit(ya,yb)*(b.omega-a.omega)/length;}


    (void)terminalProfile;
    if(!gasFinite(f.energy)||!gasFinite(f.k))return wallFatal(in,ugkp::diagnostics::Code::NonFinite,2417,f.energy);
    return true;
}
template<class R,int N,class TableReal,class Rows> UGKWP_GAS_HD inline bool evaluatePoint(const WallInput<R,N,TableReal>& in,
 const WallModelConfig<R>& c,const LayerContext<R,N>& ctx,const Rows& x,const R* y,int index,LayerPoint<R,N>& p,const bool terminalProfile=false){
    if(!loadBasePoint(in,c,ctx,x[index],p,terminalProfile)||!pointReaction(in,p,terminalProfile))return false;
    if(c.enableSst&&index>0){const int l=index-1,r=index+1<c.nodes?index+1:index;LayerPoint<R,N> a,b;
        if(!loadBasePoint(in,c,ctx,x[l],a,terminalProfile)||!loadBasePoint(in,c,ctx,x[r],b,terminalProfile))return false;
        pointSst(in,c,y[index],a,b,y[r]-y[l],p,y[index]-y[l]);}
    return applyEquilibriumViscosity(in,c,ctx,y[index],p);
}
// Flux values at a face depend on endpoint gradients through SST eddy viscosity.
// Use a two-point strain/gradient consistently, preserving a three-node residual
// stencil and block-tridiagonal Newton system.
template<class R,int N,class TableReal,class RowA,class RowB> UGKWP_GAS_HD UGKWP_WALL_DEVICE_LEAF bool intervalFlux(const WallInput<R,N,TableReal>& in,const WallModelConfig<R>& c,
 const LayerContext<R,N>& ctx,const RowA& xa,const RowB& xb,R ya,R yb,LayerFlux<R,N>& f,const bool terminalProfile=false){
    LayerPoint<R,N> a,b;if(!loadBasePoint(in,c,ctx,xa,a,terminalProfile)||!loadBasePoint(in,c,ctx,xb,b,terminalProfile))return false;
    const auto originalA=a,originalB=b;
    pointSst(in,c,ya,originalA,originalB,yb-ya,a);pointSst(in,c,yb,originalA,originalB,yb-ya,b);
    if(!applyEquilibriumViscosity(in,c,ctx,ya,a)||!applyEquilibriumViscosity(in,c,ctx,yb,b))return false;
    return layerFlux(in,c,ctx,a,b,yb-ya,f,ya,yb,terminalProfile);
}
template<class R,int N> UGKWP_GAS_HD inline void storeIntervalFlux(const LayerFlux<R,N>& f,R (*cache)[N+4]){
    for(int j=0;j<N;++j)cache[0][j]=f.species[j];
    cache[1][0]=f.momentum[0];cache[1][1]=f.momentum[1];cache[1][2]=f.energy;cache[1][3]=f.k;cache[1][4]=f.omega;
}
template<class R,int N> UGKWP_GAS_HD inline void loadIntervalFlux(const R (*cache)[N+4],LayerFlux<R,N>& f){
    for(int j=0;j<N;++j)f.species[j]=cache[0][j];
    f.momentum[0]=cache[1][0];f.momentum[1]=cache[1][1];f.energy=cache[1][2];f.k=cache[1][3];f.omega=cache[1][4];
}
// Lazy per-perturbation interval reuse preserves the row evaluation order.
// Every attempted state invalidates its two incident intervals; other entries
// depend only on the immutable Newton iterate and remain valid across columns.
template<class R,int N,class TableReal,class Rows> UGKWP_GAS_HD inline bool loadRowInterval(
 const WallInput<R,N,TableReal>& in,const WallModelConfig<R>& c,const LayerContext<R,N>& ctx,
 const Rows& x,const R* y,int face,LayerFlux<R,N>& f){
    return intervalFlux(in,c,ctx,x[face],x[face+1],y[face],y[face+1],f);
}
template<class R,int N,class TableReal,class Rows> UGKWP_GAS_HD inline bool loadRowInterval(
 const WallInput<R,N,TableReal>& in,const WallModelConfig<R>& c,const LayerContext<R,N>& ctx,
 const CachedLayerRows<R,N,Rows>& x,const R* y,int face,LayerFlux<R,N>& f){
    const int slot=face-x.cache.first;R* saved=x.fluxes[slot];
    if(!x.fluxStatus[slot]){
        if(!intervalFlux(in,c,ctx,x[face],x[face+1],y[face],y[face+1],f))return false;
        for(int j=0;j<N;++j)saved[j]=f.species[j];
        saved[N]=f.momentum[0];saved[N+1]=f.momentum[1];saved[N+2]=f.energy;
        saved[N+3]=f.k;saved[N+4]=f.omega;x.fluxStatus[slot]=1;
    }else{
        for(int j=0;j<N;++j)f.species[j]=saved[j];
        f.momentum[0]=saved[N];f.momentum[1]=saved[N+1];f.energy=saved[N+2];
        f.k=saved[N+3];f.omega=saved[N+4];
    }
    return true;
}
template<class R,int N,class TableReal,class Rows> UGKWP_GAS_HD UGKWP_WALL_DEVICE_LEAF bool layerRow(const WallInput<R,N,TableReal>& originalInput,
 const WallModelConfig<R>& c,const LayerContext<R,N>& ctx,const Rows& x,const R* y,int i,R* row,const R (*fluxes)[N+4][N+4]=nullptr){
    auto in=originalInput;in.diagnosticNode=static_cast<std::uint32_t>(i);
    constexpr int K=N+2,Z=N+3;
    for(int j=0;j<N+4;++j)row[j]=0;
    if(i==c.nodes-1){row[0]=x[i][0]-ctx.edgeU[0];row[1]=x[i][1]-ctx.edgeU[1];row[2]=x[i][2]-(in.matching.temperature-in.temperature);
        for(int j=0;j<N-1;++j)row[3+j]=x[i][3+j]-in.matching.massFraction[ctx.independent[j]];
        row[K]=x[i][K]-(c.enableSst?::sqrt(in.matching.k):0);row[Z]=x[i][Z]-(c.enableSst?R(1)/::sqrt(in.matching.omega):R(1));
        return true;}
    LayerPoint<R,N> p;if(!evaluatePoint(in,c,ctx,x,y,i,p))return false;
    LayerFlux<R,N> right;if(fluxes)loadIntervalFlux(fluxes[i],right);
    else if(!loadRowInterval(in,c,ctx,x,y,i,right))return false;
    if(i==0){row[0]=x[0][0];row[1]=x[0][1];row[2]=x[0][2];
        for(int j=0;j<N-1;++j){const int s=ctx.independent[j];row[3+j]=(right.species[s]-in.massFlux[s]-p.source[s]*y[1]/R(2))/ctx.scale[3+j];}
        row[K]=x[0][K];R wallZ=0;
        if(c.enableSst&&ctx.mass>0){R tau2=0;for(int d=0;d<2;++d){const R tau=-right.momentum[d];tau2+=tau*tau;}
            const R uTau=::sqrt(::sqrt(tau2)/p.rho),v=ctx.mass/p.rho;
            if(!(uTau>0))return wallFatal(in,ugkp::diagnostics::Code::ZeroDenominator,10012,uTau);
            const R vp=v/uTau;
            wallZ=::sqrt((p.mu/p.rho)/(uTau*uTau)*vp*(R(1)+R(5)*vp)/R(25));}
        row[Z]=x[0][Z]-(c.enableSst?wallZ:R(1));
        return true;}
    LayerFlux<R,N> left;if(fluxes)loadIntervalFlux(fluxes[i-1],left);
    else if(!loadRowInterval(in,c,ctx,x,y,i-1,left))return false;
    const R dyL=y[i]-y[i-1],dyR=y[i+1]-y[i],volume=(dyL+dyR)/R(2);
    for(int d=0;d<2;++d)row[d]=(right.momentum[d]-left.momentum[d])/ctx.scale[d];
    row[2]=(right.energy-left.energy)/ctx.scale[2];
    for(int j=0;j<N-1;++j){const int s=ctx.independent[j];row[3+j]=(right.species[s]-left.species[s]-volume*p.source[s])/ctx.scale[3+j];}
    if(!c.enableSst){row[K]=x[i][K];row[Z]=x[i][Z]-R(1);return true;}
    row[K]=(right.k-left.k-volume*p.sourceK)/ctx.scale[K];
    if(i==1){LayerPoint<R,N> a;if(!loadBasePoint(in,c,ctx,x[i-1],a))return false;
        row[Z]=p.z-a.z-y[1]*::sqrt(a.rho*in.model.sstCoefficients.beta1/(R(6)*a.mu));}
    else {
      LayerPoint<R,N> a,b;
      if(!loadBasePoint(in,c,ctx,x[i-1],a)||!loadBasePoint(in,c,ctx,x[i+1],b))return false;
      const R average=omegaLocalSquareMean(p.omega,y[i],(y[i-1]+y[i])/R(2),(y[i]+y[i+1])/R(2));
      const R crossIntegral=omegaDirectedCrossIntegral(a,p,b,y[i-1],y[i],y[i+1],volume);
      const R source=omegaSourceWithCrossDiffusionIntegral(p,in.model.sstCoefficients,volume,average,crossIntegral);
      row[Z]=-(right.omega-left.omega-volume*source)*p.z*p.z*p.z*p.z/(R(2)*volume*ctx.scale[Z]);}
    for(int j=0;j<N+4;++j)if(!gasFinite(row[j]))return wallFatal(in,ugkp::diagnostics::Code::NonFinite,2425,row[j],0,j);
    return true;
}
// Static storage errors remain errors inside a mathematical trial. Only the
// property channels actually consumed by this wall at this temperature count.
template<class R,int N,class TableReal> UGKWP_GAS_HD inline bool layerTrialStorageAvailable(
 const WallInput<R,N,TableReal>& in,R temperature,int node=-1){
    if(!gasFinite(temperature)||temperature<=0)return true;
    const auto& view=in.model.thermo;
    const auto report=[&](ugkp::diagnostics::Code code,int species,double value){
        auto* diagnostics=in.diagnostics?in.diagnostics:view.diagnostics;
        ugkp::diagnostics::recordFatal(diagnostics,ugkp::diagnostics::makeRecord(code,2432,23,
            node>=0?static_cast<unsigned>(node):in.diagnosticNode,
            ugkp::diagnostics::EntityKind::Face,in.diagnosticEntity,species,value,0,0,
            ugkp::diagnostics::Quantity::Index));
        return false;
    };
    for(int species=0;species<N;++species){
        if(!gasThermoDetail::validMetadata(species,view))
            return report(ugkp::diagnostics::Code::MissingOperator,species,0);
        const auto& data=view.species[species];
        const int segment=data.model==SpeciesThermoModel::NASA7&&temperature>data.midTemperature?7:0;
        const int required=data.model==SpeciesThermoModel::LinearCp?3:6;
        const int available=view.coefficientCount-data.coefficientOffset-segment;
        if(available<required)return report(ugkp::diagnostics::Code::InvalidIndex,species,available);
        // H and Cp do not consume the seventh NASA coefficient. Gibbs-derived
        // reverse rates do; unused entropy storage must remain optional.
        const bool missingEntropy=data.model==SpeciesThermoModel::NASA7?available<7:!data.hasEntropyReference;
        if(missingEntropy&&in.model.mode==GasMode::MixtureChemistry){
            const auto& mechanism=in.model.mechanism;
            // The immutable mechanism was checked by the real base residual.
            for(int reaction=0;reaction<mechanism.reactionCount;++reaction){
                const auto& channel=mechanism.reactions[reaction];if(!channel.reversible)continue;
                for(int side=0;side<2;++side){const auto* terms=side?mechanism.products:mechanism.reactants;
                    const int offset=side?channel.productOffset:channel.reactantOffset,count=side?channel.productCount:channel.reactantCount;
                    for(int term=0;term<count;++term)if(terms[offset+term].species==species)
                        return report(data.model==SpeciesThermoModel::NASA7?ugkp::diagnostics::Code::InvalidIndex:ugkp::diagnostics::Code::MissingOperator,species,data.model==SpeciesThermoModel::NASA7?available:0);
                }
            }
        }
    }
    return true;
}
// Complete immutable tables need no per-node trial storage checks. An absent
// unused branch only disables this fast path; it never rejects the model.
template<class R,int N,class TableReal> UGKWP_GAS_HD inline bool layerAllTrialStorageComplete(
 const WallInput<R,N,TableReal>& in){
    bool reversible=false;
    if(in.model.mode==GasMode::MixtureChemistry){
        const auto& mechanism=in.model.mechanism;
        if(mechanism.reactionCount>0&&!mechanism.reactions)return false;
        for(int i=0;i<mechanism.reactionCount;++i)reversible=reversible||mechanism.reactions[i].reversible;
    }
    const auto& view=in.model.thermo;
    for(int species=0;species<N;++species){
        if(!gasThermoDetail::validMetadata(species,view))return false;
        const auto& data=view.species[species];
        const int required=data.model==SpeciesThermoModel::NASA7?14:3;
        if(view.coefficientCount-data.coefficientOffset<required)return false;
        if(reversible&&data.model==SpeciesThermoModel::LinearCp&&!data.hasEntropyReference)return false;
    }
    return true;
}
template<class R,int N,class TableReal,class Rows> UGKWP_GAS_HD inline bool layerTrialRowStorageAvailable(
 const WallInput<R,N,TableReal>& owner,const WallModelConfig<R>& c,const Rows& rows,int row){
    if(row==c.nodes-1)return true; // Matching residual reads prescribed values only.
    const int first=row?row-1:0,last=row+1;
    for(int node=first;node<=last;++node)
        if(!layerTrialStorageAvailable(owner,owner.temperature+rows[node][2],node))return false;
    return true;
}
template<class R,int N,int M,class TableReal,class Team=SerialWallTeam>
UGKWP_GAS_HD inline bool allResidual(const WallInput<R,N,TableReal>& in,
 const WallModelConfig<R>& c,const LayerContext<R,N>& ctx,const R (&x)[M][N+4],const R* y,
 R (&f)[M][N+4],R& norm,const Team& team=Team{},
 const WallInput<R,N,TableReal>* storageOwner=nullptr,bool* storageAvailable=nullptr,bool initialValid=true,R (*intervals)[N+4][N+4]=nullptr){
    if(storageAvailable)*storageAvailable=true;
    // Only silent, storage-certified Newton trials share interval values. Live
    // diagnostics retain the row's original ordering, context and multiplicity.
    const bool reuseFlux=intervals&&!in.diagnostics&&!in.model.thermo.diagnostics&&!storageOwner;
    if(reuseFlux){
        if(!team.all(initialValid))return false;
        bool valid=true;
        for(int i=team.rank();i<c.nodes-1;i+=team.size()){
            LayerFlux<R,N> flux;
            if(!intervalFlux(in,c,ctx,x[i],x[i+1],y[i],y[i+1],flux)){valid=false;break;}
            storeIntervalFlux(flux,intervals[i]);
        }
        if(!team.all(valid))return false;
    }
    const auto* cached=reuseFlux?intervals:nullptr;
    if(std::is_same<Team,SerialWallTeam>::value){
        if(!initialValid)return false;
        norm=0;for(int i=0;i<c.nodes;++i){
            if(storageOwner&&!layerTrialRowStorageAvailable(*storageOwner,c,x,i)){if(storageAvailable)*storageAvailable=false;return false;}
            if(!layerRow(in,c,ctx,x,y,i,f[i],cached))return false;
            for(int j=0;j<N+4;++j)norm=max(norm,abs(f[i][j]));}
        return true;
    }
    bool valid=initialValid,storageValid=true;
    for(int i=team.rank();valid&&i<c.nodes;i+=team.size()){
        if(storageOwner&&!layerTrialRowStorageAvailable(*storageOwner,c,x,i)){storageValid=false;valid=false;break;}
        if(!layerRow(in,c,ctx,x,y,i,f[i],cached)){valid=false;break;}
    }
    if(!team.all(valid)){
        // Reuse the failed-residual communication point. Only rejected trials
        // need this extra team reduction; no new device or normal-path barrier.
        if(storageAvailable&&storageOwner)*storageAvailable=team.all(storageValid);
        return false;
    }
    // Retain the original node/component order, including max/NaN semantics.
    if(team.leader()){norm=0;for(int i=0;i<c.nodes;++i)
        for(int j=0;j<N+4;++j)norm=max(norm,abs(f[i][j]));}
    norm=team.broadcast(norm);return true;
}
// Map a mathematical candidate to the mass-one composition simplex before
// evaluating it. Residual evaluation, acceptance and the next Jacobian then read
// the same state. Physical input diagnostics remain in wallFraction.
template<class R,int N> UGKWP_GAS_HD inline bool feasibleLayerComposition(R* state){
    if(N==1)return true;
    R largest=1,sum=0;bool feasible=true;
    for(int j=0;j<N-1;++j){
        const R v=state[3+j];if(!gasFinite(v))return false;
        largest=max(largest,abs(v));feasible=feasible&&v>=0&&v<=1;
    }
    if(feasible){for(int j=0;j<N-1;++j)sum+=state[3+j];if(sum<=1)return true;}
    if(N==2){state[3]=min(R(1),max(R(0),state[3]));return true;}
    // Scaling is only needed near the arithmetic range. Otherwise retain the
    // direct independent/dependent representation and its usual rounding.
    constexpr R maximum=std::numeric_limits<R>::max();
    const R scale=largest>maximum/R(4*N)?largest:R(1);
    R values[N],top=-maximum;sum=0;
    for(int j=0;j<N-1;++j){values[j]=state[3+j]/scale;sum+=values[j];top=max(top,values[j]);}
    values[N-1]=R(1)/scale-sum;top=max(top,values[N-1]);
    // In a unit simplex, a component more than one below the maximum cannot
    // be active. Centering and discarding it also avoid large-value overflow.
    R levels[N],ordered[N];
    for(int j=0;j<N;++j){
        const R gap=values[j]<top-R(1)/scale?R(-1):(values[j]-top)*scale;
        levels[j]=ordered[j]=gap;
    }
    for(int j=1;j<N;++j){const R v=ordered[j];int k=j;while(k>0&&ordered[k-1]<v){ordered[k]=ordered[k-1];--k;}ordered[k]=v;}
    R prefix=0,threshold=-1;
    for(int j=0;j<N;++j){prefix+=ordered[j];const R candidate=(prefix-R(1))/R(j+1);if(ordered[j]>candidate)threshold=candidate;}
    int largestIndependent=0;sum=0;
    for(int j=0;j<N-1;++j){state[3+j]=max(R(0),levels[j]-threshold);sum+=state[3+j];if(state[3+j]>state[3+largestIndependent])largestIndependent=j;}
    // The dependent species is represented by 1-sum(Y_independent). Ensure
    // that this very arithmetic remains nonnegative after projection rounding.
    while(sum>1){
        const R old=state[3+largestIndependent];R adjusted=max(R(0),old-(sum-R(1)));
        if(adjusted==old)adjusted=std::is_same<R,float>::value?R(::nextafterf(float(old),0.f)):R(::nextafter(double(old),0.));
        state[3+largestIndependent]=adjusted;sum=0;
        for(int j=0;j<N-1;++j)sum+=state[3+j];
    }
    return true;
}
// Newton globalization only: trial iterates do not advance physical time or
// modify the current profile. Discarded mathematical candidates must not publish
// physical diagnostics. An accepted feasible iterate keeps its certified residual.
template<class R,int N,int M,class TableReal,class Team=SerialWallTeam>
UGKWP_GAS_HD UGKWP_WALL_DEVICE_LEAF bool applyLayerNewtonUpdate(const WallInput<R,N,TableReal>& in,
 const WallModelConfig<R>& c,const LayerContext<R,N>& ctx,WallWorkspace<R,N,M>& w,
 R& norm,R& acceptedAlpha,const Team& team=Team{}){
    const R previous=norm;R alpha=1;acceptedAlpha=0;
    const bool checkStorage=!ctx.trialStorageComplete;
    auto candidateInput=in;candidateInput.diagnostics=nullptr;
    candidateInput.model.thermo.diagnostics=nullptr;
    // Retain the historical bounded bisection budget, without a case parameter.
    for(int line=0;line<40;++line){
        bool feasible=true;
        for(int i=team.rank();i<c.nodes;i+=team.size()){
            for(int j=0;j<N+4;++j)w.trial[i][j]=w.state[i][j]+alpha*w.delta[i][j];
            feasible=feasibleLayerComposition<R,N>(w.trial[i])&&feasible;
        }
        team.sync();R next=0;
        bool storageAvailable=true;
        const bool valid=allResidual(candidateInput,c,ctx,w.trial,w.y,w.residual,next,team,checkStorage?&in:nullptr,&storageAvailable,feasible,w.upper);
        if(!storageAvailable)return false;
        if(valid&&gasFinite(next)&&next<previous){
            for(int i=team.rank();i<c.nodes;i+=team.size())for(int j=0;j<N+4;++j)
                w.state[i][j]=w.trial[i][j];
            team.sync();acceptedAlpha=alpha;
            // The certified residual belongs to this exact projected state.
            // Its only warning producers are composition clipping and mass-sum
            // checks: the simplex projection leaves every nodal component in
            // [0,1], and endpoint averaging preserves unity within the existing
            // 64*N*epsilon diffusion tolerance. Thus an accepted trial has no
            // suppressed warning to replay. Property/storage errors cannot pass
            // the complete residual above. Initial/final diagnostics stay live.
            norm=next;return true;
        }
        alpha*=R(.5);
    }
    // Restore the current iterate after an exhausted line search. The caller
    // must still certify convergence before publishing any physical flux.
    return allResidual(in,c,ctx,w.state,w.y,w.residual,norm,team);
}
template<class R,int B> UGKWP_GAS_HD inline bool eliminateBlock(R (&a)[B][B],R (&upper)[B][B],R* rhs,int active=B){
    for(int col=0;col<active;++col){int pivot=col;for(int row=col+1;row<active;++row)if(abs(a[row][col])>abs(a[pivot][col]))pivot=row;
        if(!gasFinite(a[pivot][col])||abs(a[pivot][col])<std::numeric_limits<R>::min()*R(100))return false;
        if(pivot!=col){for(int k=0;k<active;++k){R v=a[col][k];a[col][k]=a[pivot][k];a[pivot][k]=v;v=upper[col][k];upper[col][k]=upper[pivot][k];upper[pivot][k]=v;}
            R v=rhs[col];rhs[col]=rhs[pivot];rhs[pivot]=v;}
        const R diagonal=a[col][col];for(int k=0;k<active;++k){a[col][k]/=diagonal;upper[col][k]/=diagonal;}rhs[col]/=diagonal;
        for(int row=0;row<active;++row)if(row!=col){const R factor=a[row][col];for(int k=0;k<active;++k){a[row][k]-=factor*a[col][k];upper[row][k]-=factor*upper[col][k];}rhs[row]-=factor*rhs[col];}}
    return true;
}
// A thread owns one perturbed node and its three Jacobian blocks. The view
// overlays only that node; neighboring evaluations read the immutable iterate.
template<class R,int N> struct PerturbedLayerState {
    const R (*state)[N+4];int node;R row[N+4];
    UGKWP_GAS_HD PerturbedLayerState(const R (*values)[N+4],int index):state(values),node(index){
        for(int col=0;col<N+4;++col)row[col]=values[index][col];}
    UGKWP_GAS_HD const R* operator[](int index)const{return index==node?row:state[index];}
};
template<class R,int N,int M,class TableReal> UGKWP_GAS_HD UGKWP_WALL_DEVICE_LEAF bool assembleJacobianNode(
 const WallInput<R,N,TableReal>& in,const WallModelConfig<R>& c,const LayerContext<R,N>& ctx,
 WallWorkspace<R,N,M>& w,int j){
    constexpr int B=N+4;PerturbedLayerState<R,N> perturbed(w.state,j);
    CachedLayerRows<R,N,PerturbedLayerState<R,N>> cached(perturbed,j);

        // A local simplex tangent basis remains feasible even when the globally
        // dependent species and an independent species are both exactly zero.
        R sum=0;for(int k=3;k<N+2;++k)sum+=perturbed.row[k];
        R donorMass=R(1)-sum;int donor=-1;
        for(int k=3;k<N+2;++k)if(perturbed.row[k]>donorMass){donorMass=perturbed.row[k];donor=k;}
        // EquilibriumOde's two unused SST rows are exact identity equations.
        // They do not affect transport; avoid two full thermochemistry/flux
        // probes per node while preserving the existing block solve/order.
        const bool algebraicSst=c.model==BoundaryLayerModel::EquilibriumOde&&!c.enableSst;
        if(algebraicSst){w.diagonal[j][N+2][N+2]=R(1);w.diagonal[j][N+3][N+3]=R(1);}
        const int columns=algebraicSst?N+2:B;
        for(int pass=0;pass<columns;++pass){const int col=donor>=3?(pass==3?donor:(pass==donor?3:pass)):pass;
            const bool species=col>=3&&col<N+2,transfer=species&&donor>=3&&col!=donor;
            const R old=perturbed.row[col],oldDonor=transfer?perturbed.row[donor]:R(0);
            const R rootEpsilon=::sqrt(std::numeric_limits<R>::epsilon());
            R variableScale=col==2?R(100):R(1);
            // Match the k residual scale so an exactly zero boundary k still
            // has a representable probe of its linear sqrt(k) constraint.
            if(c.enableSst&&col==N+2)variableScale=::sqrt(max(R(1),in.matching.k))*rootEpsilon;
            if(c.enableSst&&col==N+3)variableScale=rootEpsilon/::sqrt(in.matching.omega);
            R h=rootEpsilon*max(abs(old),variableScale);
            if(species){h=min(h,donorMass/R(4));if(col==donor)h=-h;}
            bool valid=false;R rows[3][B]{};const int start=j>0?j-1:0,end=j+1<c.nodes?j+1:c.nodes-1;
            auto derivativeInput=in;derivativeInput.diagnostics=nullptr;
            derivativeInput.model.thermo.diagnostics=nullptr;
            for(int attempt=0;attempt<2&&!valid;++attempt){
                perturbed.row[col]=old+h;
                if(transfer)perturbed.row[donor]=oldDonor-h;
                if(!ctx.trialStorageComplete
                    &&!layerTrialStorageAvailable(in,in.temperature+perturbed.row[2],j))return false;
                cached.invalidate(j,algebraicSst&&col<2);valid=true;
                for(int i=start;i<=end;++i)
                    if(!layerRow(derivativeInput,c,ctx,cached,w.y,i,rows[i-start])){valid=false;break;}
                if(!valid){
                    perturbed.row[col]=old;if(transfer)perturbed.row[donor]=oldDonor;
                    // The opposite direction keeps a valid interior base point;
                    // simplex transfers instead reduce their feasible step.
                    h=species?h/R(2):-h;
                }
            }
            const R actualStep=perturbed.row[col]-old;
            const R donorStep=transfer?oldDonor-perturbed.row[donor]:R(0);
            perturbed.row[col]=old;if(transfer)perturbed.row[donor]=oldDonor;
            if(!valid)return false;
            if(actualStep==0)return wallFatal(in,ugkp::diagnostics::Code::ZeroDenominator,2418,actualStep,0,col);
            // A nonzero sqrt(k) can have an entirely unresolvable relative
            // difference when neighboring fluxes dominate this column. Rebuild
            // only a zero-signal column with a symmetric physical-scale probe;
            // a true sqrt(k)=0 derivative remains exactly zero by symmetry.
            bool zeroSignal=c.enableSst&&col==N+2&&old!=R(0);
            // A coupled omega/energy row can change by roundoff while every
            // k equation remains unresolved. Only the k rows can certify this
            // probe; otherwise their zero column makes the Newton block singular.
            if(zeroSignal)for(int i=start;i<=end;++i)
                if(rows[i-start][N+2]!=w.residual[i][N+2])zeroSignal=false;
            R opposite[3][B]{};R denominator=actualStep;
            if(zeroSignal){
                // The existing k residual normalization uses max(1,k_match).
                // cbrt(epsilon) balances central O(h^2) truncation and the
                // O(epsilon/h) subtraction error, without a fitted threshold.
                h=::cbrt(std::numeric_limits<R>::epsilon())
                    *max(abs(old),::sqrt(max(R(1),in.matching.k)));
                perturbed.row[col]=old+h;const R plusStep=perturbed.row[col]-old;
                cached.invalidate(j);bool resolved=true;
                for(int i=start;i<=end;++i)if(!layerRow(derivativeInput,c,ctx,cached,w.y,i,rows[i-start])){resolved=false;break;}
                perturbed.row[col]=old-h;denominator=plusStep-(perturbed.row[col]-old);
                cached.invalidate(j);
                if(resolved)for(int i=start;i<=end;++i)if(!layerRow(derivativeInput,c,ctx,cached,w.y,i,opposite[i-start])){resolved=false;break;}
                perturbed.row[col]=old;
                if(!resolved)return false;
            }
            for(int i=start;i<=end;++i)for(int row=0;row<B;++row){const R forward=(rows[i-start][row]-(zeroSignal?opposite[i-start][row]:w.residual[i][row]))/denominator;
                R derivative=forward;
                // Direction e_col-e_donor: recover the original-coordinate
                // column using the donor column evaluated first above.
                R* block=j<i?w.lower[i][row]:(j==i?w.diagonal[i][row]:w.upper[i][row]);
                if(transfer)derivative+=(donorStep/actualStep)*block[donor];
                block[col]=derivative;}
        }
        return true;
}
template<class R,int N,int M> UGKWP_GAS_HD inline bool solveLayerNewtonSystem(
 const WallModelConfig<R>& c,WallWorkspace<R,N,M>& w){
    constexpr int B=N+4;
    // The omitted rows/columns form an uncoupled identity block. Keeping
    // their initialized zero increments solves the same full system.
    const int active=c.model==BoundaryLayerModel::EquilibriumOde&&!c.enableSst?N+2:B;
    for(int i=0;i<c.nodes;++i){if(i>0){for(int row=0;row<active;++row){R correction=0;for(int k=0;k<active;++k)correction+=w.lower[i][row][k]*w.delta[i-1][k];w.delta[i][row]-=correction;
          for(int col=0;col<active;++col){R sum=0;for(int k=0;k<active;++k)sum+=w.lower[i][row][k]*w.upper[i-1][k][col];w.diagonal[i][row][col]-=sum;}}}
        if(!eliminateBlock(w.diagonal[i],w.upper[i],w.delta[i],active))return false;}
    for(int i=c.nodes-2;i>=0;--i)for(int row=0;row<active;++row){R sum=0;for(int col=0;col<active;++col)sum+=w.upper[i][row][col]*w.delta[i+1][col];w.delta[i][row]-=sum;}
    // A zero-species active constraint can acquire a negative step solely from
    // block elimination roundoff. Remove only O(epsilon) outward motion; the
    // Newton globalization preserves the physical domain; finite Y is separately clipped.
    for(int i=0;i<c.nodes;++i){R scale=1;for(int j=3;j<N+2;++j)scale=max(scale,abs(w.delta[i][j]));
        const R roundoff=R(16*N)*std::numeric_limits<R>::epsilon()*scale;
        for(int j=3;j<N+2;++j)if(w.state[i][j]==0&&w.delta[i][j]<0&&-w.delta[i][j]<=roundoff)w.delta[i][j]=0;}

    return true;
}
template<class R,int N,int M,class TableReal,class Team=SerialWallTeam>
UGKWP_GAS_HD inline bool newtonStep(const WallInput<R,N,TableReal>& in,
 const WallModelConfig<R>& c,const LayerContext<R,N>& ctx,WallWorkspace<R,N,M>& w,const Team& team=Team{}){
    constexpr int B=N+4;
    for(int i=team.rank();i<c.nodes;i+=team.size())for(int row=0;row<B;++row){
        w.delta[i][row]=-w.residual[i][row];
        for(int col=0;col<B;++col)w.lower[i][row][col]=w.diagonal[i][row][col]=w.upper[i][row][col]=0;}
    team.sync();bool valid=true;
    for(int j=team.rank();j<c.nodes;j+=team.size())
        if(!assembleJacobianNode(in,c,ctx,w,j)){valid=false;break;}
    if(!team.all(valid))return false;
    // Preserve partial pivoting, block-Thomas elimination and back substitution.
    bool solved=false;if(team.leader())solved=solveLayerNewtonSystem(c,w);
    return team.broadcast(solved);
}
template<class R,int N,class TableReal> UGKWP_GAS_HD inline bool makeContext(const WallInput<R,N,TableReal>& in,const WallModelConfig<R>& c,LayerContext<R,N>& ctx){
    if(!wallTangentBasis(in.normal,ctx.tangent))return wallFatal(in,ugkp::diagnostics::Code::ZeroDenominator,10013);
    R relative[3];for(int a=0;a<3;++a)relative[a]=in.matching.velocity[a]-in.velocity[a];
    for(int d=0;d<2;++d){ctx.wallU[d]=dot(in.velocity,ctx.tangent[d]);ctx.edgeU[d]=dot(relative,ctx.tangent[d]);}
    ctx.wallNormal=dot(in.velocity,in.normal);
    for(int j=0;j<N;++j){ctx.mass+=in.massFlux[j];if(in.matching.massFraction[j]>in.matching.massFraction[ctx.dependent])ctx.dependent=j;}
    for(int j=0,k=0;j<N;++j)if(j!=ctx.dependent)ctx.independent[k++]=j;
    for(int j=0;j<N;++j){ctx.enthalpyJump[j]=speciesEnthalpyBranchJumpAs<R>(j,in.model.thermo);bool available=false;ctx.wallEnthalpy[j]=wallProperty<true>(j,in.temperature,in.model.thermo,&available);if(!available)return wallFatal(in,ugkp::diagnostics::Code::NonFinite,10014);ctx.energyReference+=in.matching.massFraction[j]*ctx.wallEnthalpy[j];}
    R gasR=0,cp=0,maxD=0;for(int j=0;j<N;++j){gasR+=in.matching.massFraction[j]*universalGasConstant<R>()/in.model.thermo.species[j].molarMass;bool available=false;const R value=wallProperty<false>(j,in.matching.temperature,in.model.thermo,&available);if(!available)return wallFatal(in,ugkp::diagnostics::Code::NonFinite,10015);cp+=in.matching.massFraction[j]*value;maxD=max(maxD,in.model.diffusivity[j]);}
    if(gasR==0)return wallFatal(in,ugkp::diagnostics::Code::ZeroDenominator,10016,gasR);
    const R rho=in.pressure/(gasR*in.matching.temperature),L=in.matchingDistance;
    // A residual scale measures activity, not signed transport. Suction
    // must not cancel diffusion and turn roundoff into a large residual.
    const R matchingMu=wallMolecularViscosity(in,in.matching.temperature,in.matching.massFraction);
    const R massMagnitude=abs(ctx.mass);
    for(int d=0;d<2;++d)ctx.scale[d]=max(R(1e-12),(massMagnitude+matchingMu/L)*max(R(1),abs(ctx.edgeU[d])));
    const R lambda=in.model.molecularPrandtl>0?matchingMu*cp/in.model.molecularPrandtl:in.model.conductivity;
    ctx.scale[2]=max(R(1e-12),(massMagnitude*cp+lambda/L)*max(R(1),max(in.temperature,in.matching.temperature)));
    for(int j=0;j<N-1;++j)ctx.scale[3+j]=max(R(1e-12),massMagnitude+rho*maxD/L);
    ctx.scale[N+2]=max(R(1e-12),(massMagnitude+matchingMu/L)*max(R(1),in.matching.k));ctx.scale[N+3]=rho;
    ctx.trialStorageComplete=layerAllTrialStorageComplete(in);
    (void)c;return true;
}
template<class R,int N,int M> UGKWP_GAS_HD inline bool seedScalarUsingScratch(
 const WallModelConfig<R>& c,WallWorkspace<R,N,M>& w,int variable,int diffusion,int source,int sink,
 R mass,R leftValue,R rightValue,int first,int scratchRow){
    // Each system owns a distinct upper-recurrence row.
    for(int i=first;i<c.nodes;++i){R low=0,diagonal=1,upper=0,rhs=0;
        if(i==first)rhs=leftValue;
        else if(i==c.nodes-1)rhs=rightValue;
        else {const R dl=w.diagonal[i-1][0][diffusion]/(w.y[i]-w.y[i-1]);
            const R dr=w.diagonal[i][0][diffusion]/(w.y[i+1]-w.y[i]);
            const R volume=(w.y[i+1]-w.y[i-1])/R(2);
            low=-dl-mass;upper=-dr;diagonal=dl+dr+mass+(sink>=0?w.seed[i][sink]*volume:R(0));rhs=source>=0?w.seed[i][source]*volume:R(0);}
        if(i>first){diagonal-=low*w.upper[i-1][scratchRow][0];rhs-=low*w.delta[i-1][variable];}
        if(!gasFinite(diagonal)||diagonal<=0)return false;
        w.upper[i][scratchRow][0]=upper/diagonal;w.delta[i][variable]=rhs/diagonal;}
    for(int i=c.nodes-2;i>=first;--i)w.delta[i][variable]-=w.upper[i][scratchRow][0]*w.delta[i+1][variable];
    return true;
}
template<class R,int N,int M> UGKWP_GAS_HD inline bool seedScalar(
 const WallModelConfig<R>& c,WallWorkspace<R,N,M>& w,int variable,int diffusion,int source,int sink,
 R mass,R leftValue,R rightValue,int first=0){
    if(!seedScalarUsingScratch(c,w,variable,diffusion,source,sink,mass,leftValue,rightValue,first,0))return false;
    return true;
}
// Both tangential velocities have the same transport operator. Preserve each
// scalar RHS recurrence and back substitution while factoring that operator once.
template<class R,int N,int M> UGKWP_GAS_HD inline bool seedVelocityPair(
 const WallModelConfig<R>& c,WallWorkspace<R,N,M>& w,R mass,R edgeU0,R edgeU1){
    for(int i=0;i<c.nodes;++i){R low=0,diagonal=1,upper=0,rhs0=0,rhs1=0;
        if(i==c.nodes-1){rhs0=edgeU0;rhs1=edgeU1;}
        else if(i>0){
            const R dl=w.diagonal[i-1][0][0]/(w.y[i]-w.y[i-1]);
            const R dr=w.diagonal[i][0][0]/(w.y[i+1]-w.y[i]);
            low=-dl-mass;upper=-dr;diagonal=dl+dr+mass+R(0);
        }
        if(i>0){diagonal-=low*w.upper[i-1][0][0];
            rhs0-=low*w.delta[i-1][0];rhs1-=low*w.delta[i-1][1];}
        if(!gasFinite(diagonal)||diagonal<=0)return false;
        w.upper[i][0][0]=upper/diagonal;
        w.delta[i][0]=rhs0/diagonal;w.delta[i][1]=rhs1/diagonal;
    }
    for(int d=0;d<2;++d)for(int i=c.nodes-2;i>=0;--i)
        w.delta[i][d]-=w.upper[i][0][0]*w.delta[i+1][d];
    return true;
}
template<class R,int N,int M,class Team> UGKWP_GAS_HD inline bool solveSeedSystemsWithTeam(
 const WallModelConfig<R>& c,WallWorkspace<R,N,M>& w,R mass,R edgeU0,R edgeU1,R edgeK,
 R firstOmega,R edgeOmega,const Team& team,std::true_type){
    // Preserve scalar short-circuit order and the original scratch row exactly.
    const bool solved=seedVelocityPair(c,w,mass,edgeU0,edgeU1)
        &&seedScalar(c,w,N+2,1,3,4,mass,R(0),edgeK)
        &&seedScalar(c,w,N+3,2,5,6,mass,firstOmega,edgeOmega,1);
    return team.broadcast(solved);
}
template<class R,int N,int M,class Team> UGKWP_GAS_HD inline bool solveSeedSystemsWithTeam(
 const WallModelConfig<R>& c,WallWorkspace<R,N,M>& w,R mass,R edgeU0,R edgeU1,R edgeK,
 R firstOmega,R edgeOmega,const Team& team,std::false_type){
    // Four independent tridiagonal systems, one recurrence row per variable.
    // This scratch is dead before Newton zeros/rebuilds all Jacobian blocks.
    bool solved=true;
    for(int job=team.rank();job<4;job+=team.size()){
        const int variable=job<2?job:N+job,diffusion=job<2?0:job-1;
        const int source=job<2?-1:2*job-1,sink=job<2?-1:2*job;
        const int first=job==3?1:0;
        const R leftValue=job==3?firstOmega:R(0);
        const R rightValue=job==0?edgeU0:(job==1?edgeU1:(job==2?edgeK:edgeOmega));
        // One common callsite keeps the participating lanes in the same SIMT
        // recurrence body rather than serializing four divergent call bodies.
        const bool ok=seedScalarUsingScratch(c,w,variable,diffusion,source,sink,
            mass,leftValue,rightValue,first,variable);
        if(!ok){solved=false;break;}
    }
    return team.all(solved);
}
template<class R,int N,int M,class Team> UGKWP_GAS_HD inline bool solveSeedSystemsWithTeam(
 const WallModelConfig<R>& c,WallWorkspace<R,N,M>& w,R mass,R edgeU0,R edgeU1,R edgeK,
 R firstOmega,R edgeOmega,const Team& team){
    return solveSeedSystemsWithTeam(c,w,mass,edgeU0,edgeU1,edgeK,firstOmega,edgeOmega,team,
        typename std::is_same<Team,SerialWallTeam>::type{});
}
// Positive transport/destruction Picard iteration is the Newton initializer.
// Failure is reported; no alternate original-profile solve is attempted.


template<class R,int N,int M,class TableReal,class Team=SerialWallTeam> UGKWP_GAS_HD UGKWP_WALL_DEVICE_LEAF bool seedSstLayer(const WallInput<R,N,TableReal>& in,
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
        const SeedThermoRows<R,N,decltype(w.state)> base(w.state,w.lower,reuseThermo);
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
        const SeedThermoRows<R,N,decltype(w.state)> faceBase(w.state,w.lower,reuseThermo);
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

template<class R,int N,int M,class TableReal> UGKWP_GAS_HD UGKWP_WALL_DEVICE_LEAF bool publishLayer(const WallInput<R,N,TableReal>& in,const WallModelConfig<R>& c,
 const LayerContext<R,N>& ctx,WallWorkspace<R,N,M>& w,int iterations,R norm,WallOutput<R,N>& output,const bool terminalProfile=false){
    WallOutput<R,N> out;LayerPoint<R,N> wall;if(!pointState(in,c,ctx,w.state[0],wall,terminalProfile))return false;
    if(!trace(in,wall.Y,out,terminalProfile))return false;
    LayerFlux<R,N> first,last;
    R wallEnergy=ctx.mass*wall.normalVelocity*wall.normalVelocity/R(2);
    for(int j=0;j<N;++j)wallEnergy+=in.massFlux[j]*(ctx.wallEnthalpy[j]-ctx.energyReference);
    R resistance=0,advectiveIntegral=0,viscousIntegral=0;bool heatAuditAvailable=true;
    for(int i=0;i+1<c.nodes;++i){LayerFlux<R,N> f;
        if(!intervalFlux(in,c,ctx,w.state[i],w.state[i+1],w.y[i],w.y[i+1],f,terminalProfile))return false;
        if(i==0)first=f;
        if(i+2==c.nodes)last=f;
        if(c.enableSst&&in.volumeSstClosure){
            const R za=w.state[i][N+3],zb=w.state[i+1][N+3];
            const R omegaA=za>0?R(1)/(za*za):R(0),omegaB=zb>0?R(1)/(zb*zb):R(0);
            out.kFluxHeight[i+1]=(w.y[i]+w.y[i+1])/R(2);
            out.kTransportFlux[i+1]=f.k;w.seed[i+1][3]=f.k;
            out.omegaDiffusiveFlux[i+1]=f.omega-ctx.mass*(omegaA+omegaB)/R(2);
        }
        if(!(f.conductivity>0)){heatAuditAvailable=false;continue;}
        const R r=(w.y[i+1]-w.y[i])/f.conductivity;
        if(!gasFinite(r)||r>std::numeric_limits<R>::max()-resistance){heatAuditAvailable=false;continue;}
        const R total=resistance+r,a=resistance/total,b=r/total;
        advectiveIntegral=a*advectiveIntegral+b*f.advectiveEnergy;
        viscousIntegral=a*viscousIntegral+b*f.stressWork;resistance=total;
    }
    for(int d=0;d<2;++d){const R tau=-first.momentum[d];for(int a=0;a<3;++a)out.traction[a]+=tau*ctx.tangent[d][a];}
    out.conductiveHeatFlux=first.energy-wallEnergy;
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
        const R slope=::sqrt(wall.rho*in.model.sstCoefficients.beta1/(R(6)*wall.mu));
        const R ell=w.state[0][N+3]/slope;
        const R power=(R(1)+::sqrt(R(1)+R(24)*in.model.sstCoefficients.betaStar/in.model.sstCoefficients.beta1))/R(2);
        const R advection=ctx.mass/(R(2)*wall.mu),firstK=w.state[1][N+2]*w.state[1][N+2];
        out.wallKFlux=-wall.mu*firstK*nearWallKDerivativeAtWall(w.y[1],ell,power,advection);
        const R matchingHalf=(w.y[c.nodes-1]-w.y[c.nodes-2])/R(2);
        const R matchingKFlux=last.k+(in.volumeSstClosure?matchingHalf*w.seed[c.nodes-1][0]:R(0));
        if(in.volumeSstClosure){out.kFluxCount=c.nodes+1;
            out.kFluxHeight[0]=0;out.kTransportFlux[0]=out.wallKFlux;w.seed[0][3]=out.wallKFlux;
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
            }else average=omegaLocalSquareMean(edge.omega,w.y[end],
                (w.y[end-1]+w.y[end])/R(2),w.y[end]);
            const R crossIntegral=end==1?omegaCrossDiffusionIntegral(previous,edge,w.y[end-1],w.y[end],
                (w.y[end-1]+w.y[end])/R(2),w.y[end]):omegaMatchingCrossIntegral(previous,edge,w.y[end-1],w.y[end],matchingHalf);
            const R omegaSource=omegaSourceWithCrossDiffusionIntegral(edge,in.model.sstCoefficients,
                matchingHalf,average,crossIntegral);
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
        // The macro omega row is a representative value at the cell centre.
        // A k-weighted destruction moment is finite even for a smooth wall,
        // but is not this point value and must not set macro eddy viscosity.
        // Publish its independent decay rate in the gas FV adapter instead.

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
template<int M,class R,int N,class TableReal,class Team>
UGKWP_GAS_HD inline bool prepareReactingWallInput(WallInput<R,N,TableReal>&in,
 const WallModelConfig<R>&c,WallStatus&status,const Team&team){
    bool fractionsValid=true;
    for(int j=0;j<N;++j)if(!wallFraction(in,in.matching.massFraction[j],in.matching.massFraction[j],j))fractionsValid=false;
    if(!fractionsValid)return false;
    bool inputValid=false;if(team.leader())inputValid=validateInput(in,c,status);
    if(!team.broadcast(inputValid))return false;
    if(c.nodes<2||c.nodes>M){if(team.leader()){fail(status,WallCode::InvalidInput);wallFatal(in,ugkp::diagnostics::Code::InvalidIndex,2423,c.nodes,ugkp::diagnostics::Quantity::Index);}return false;}
    if(in.model.mode!=GasMode::MixtureFrozen&&in.model.mode!=GasMode::MixtureChemistry&&in.model.mode!=GasMode::SingleLegacy){if(team.leader()){fail(status,WallCode::Unsupported);wallFatal(in,ugkp::diagnostics::Code::MissingOperator,10020);}return false;}
    if(c.enableSst&&(!gasFinite(in.matching.k)||!gasFinite(in.matching.omega)||in.matching.k<0||in.matching.omega<=0)){
        if(team.leader()){fail(status,WallCode::InvalidInput);wallFatal(in,(gasFinite(in.matching.k)&&gasFinite(in.matching.omega)?ugkp::diagnostics::Code::InvalidPhysicalState:ugkp::diagnostics::Code::NonFinite),2424,in.matching.omega);}return false;}
    const bool strictMechanism=in.model.mode!=GasMode::MixtureChemistry
        ||bool(validateGasMechanismView(in.model.thermo,in.model.mechanism));
    if(!strictMechanism){if(team.leader())wallFatal(in,ugkp::diagnostics::Code::MissingOperator,10021);return false;}
    return true;
}
} // detail

template<class R,int N,int M,class TableReal,class Team=detail::SerialWallTeam> UGKWP_GAS_HD inline bool evaluateReactingWallLayerImpl(
 const WallInput<R,N,TableReal>& originalInput,const WallModelConfig<R>& c,WallWorkspace<R,N,M>& w,WallOutput<R,N>& output,WallStatus& status,const Team& team=Team{}){
    using namespace detail;
    auto in=originalInput;in.model.thermo.diagnostics=in.diagnostics;
    if(!prepareReactingWallInput<M>(in,c,status,team))return false;
    LayerContext<R,N> ctx;if(!team.all(makeContext(in,c,ctx))){if(team.leader())fail(status,WallCode::InvalidThermodynamics);return false;}
    for(int i=team.rank();i<c.nodes;i+=team.size()){const R eta=R(i)/R(c.nodes-1),f=::pow(eta,c.stretch);w.y[i]=in.matchingDistance*f;
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
    bool seeded=false;
    if(seedIndependentSum<=R(1))seeded=seedSstLayer(in,c,ctx,w,team);
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
    int iteration=0;
    for(;norm>tolerance&&iteration<c.maxIterations;++iteration){
        if(!newtonStep(in,c,ctx,w,team)){
            if(team.leader())wallFatal(in,ugkp::diagnostics::Code::ZeroDenominator,2421);
            return false;
        }
        R alpha=0;
        if(!applyLayerNewtonUpdate(in,c,ctx,w,norm,alpha,team))return false;
        if(alpha==0)break;
    }
    if(team.leader()){status.iteration=iteration;status.residual=double(norm);}
    // A finite, admissible iterate is not a solved wall boundary condition.
    // In particular, its extrapolated matching flux can be arbitrarily large.
    // Keep output untouched on failure so it cannot become a ready closure.
    if(!gasFinite(norm)||norm>tolerance){
        if(team.leader()){
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
