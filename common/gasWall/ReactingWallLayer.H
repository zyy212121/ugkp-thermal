#pragma once
#include "WallLayerMath.H"
#include "WallThermoMath.H"
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
};
template<class R,int N> struct LayerPoint {
    R T=0,theta=0,normalCoefficient=0,Y[N]{},rho=0,cp=0,h[N]{},u[2]{},normalVelocity=0,k=0,z=0,omega=0;
    R nut=0,f1=0,f2=0,sourceK=0,sourceOmega=0,source[N]{};
};
template<class R,int N> UGKWP_GAS_HD UGKWP_WALL_DEVICE_LEAF bool pointState(const WallInput<R,N>& in,
 const WallModelConfig<R>& c,const LayerContext<R,N>& ctx,const R* state,LayerPoint<R,N>& p){
    p.theta=state[2];p.T=in.temperature+p.theta;p.u[0]=state[0];p.u[1]=state[1];p.k=state[N+2]*state[N+2];p.z=state[N+3];
    if(!gasFinite(p.T)||p.T<=0||!gasFinite(p.k)||p.k<0||!gasFinite(p.z)||p.z<0)return false;
    R sum=0,gasR=universalGasConstant<R>()/in.model.thermo.species[ctx.dependent].molarMass;for(int j=0;j<N-1;++j){const R y=state[3+j];if(!gasFinite(y)||y<0||y>1)return false;p.Y[ctx.independent[j]]=y;sum+=y;}
    p.Y[ctx.dependent]=R(1)-sum;if(p.Y[ctx.dependent]<0||p.Y[ctx.dependent]>1)return false;
    for(int j=0;j<N;++j){if(p.T<in.model.thermo.species[j].minTemperature||p.T>in.model.thermo.species[j].maxTemperature)return false;
        p.h[j]=speciesH(j,p.T,in.model.thermo);const R cp=speciesCp(j,p.T,in.model.thermo);
        if(!gasFinite(p.h[j])||!gasFinite(cp)||cp<=0)return false;
        p.cp+=p.Y[j]*cp;gasR+=p.Y[j]*(universalGasConstant<R>()/in.model.thermo.species[j].molarMass-universalGasConstant<R>()/in.model.thermo.species[ctx.dependent].molarMass);}
    p.rho=in.pressure/(gasR*p.T);p.normalCoefficient=ctx.mass*gasR/in.pressure;p.normalVelocity=ctx.wallNormal+p.normalCoefficient*p.T;
    if(!gasFinite(p.rho)||p.rho<=0||!gasFinite(p.normalVelocity))return false;
    if(c.enableSst&&p.z>0)p.omega=R(1)/(p.z*p.z);
    return true;
}
// Immutable mechanism was checked once at solve entry. Reuse the common rate
// arithmetic directly here; repeating an O(mechanism size) schema audit at
// every Newton perturbation is unnecessary and materially more expensive.
template<class R,int N> UGKWP_GAS_HD UGKWP_WALL_DEVICE_LEAF bool pointReaction(const WallInput<R,N>& in,LayerPoint<R,N>& p){
    if(in.model.mode!=GasMode::MixtureChemistry)return true;
    R concentration[N];for(int j=0;j<N;++j)concentration[j]=p.rho*p.Y[j]/in.model.thermo.species[j].molarMass;
    const auto& mech=in.model.mechanism;
    for(int r=0;r<mech.reactionCount;++r){R progress;
        if(!gasChemistryDetail::progressRate(r,p.T,concentration,in.model.thermo,mech,progress))return false;
        const auto& rx=mech.reactions[r];
        for(int j=0;j<rx.reactantCount;++j){const auto& v=mech.reactants[rx.reactantOffset+j];p.source[v.species]-=v.coefficient*progress*in.model.thermo.species[v.species].molarMass;}
        for(int j=0;j<rx.productCount;++j){const auto& v=mech.products[rx.productOffset+j];p.source[v.species]+=v.coefficient*progress*in.model.thermo.species[v.species].molarMass;}}
    for(int j=0;j<N;++j)if(!gasFinite(p.source[j]))return false;
    return true;
}
template<class R,int N> UGKWP_GAS_HD inline void applySstGradients(const WallInput<R,N>& in,
 R y,R du0,R du1,R dv,R dk,R domega,LayerPoint<R,N>& p){
    const auto& a=in.model.sstCoefficients;
    const R cd=sstCrossDiffusion(p.omega,dk*domega,a);
    const R s2=du0*du0+du1*du1+R(2)*dv*dv,g=s2-R(2)/R(3)*dv*dv;
    p.f1=sstF1(p.k,p.omega,in.model.viscosity/p.rho,y,cd,a);
    p.f2=sstF2(p.k,p.omega,in.model.viscosity/p.rho,y,a);
    p.nut=sstNut(p.k,p.omega,s2,p.f2,a);
    p.sourceK=sstKSource(p.rho,p.k,p.omega,dv,p.nut,g,a);
    p.sourceOmega=sstOmegaSource(p.rho,p.k,p.omega,dv,g,s2,p.f1,p.f2,cd,a);
}
template<class R,int N> UGKWP_GAS_HD UGKWP_WALL_DEVICE_LEAF void pointSst(const WallInput<R,N>& in,const WallModelConfig<R>& c,
 R y,const LayerPoint<R,N>& left,const LayerPoint<R,N>& right,R length,LayerPoint<R,N>& p,R leftSpacing=R(0)){
    if(!c.enableSst||p.z<=0||y<=0)return;
    const R hr=length-leftSpacing;const bool centered=leftSpacing>0&&hr>0;
    const R du0=centered?centeredDerivative(left.u[0],p.u[0],right.u[0],leftSpacing,hr):(right.u[0]-left.u[0])/length;
    const R du1=centered?centeredDerivative(left.u[1],p.u[1],right.u[1],leftSpacing,hr):(right.u[1]-left.u[1])/length;
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
template<class R,int N> struct LayerFlux {R species[N]{},momentum[2]{},energy=0,k=0,omega=0;};
// Restore formation-inclusive absolute-frame energy for diagnostic probes.
// This is a constant linear combination of the solved conservative rows.
template<class R,int N> UGKWP_GAS_HD inline R physicalLayerEnergyFlux(const LayerContext<R,N>& ctx,const LayerFlux<R,N>& f){
    R energy=f.energy+ctx.mass*ctx.energyReference;
    for(int d=0;d<2;++d)energy+=ctx.wallU[d]*(f.momentum[d]+ctx.mass*ctx.wallU[d]/R(2));
    return energy;
}
template<class R,int N> UGKWP_GAS_HD inline bool layerFlux(const WallInput<R,N>& in,const WallModelConfig<R>& c,
 const LayerContext<R,N>& ctx,const LayerPoint<R,N>& a,const LayerPoint<R,N>& b,R length,LayerFlux<R,N>& f,R ya=0,R yb=0){
    const R rho=(a.rho+b.rho)/R(2),nut=(a.nut+b.nut)/R(2),T=(a.T+b.T)/R(2);
    R Y[N],grad[N],diff[N],diffEnthalpy=0;
    for(int j=0;j<N;++j){Y[j]=(a.Y[j]+b.Y[j])/R(2);grad[j]=(b.Y[j]-a.Y[j])/length;}
    if(!correctedSpeciesDiffusion(Y,grad,rho,in.model.diffusivity,nut,c.turbulentSchmidt,T,in.model.thermo,diff,diffEnthalpy))return false;
    const R mu=in.model.viscosity+rho*nut,lambda=(in.model.molecularPrandtl>0?in.model.viscosity*(a.cp+b.cp)/(R(2)*in.model.molecularPrandtl):in.model.conductivity)+rho*nut*(a.cp+b.cp)/(R(2)*c.turbulentPrandtl);
    R kinetic=R(.5)*((a.normalVelocity+b.normalVelocity)/R(2))*((a.normalVelocity+b.normalVelocity)/R(2));
    for(int j=0;j<N;++j){f.species[j]=ctx.mass*Y[j]+diff[j];f.energy+=f.species[j]*((ctx.wallEnthalpy[j]-ctx.energyReference)+speciesEnthalpyIncrement(j,in.temperature,(a.theta+b.theta)/R(2),in.model.thermo,ctx.enthalpyJump[j]));}
    for(int d=0;d<2;++d){const R u=(a.u[d]+b.u[d])/R(2),tau=mu*(b.u[d]-a.u[d])/length;
        f.momentum[d]=ctx.mass*u-tau;f.energy-=tau*u;kinetic+=R(.5)*u*u;}
    f.energy+=ctx.mass*kinetic-lambda*(b.theta-a.theta)/length;
    const R dk=in.model.viscosity+rho*sstAlphaK((a.f1+b.f1)/R(2),in.model.sstCoefficients)*nut;
    f.k=ctx.mass*(a.k+b.k)/R(2)-dk*(b.k-a.k)/length;
    const R dw=in.model.viscosity+rho*sstAlphaOmega((a.f1+b.f1)/R(2),in.model.sstCoefficients)*nut;
    f.omega=ctx.mass*(a.omega+b.omega)/R(2)-dw*omegaGradientFit(ya,yb)*(b.omega-a.omega)/length;
    return gasFinite(f.energy)&&gasFinite(f.k);
}
template<class R,int N,int M> UGKWP_GAS_HD inline bool evaluatePoint(const WallInput<R,N>& in,
 const WallModelConfig<R>& c,const LayerContext<R,N>& ctx,const R (&x)[M][N+4],const R* y,int index,LayerPoint<R,N>& p){
    if(!pointState(in,c,ctx,x[index],p)||!pointReaction(in,p))return false;
    if(c.enableSst&&index>0){const int l=index-1,r=index+1<c.nodes?index+1:index;LayerPoint<R,N> a,b;
        if(!pointState(in,c,ctx,x[l],a)||!pointState(in,c,ctx,x[r],b))return false;
        pointSst(in,c,y[index],a,b,y[r]-y[l],p,y[index]-y[l]);}
    return true;
}
// Flux values at a face depend on endpoint gradients through SST eddy viscosity.
// Use a two-point strain/gradient consistently, preserving a three-node residual
// stencil and block-tridiagonal Newton system.
template<class R,int N> UGKWP_GAS_HD UGKWP_WALL_DEVICE_LEAF bool intervalFlux(const WallInput<R,N>& in,const WallModelConfig<R>& c,
 const LayerContext<R,N>& ctx,const R* xa,const R* xb,R ya,R yb,LayerFlux<R,N>& f){
    LayerPoint<R,N> a,b;if(!pointState(in,c,ctx,xa,a)||!pointState(in,c,ctx,xb,b))return false;
    const auto originalA=a,originalB=b;
    pointSst(in,c,ya,originalA,originalB,yb-ya,a);pointSst(in,c,yb,originalA,originalB,yb-ya,b);
    return layerFlux(in,c,ctx,a,b,yb-ya,f,ya,yb);
}
template<class R,int N,int M> UGKWP_GAS_HD UGKWP_WALL_DEVICE_LEAF bool layerRow(const WallInput<R,N>& in,
 const WallModelConfig<R>& c,const LayerContext<R,N>& ctx,const R (&x)[M][N+4],const R* y,int i,R* row){
    constexpr int K=N+2,Z=N+3;
    for(int j=0;j<N+4;++j)row[j]=0;
    if(i==c.nodes-1){row[0]=x[i][0]-ctx.edgeU[0];row[1]=x[i][1]-ctx.edgeU[1];row[2]=x[i][2]-(in.matching.temperature-in.temperature);
        for(int j=0;j<N-1;++j)row[3+j]=x[i][3+j]-in.matching.massFraction[ctx.independent[j]];
        row[K]=x[i][K]-(c.enableSst?::sqrt(in.matching.k):0);row[Z]=x[i][Z]-(c.enableSst?R(1)/::sqrt(in.matching.omega):R(1));
        return true;}
    LayerPoint<R,N> p;if(!evaluatePoint(in,c,ctx,x,y,i,p))return false;
    LayerFlux<R,N> right;if(!intervalFlux(in,c,ctx,x[i],x[i+1],y[i],y[i+1],right))return false;
    if(i==0){row[0]=x[0][0];row[1]=x[0][1];row[2]=x[0][2];
        for(int j=0;j<N-1;++j){const int s=ctx.independent[j];row[3+j]=(right.species[s]-in.massFlux[s]-p.source[s]*y[1]/R(2))/ctx.scale[3+j];}
        row[K]=x[0][K];R wallZ=0;
        if(c.enableSst&&ctx.mass>0){R tau2=0;for(int d=0;d<2;++d){const R tau=-right.momentum[d];tau2+=tau*tau;}
            const R uTau=::sqrt(::sqrt(tau2)/p.rho),v=ctx.mass/p.rho;
            if(!(uTau>0))return false;
            const R vp=v/uTau;
            wallZ=::sqrt((in.model.viscosity/p.rho)/(uTau*uTau)*vp*(R(1)+R(5)*vp)/R(25));}
        row[Z]=x[0][Z]-(c.enableSst?wallZ:R(1));
        return true;}
    LayerFlux<R,N> left;if(!intervalFlux(in,c,ctx,x[i-1],x[i],y[i-1],y[i],left))return false;
    const R dyL=y[i]-y[i-1],dyR=y[i+1]-y[i],volume=(dyL+dyR)/R(2);
    for(int d=0;d<2;++d)row[d]=(right.momentum[d]-left.momentum[d])/ctx.scale[d];
    row[2]=(right.energy-left.energy)/ctx.scale[2];
    for(int j=0;j<N-1;++j){const int s=ctx.independent[j];row[3+j]=(right.species[s]-left.species[s]-volume*p.source[s])/ctx.scale[3+j];}
    if(!c.enableSst){row[K]=x[i][K];row[Z]=x[i][Z]-R(1);return true;}
    row[K]=(right.k-left.k-volume*p.sourceK)/ctx.scale[K];
    LayerPoint<R,N> a,b;if(!pointState(in,c,ctx,x[i-1],a)||!pointState(in,c,ctx,x[i+1],b))return false;
    if(i==1){row[Z]=p.z-a.z-y[1]*::sqrt(a.rho*in.model.sstCoefficients.beta1/(R(6)*in.model.viscosity));}
    else {const R yl=(y[i-1]+y[i])/R(2),yr=(y[i]+y[i+1])/R(2);
      const R fit=omegaDestructionFit(y[i],yl,yr);
      const R destruction=p.rho*sstBeta(p.f1,in.model.sstCoefficients)*p.omega*p.omega;
      const R source=p.sourceOmega+destruction*(R(1)-fit);
      row[Z]=-(right.omega-left.omega-volume*source)*p.z*p.z*p.z*p.z/(R(2)*volume*ctx.scale[Z]);}
    for(int j=0;j<N+4;++j)if(!gasFinite(row[j]))return false;
    return true;
}
template<class R,int N,int M> UGKWP_GAS_HD inline bool allResidual(const WallInput<R,N>& in,
 const WallModelConfig<R>& c,const LayerContext<R,N>& ctx,const R (&x)[M][N+4],const R* y,R (&f)[M][N+4],R& norm){
    norm=0;for(int i=0;i<c.nodes;++i){if(!layerRow(in,c,ctx,x,y,i,f[i]))return false;
        for(int j=0;j<N+4;++j)norm=max(norm,abs(f[i][j]));}return true;
}
template<class R,int B> UGKWP_GAS_HD inline bool eliminateBlock(R (&a)[B][B],R (&upper)[B][B],R* rhs){
    for(int col=0;col<B;++col){int pivot=col;for(int row=col+1;row<B;++row)if(abs(a[row][col])>abs(a[pivot][col]))pivot=row;
        if(!gasFinite(a[pivot][col])||abs(a[pivot][col])<std::numeric_limits<R>::min()*R(100))return false;
        if(pivot!=col){for(int k=0;k<B;++k){R v=a[col][k];a[col][k]=a[pivot][k];a[pivot][k]=v;v=upper[col][k];upper[col][k]=upper[pivot][k];upper[pivot][k]=v;}
            R v=rhs[col];rhs[col]=rhs[pivot];rhs[pivot]=v;}
        const R diagonal=a[col][col];for(int k=0;k<B;++k){a[col][k]/=diagonal;upper[col][k]/=diagonal;}rhs[col]/=diagonal;
        for(int row=0;row<B;++row)if(row!=col){const R factor=a[row][col];for(int k=0;k<B;++k){a[row][k]-=factor*a[col][k];upper[row][k]-=factor*upper[col][k];}rhs[row]-=factor*rhs[col];}}
    return true;
}
template<class R,int N,int M> UGKWP_GAS_HD inline bool newtonStep(const WallInput<R,N>& in,
 const WallModelConfig<R>& c,const LayerContext<R,N>& ctx,WallWorkspace<R,N,M>& w){
    constexpr int B=N+4;
    for(int i=0;i<c.nodes;++i)for(int row=0;row<B;++row){w.delta[i][row]=-w.residual[i][row];for(int col=0;col<B;++col)w.lower[i][row][col]=w.diagonal[i][row][col]=w.upper[i][row][col]=0;}
    for(int j=0;j<c.nodes;++j){
        // A local simplex tangent basis remains feasible even when the globally
        // dependent species and an independent species are both exactly zero.
        R sum=0;for(int k=3;k<N+2;++k)sum+=w.state[j][k];
        R donorMass=R(1)-sum;int donor=-1;
        for(int k=3;k<N+2;++k)if(w.state[j][k]>donorMass){donorMass=w.state[j][k];donor=k;}
        for(int pass=0;pass<B;++pass){const int col=donor>=3?(pass==3?donor:(pass==donor?3:pass)):pass;
            const bool species=col>=3&&col<N+2,transfer=species&&donor>=3&&col!=donor;
            const R old=w.state[j][col],oldDonor=transfer?w.state[j][donor]:R(0);
            const R rootEpsilon=::sqrt(std::numeric_limits<R>::epsilon());
            R variableScale=col==2?R(100):R(1);
            if(c.enableSst&&col==N+2)variableScale=::sqrt(in.matching.k)*rootEpsilon;
            if(c.enableSst&&col==N+3)variableScale=rootEpsilon/::sqrt(in.matching.omega);
            R h=(sizeof(R)==4&&!species?R(::cbrt(std::numeric_limits<R>::epsilon())):rootEpsilon)*max(abs(old),variableScale);
            if(sizeof(R)==4&&!species&&col!=2){R variation=0;
                if(j>0)variation=max(variation,abs(old-w.state[j-1][col]));
                if(j+1<c.nodes)variation=max(variation,abs(w.state[j+1][col]-old));
                h=max(R(::cbrt(std::numeric_limits<R>::epsilon()))*variation,R(16)*std::numeric_limits<R>::epsilon()*max(abs(old),variableScale));}
            if(species){h=min(h,donorMass/R(4));if(col==donor)h=-h;}
            bool valid=false;R rows[3][B]{};const int start=j>0?j-1:0,end=j+1<c.nodes?j+1:c.nodes-1;
            for(int attempt=0;attempt<2&&!valid;++attempt){w.state[j][col]=old+h;
                if(transfer)w.state[j][donor]=oldDonor-h;
                valid=true;
                for(int i=start;i<=end;++i)if(!layerRow(in,c,ctx,w.state,w.y,i,rows[i-start])){valid=false;break;}
                if(!valid){w.state[j][col]=old;if(transfer)w.state[j][donor]=oldDonor;h=species?h/R(2):-h;}}
            const R actualStep=w.state[j][col]-old;
            const R donorStep=transfer?oldDonor-w.state[j][donor]:R(0);
            w.state[j][col]=old;if(transfer)w.state[j][donor]=oldDonor;
            if(!valid||actualStep==0)return false;
            R minus[3][B]{},minusStep=0;bool centered=sizeof(R)==4&&!species&&actualStep>0;
            if(centered){w.state[j][col]=old-h;minusStep=old-w.state[j][col];if(minusStep<=0)centered=false;for(int i=start;i<=end;++i)if(!layerRow(in,c,ctx,w.state,w.y,i,minus[i-start])){centered=false;break;}w.state[j][col]=old;}
            for(int i=start;i<=end;++i)for(int row=0;row<B;++row){const R forward=(rows[i-start][row]-w.residual[i][row])/actualStep;
                R derivative=centered?(minusStep*forward+actualStep*(w.residual[i][row]-minus[i-start][row])/minusStep)/(actualStep+minusStep):forward;
                // Direction e_col-e_donor: recover the original-coordinate
                // column using the donor column evaluated first above.
                R* block=j<i?w.lower[i][row]:(j==i?w.diagonal[i][row]:w.upper[i][row]);
                if(transfer)derivative+=(donorStep/actualStep)*block[donor];
                block[col]=derivative;}
        }
    }
    for(int i=0;i<c.nodes;++i){if(i>0){for(int row=0;row<B;++row){R correction=0;for(int k=0;k<B;++k)correction+=w.lower[i][row][k]*w.delta[i-1][k];w.delta[i][row]-=correction;
          for(int col=0;col<B;++col){R sum=0;for(int k=0;k<B;++k)sum+=w.lower[i][row][k]*w.upper[i-1][k][col];w.diagonal[i][row][col]-=sum;}}}
        if(!eliminateBlock(w.diagonal[i],w.upper[i],w.delta[i]))return false;}
    for(int i=c.nodes-2;i>=0;--i)for(int row=0;row<B;++row){R sum=0;for(int col=0;col<B;++col)sum+=w.upper[i][row][col]*w.delta[i+1][col];w.delta[i][row]-=sum;}
    // A zero-species active constraint can acquire a negative step solely from
    // block elimination roundoff. Remove only O(epsilon) outward motion; the
    // full physical residual still decides acceptance, never a clipped state.
    for(int i=0;i<c.nodes;++i){R scale=1;for(int j=3;j<N+2;++j)scale=max(scale,abs(w.delta[i][j]));
        const R roundoff=R(16*N)*std::numeric_limits<R>::epsilon()*scale;
        for(int j=3;j<N+2;++j)if(w.state[i][j]==0&&w.delta[i][j]<0&&-w.delta[i][j]<=roundoff)w.delta[i][j]=0;}

    return true;
}
template<class R,int N> UGKWP_GAS_HD inline void makeContext(const WallInput<R,N>& in,const WallModelConfig<R>& c,LayerContext<R,N>& ctx){
    int axis=0;if(abs(in.normal[1])<abs(in.normal[axis]))axis=1;if(abs(in.normal[2])<abs(in.normal[axis]))axis=2;
    R tangentNorm=0;for(int d=0;d<3;++d){ctx.tangent[0][d]=(d==axis?R(1):R(0))-in.normal[axis]*in.normal[d];tangentNorm+=ctx.tangent[0][d]*ctx.tangent[0][d];}
    for(int d=0;d<3;++d)ctx.tangent[0][d]/=::sqrt(tangentNorm);
    ctx.tangent[1][0]=in.normal[1]*ctx.tangent[0][2]-in.normal[2]*ctx.tangent[0][1];
    ctx.tangent[1][1]=in.normal[2]*ctx.tangent[0][0]-in.normal[0]*ctx.tangent[0][2];
    ctx.tangent[1][2]=in.normal[0]*ctx.tangent[0][1]-in.normal[1]*ctx.tangent[0][0];
    R relative[3];for(int a=0;a<3;++a)relative[a]=in.matching.velocity[a]-in.velocity[a];
    for(int d=0;d<2;++d){ctx.wallU[d]=dot(in.velocity,ctx.tangent[d]);ctx.edgeU[d]=dot(relative,ctx.tangent[d]);}
    ctx.wallNormal=dot(in.velocity,in.normal);
    for(int j=0;j<N;++j){ctx.mass+=in.massFlux[j];if(in.matching.massFraction[j]>in.matching.massFraction[ctx.dependent])ctx.dependent=j;}
    for(int j=0,k=0;j<N;++j)if(j!=ctx.dependent)ctx.independent[k++]=j;
    for(int j=0;j<N;++j){ctx.enthalpyJump[j]=speciesEnthalpyBranchJump(j,in.model.thermo);ctx.wallEnthalpy[j]=speciesH(j,in.temperature,in.model.thermo);ctx.energyReference+=in.matching.massFraction[j]*ctx.wallEnthalpy[j];}
    R gasR=0,cp=0,maxD=0;for(int j=0;j<N;++j){gasR+=in.matching.massFraction[j]*universalGasConstant<R>()/in.model.thermo.species[j].molarMass;cp+=in.matching.massFraction[j]*speciesCp(j,in.matching.temperature,in.model.thermo);maxD=max(maxD,in.model.diffusivity[j]);}
    const R rho=in.pressure/(gasR*in.matching.temperature),L=in.matchingDistance;
    for(int d=0;d<2;++d)ctx.scale[d]=max(R(1e-12),(ctx.mass+in.model.viscosity/L)*max(R(1),abs(ctx.edgeU[d])));
    const R lambda=in.model.molecularPrandtl>0?in.model.viscosity*cp/in.model.molecularPrandtl:in.model.conductivity;
    ctx.scale[2]=max(R(1e-12),(ctx.mass*cp+lambda/L)*max(R(1),max(in.temperature,in.matching.temperature)));
    for(int j=0;j<N-1;++j)ctx.scale[3+j]=max(R(1e-12),ctx.mass+rho*maxD/L);
    ctx.scale[N+2]=max(R(1e-12),(ctx.mass+in.model.viscosity/L)*max(R(1),in.matching.k));ctx.scale[N+3]=rho;
    (void)c;
}
template<class R,int N,int M> UGKWP_GAS_HD inline bool seedScalar(
 const WallModelConfig<R>& c,WallWorkspace<R,N,M>& w,int variable,int diffusion,int source,int sink,
 R mass,R leftValue,R rightValue,int first=0){
    for(int i=first;i<c.nodes;++i){R low=0,diagonal=1,upper=0,rhs=0;
        if(i==first)rhs=leftValue;
        else if(i==c.nodes-1)rhs=rightValue;
        else {const R dl=(w.seed[i-1][diffusion]+w.seed[i][diffusion])/(R(2)*(w.y[i]-w.y[i-1]));
            const R dr=(w.seed[i+1][diffusion]+w.seed[i][diffusion])/(R(2)*(w.y[i+1]-w.y[i]));
            const R volume=(w.y[i+1]-w.y[i-1])/R(2);
            low=-dl-mass;upper=-dr;diagonal=dl+dr+mass+(sink>=0?w.seed[i][sink]*volume:R(0));rhs=source>=0?w.seed[i][source]*volume:R(0);}
        if(i>first){diagonal-=low*w.upper[i-1][0][0];rhs-=low*w.delta[i-1][variable];}
        if(!gasFinite(diagonal)||diagonal<=0)return false;
        w.upper[i][0][0]=upper/diagonal;w.delta[i][variable]=rhs/diagonal;}
    for(int i=c.nodes-2;i>=first;--i)w.delta[i][variable]-=w.upper[i][0][0]*w.delta[i+1][variable];
    return true;
}
// Positive transport/destruction Picard iteration is only a Newton initializer.
// Its upwind convection and first-node asymptotic profile never authorize a
// result: the coupled conservative final residual must independently converge.
template<class R,int N,int M> UGKWP_GAS_HD inline bool seedSstLayer(const WallInput<R,N>& in,
 const WallModelConfig<R>& c,const LayerContext<R,N>& ctx,WallWorkspace<R,N,M>& w){
    if(!c.enableSst)return true;
    const int K=N+2,Z=N+3;const auto& coefficients=in.model.sstCoefficients;
    LayerPoint<R,N> wall;if(!pointState(in,c,ctx,w.state[0],wall))return false;
    const R slope=::sqrt(wall.rho*coefficients.beta1/(R(6)*in.model.viscosity));
    for(int i=1;i<c.nodes;++i){const R omega=R(1)/(slope*slope*w.y[i]*w.y[i])+in.matching.omega;
        w.state[i][Z]=R(1)/::sqrt(omega);}
    w.state[c.nodes-1][Z]=R(1)/::sqrt(in.matching.omega);
    for(int iteration=0;iteration<200;++iteration){
        for(int i=0;i<c.nodes;++i){LayerPoint<R,N> p;if(!evaluatePoint(in,c,ctx,w.state,w.y,i,p))return false;
            w.seed[i][0]=in.model.viscosity+p.rho*p.nut;
            w.seed[i][1]=in.model.viscosity+p.rho*sstAlphaK(p.f1,coefficients)*p.nut;
            w.seed[i][2]=in.model.viscosity+p.rho*sstAlphaOmega(p.f1,coefficients)*p.nut;
            const R kRest=p.sourceK+p.rho*coefficients.betaStar*p.k*p.omega;
            w.seed[i][3]=max(R(0),kRest);
            w.seed[i][4]=p.rho*coefficients.betaStar*p.omega+(p.k>0?max(R(0),-kRest)/p.k:R(0));
            const R omegaRest=p.sourceOmega+p.rho*sstBeta(p.f1,coefficients)*p.omega*p.omega;
            w.seed[i][5]=max(R(0),omegaRest);
            w.seed[i][6]=p.rho*sstBeta(p.f1,coefficients)*p.omega+(p.omega>0?max(R(0),-omegaRest)/p.omega:R(0));
            w.seed[i][7]=p.omega;}
        LayerFlux<R,N> first;if(!intervalFlux(in,c,ctx,w.state[0],w.state[1],w.y[0],w.y[1],first))return false;
        R wallZ=0;if(ctx.mass>0){R tau2=0;for(int d=0;d<2;++d){const R tau=-first.momentum[d];tau2+=tau*tau;}
            const R ut=::sqrt(::sqrt(tau2)/wall.rho);if(!(ut>0))return false;const R vp=ctx.mass/(wall.rho*ut);
            wallZ=::sqrt(in.model.viscosity/wall.rho/(ut*ut)*vp*(R(1)+R(5)*vp)/R(25));}
        const R firstZ=wallZ+w.y[1]*slope,firstOmega=R(1)/(firstZ*firstZ);
        if(!seedScalar(c,w,0,0,-1,-1,ctx.mass,R(0),ctx.edgeU[0])
            ||!seedScalar(c,w,1,0,-1,-1,ctx.mass,R(0),ctx.edgeU[1])
            ||!seedScalar(c,w,K,1,3,4,ctx.mass,R(0),in.matching.k)
            ||!seedScalar(c,w,Z,2,5,6,ctx.mass,firstOmega,in.matching.omega,1))return false;
        R change=0;
        for(int i=1;i<c.nodes-1;++i){for(int d=0;d<2;++d){change=max(change,abs(w.delta[i][d]-w.state[i][d])/max(R(1),abs(ctx.edgeU[d])));w.state[i][d]=(w.state[i][d]+w.delta[i][d])/R(2);}
            if(w.delta[i][K]<0||w.delta[i][Z]<=0)return false;
            const R oldK=w.state[i][K]*w.state[i][K],oldOmega=w.seed[i][7];
            change=max(change,abs(w.delta[i][K]-oldK)/max(R(1),in.matching.k));
            change=max(change,abs(w.delta[i][Z]-oldOmega)/oldOmega);
            w.state[i][K]=::sqrt((oldK+w.delta[i][K])/R(2));w.state[i][Z]=R(1)/::sqrt((oldOmega+w.delta[i][Z])/R(2));}
        w.state[0][Z]=wallZ;w.state[1][Z]=firstZ;
        if(change<R(1e-5))break;
    }
    return true;
}

template<class R,int N,int M> UGKWP_GAS_HD inline bool publishLayer(const WallInput<R,N>& in,const WallModelConfig<R>& c,
 const LayerContext<R,N>& ctx,WallWorkspace<R,N,M>& w,int iterations,R norm,WallOutput<R,N>& output){
    WallOutput<R,N> out;LayerPoint<R,N> wall;if(!pointState(in,c,ctx,w.state[0],wall))return false;
    if(!trace(in,wall.Y,out))return false;
    LayerFlux<R,N> first,last;if(!intervalFlux(in,c,ctx,w.state[0],w.state[1],w.y[0],w.y[1],first)
        ||!intervalFlux(in,c,ctx,w.state[c.nodes-2],w.state[c.nodes-1],w.y[c.nodes-2],w.y[c.nodes-1],last))return false;
    for(int d=0;d<2;++d){const R tau=-first.momentum[d];for(int a=0;a<3;++a)out.traction[a]+=tau*ctx.tangent[d][a];}
    R wallEnergy=ctx.mass*wall.normalVelocity*wall.normalVelocity/R(2);
    for(int j=0;j<N;++j)wallEnergy+=in.massFlux[j]*(ctx.wallEnthalpy[j]-ctx.energyReference);
    out.conductiveHeatFlux=first.energy-wallEnergy;
    // The matching endpoint also has a half control volume. Its source is
    // included consistently in both the diagnostic integral and edge flux.
    for(int i=0;i<c.nodes;++i){LayerPoint<R,N> p;if(!evaluatePoint(in,c,ctx,w.state,w.y,i,p))return false;
        w.seed[i][0]=p.sourceK; // Same full nodal source as the converged BVP.
        const R width=i==0?w.y[1]/R(2):(i==c.nodes-1?(w.y[i]-w.y[i-1])/R(2):(w.y[i+1]-w.y[i-1])/R(2));
        for(int j=0;j<N;++j)out.reactionIntegral[j]+=width*p.source[j];
        if(i==c.nodes-1)for(int j=0;j<N;++j)out.matchingSpeciesFlux[j]=last.species[j]+width*p.source[j];}
    for(int j=0;j<N;++j){out.speciesBalanceResidual[j]=out.matchingSpeciesFlux[j]-in.massFlux[j]-out.reactionIntegral[j];
        out.chemistryMismatch[j]=in.coarseReactionIntegral?out.reactionIntegral[j]-in.coarseReactionIntegral[j]:std::numeric_limits<R>::quiet_NaN();}
    out.chemistryMismatchAvailable=in.coarseReactionIntegral!=nullptr;
    out.wallKFlux=first.k;
    if(c.enableSst){const auto& q=in.quadrature;R total=0,moment=0;
        const R slope=::sqrt(wall.rho*in.model.sstCoefficients.beta1/(R(6)*in.model.viscosity));
        const R ell=w.state[0][N+3]/slope;
        const R power=(R(1)+::sqrt(R(1)+R(24)*in.model.sstCoefficients.betaStar/in.model.sstCoefficients.beta1))/R(2);
        const R advection=ctx.mass/(R(2)*in.model.viscosity),firstK=w.state[1][N+2]*w.state[1][N+2];
        out.wallKFlux=-in.model.viscosity*firstK*nearWallKDerivativeAtWall(w.y[1],ell,power,advection);
        if(q.count<=0||!q.distance||!q.volumeWeight||!gasFinite(q.volume)||q.volume<=0||!gasFinite(q.firstMoment))return false;
        for(int qi=0;qi<q.count;++qi){const R y=q.distance[qi],weight=q.volumeWeight[qi];
            if(!gasFinite(y)||!(y>0&&y<in.matchingDistance)||!gasFinite(weight)||weight<=0)return false;
            total+=weight;moment+=weight*y;int l=0;while(l+1<c.nodes-1&&w.y[l+1]<y)++l;
            const R fraction=(y-w.y[l])/(w.y[l+1]-w.y[l]);
            // Reconstruct the already-discretized full source with positive
            // nodal weights. Recomputing it with interval secant gradients
            // destroys consistency when local production/destruction cancel.
            if(l>0||fraction==R(1)){out.integratedKSource+=weight*((R(1)-fraction)*w.seed[l][0]+fraction*w.seed[l+1][0]);continue;}
            R state[N+4];for(int j=0;j<N+4;++j)state[j]=w.state[l][j]+fraction*(w.state[l+1][j]-w.state[l][j]);
            if(l==0)state[N+2]=::sqrt(firstK*nearWallKShape(y,w.y[1],ell,power,advection));
            LayerPoint<R,N> p,a,b;if(!pointState(in,c,ctx,state,p)||!pointState(in,c,ctx,w.state[l],a)||!pointState(in,c,ctx,w.state[l+1],b))return false;
            if(l==0){const R dy=w.y[1];
                const R dk=firstK*nearWallKDerivative(y,dy,ell,power,advection);
                applySstGradients(in,y,(b.u[0]-a.u[0])/dy,(b.u[1]-a.u[1])/dy,((b.normalCoefficient+a.normalCoefficient)/R(2)*(b.theta-a.theta)+(b.T+a.T)/R(2)*(b.normalCoefficient-a.normalCoefficient))/dy,dk,-R(2)*(b.z-a.z)/(dy*p.z*p.z*p.z),p);}
            else pointSst(in,c,y,a,b,w.y[l+1]-w.y[l],p);
            out.integratedKSource+=weight*p.sourceK;}
        const R tol=R(512)*std::numeric_limits<R>::epsilon();
        if(abs(total-q.volume)>tol*max(total,q.volume)||abs(moment-q.firstMoment)>tol*max(abs(moment),abs(q.firstMoment)))return false;
        int l=0;while(l+1<c.nodes-1&&w.y[l+1]<in.ownerDistance)++l;
        const R fraction=(in.ownerDistance-w.y[l])/(w.y[l+1]-w.y[l]);const R z=w.state[l][N+3]+fraction*(w.state[l+1][N+3]-w.state[l][N+3]);
        if(!(z>0))return false;
        out.ownerOmega=R(1)/(z*z);}
    out.iterations=iterations;out.residual=norm;output=out;return true;
}
} // detail

template<class R,int N,int M> UGKWP_GAS_HD inline bool evaluateReactingWallLayer(
 const WallInput<R,N>& in,const WallModelConfig<R>& c,WallWorkspace<R,N,M>& w,WallOutput<R,N>& output,WallStatus& status){
    using namespace detail;
    if(!validateInput(in,c,status))return false;
    if(c.nodes<4||c.nodes>M||c.maxIterations<1)return fail(status,WallCode::InvalidInput);
    // Common SST algebra uses the native gas precision. Reject a mixed-R SST
    // call explicitly instead of silently narrowing CPU-double calculations.
    if(c.enableSst&&!std::is_same<R,GpuReal>::value)return fail(status,WallCode::Unsupported);
    if(in.model.mode!=GasMode::MixtureFrozen&&in.model.mode!=GasMode::MixtureChemistry&&in.model.mode!=GasMode::SingleLegacy)return fail(status,WallCode::Unsupported);
    if(in.model.mode==GasMode::MixtureChemistry&&!validateGasMechanismView(in.model.thermo,in.model.mechanism))return fail(status,WallCode::ChemistryFailure);
    if(c.enableSst&&(!gasFinite(in.matching.k)||in.matching.k<=0||!gasFinite(in.matching.omega)||in.matching.omega<=0))return fail(status,WallCode::InvalidInput);
    LayerContext<R,N> ctx;makeContext(in,c,ctx);
    for(int i=0;i<c.nodes;++i){const R eta=R(i)/R(c.nodes-1),f=::pow(eta,c.stretch);w.y[i]=in.matchingDistance*f;
        w.state[i][0]=f*ctx.edgeU[0];w.state[i][1]=f*ctx.edgeU[1];
        w.state[i][2]=f*(in.matching.temperature-in.temperature);
        for(int j=0;j<N-1;++j)w.state[i][3+j]=in.matching.massFraction[ctx.independent[j]];
        w.state[i][N+2]=c.enableSst?::sqrt(in.matching.k)*f:R(0);w.state[i][N+3]=c.enableSst?f/::sqrt(in.matching.omega):R(1);}
    if(!seedSstLayer(in,c,ctx,w))return fail(status,WallCode::InadmissibleState);
    R norm=0;if(!allResidual(in,c,ctx,w.state,w.y,w.residual,norm))return fail(status,WallCode::InadmissibleState);
    const R tolerance=c.absoluteTolerance+c.relativeTolerance;
    for(int iteration=0;iteration<=c.maxIterations;++iteration){status.iteration=iteration;status.residual=double(norm);
        if(norm<=tolerance){if(!publishLayer(in,c,ctx,w,iteration,norm,output))return fail(status,WallCode::InvalidGeometry);return true;}
        if(iteration==c.maxIterations)break;
        if(!newtonStep(in,c,ctx,w))return fail(status,WallCode::SingularJacobian);
        R alpha=1;bool accepted=false;
        for(int line=0;line<40;++line){for(int i=0;i<c.nodes;++i)for(int j=0;j<N+4;++j)w.trial[i][j]=w.state[i][j]+alpha*w.delta[i][j];
            R next=0;if(allResidual(in,c,ctx,w.trial,w.y,w.residual,next)&&next<norm){for(int i=0;i<c.nodes;++i)for(int j=0;j<N+4;++j)w.state[i][j]=w.trial[i][j];norm=next;accepted=true;break;}
            alpha*=R(.5);}
        if(!accepted)return fail(status,WallCode::NonConvergence);
    }
    return fail(status,WallCode::NonConvergence);
}
}} // namespace

#undef UGKWP_WALL_DEVICE_LEAF
