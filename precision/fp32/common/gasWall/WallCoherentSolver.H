#pragma once
#include "WallCoherentNumerics.H"
#include "reconstruction/ConservativeProfile.H"
#include "reconstruction/GeneralEnergyIdentity.H"
namespace ugkwp { namespace gaswall { namespace coherent {
using namespace reconstruction;
struct BoundaryTransform {int column=0,lo=0,hi=0;GpuReal diagonal=0;GpuReal values[35]{};};
// Numerical cache belongs to a physical wall; it is not accepted macro history.
template<int M> struct Cache {
 bool haveState=false;
 CachedLU<M> factors;
 BoundedVector<GpuReal,M*6> scale;
 BoundedVector<int,2*M> branches;
 BoundedVector<BoundaryTransform,11> boundary;
 int dirichletMask=-1;
};
// Bounded worker scratch: never a separate dense matrix for every wall or lane.
template<int M> struct Scratch {
 WallWorkspace<GpuReal,2,M> probe;
 WallOutput<GpuReal,2> trialOut;
 GpuReal cachedUslopes[M][2],transformSlope[M][3],low[M][6];
 GpuReal publicationIntegral[M][3];
 detail::LayerPoint<GpuReal,2> points[M];
 detail::LayerFlux<GpuReal,2> oldFlux[M];
 physical::Face<GpuReal,2> faces[M];
 bool collectNutBranches=false,nutBranchPhase=false;
 BoundedVector<unsigned char,9*M> currentNutBranches,baseNutBranches;
 BoundedVector<GpuReal,M*6> base,res,rhs,scale,plus,steps;
 // Save only the active colored entries when trying an opposite FD batch.
 // This is worker-owned, not a per-lane vector or persistent wall cache.
 GpuReal originalSteps[M*6];
 BoundedVector<GpuReal,M*6*M*6> matrix;
 // Candidate positions are deterministic; compaction preserves scalar observer order.
 unsigned char nutSlotValue[9*M];bool nutSlotUsed[9*M];
 unsigned long long fdRoundoff=0,fdBranchHalves=0,fdFailures=0,factorFailures=0,centralDirections=0;
};
// Scalar arithmetic reference; the cooperative engine below reuses its physical helpers.
template<int M> struct ScalarEngine {
 using W=WallWorkspace<GpuReal,2,M>;using Ctx=detail::LayerContext<GpuReal,2>;
 Scratch<M>&s;
 GpuReal (&cachedUslopes)[M][2];
 bool&collectNutBranches;bool&nutBranchPhase;
 BoundedVector<unsigned char,9*M>&currentNutBranches;
 int observerSlot=-1; // Lane-local fixed position, or scalar append when negative.
 UGKWP_GAS_HD explicit ScalarEngine(Scratch<M>&scratch):s(scratch),cachedUslopes(s.cachedUslopes),collectNutBranches(s.collectNutBranches),nutBranchPhase(s.nutBranchPhase),currentNutBranches(s.currentNutBranches){}
UGKWP_GAS_HD bool trackedEvaluate(const Input&in,const Config&cfg,const physical::Primitive<GpuReal,2>&v,GpuReal y,const GpuReal*normal,GpuReal mesh,physical::Face<GpuReal,2>&f,physical::Source<GpuReal,2>&src){int slot=observerSlot<0?-1:observerSlot++;bool ok=physical::evaluatePrimitive(in,cfg,v,y,normal,mesh,f,src);if(collectNutBranches&&nutBranchPhase&&cfg.enableSst&&y>0){GpuReal s2=0;for(int a=0;a<3;++a){s2+=2*v.gradU[a][a]*v.gradU[a][a];for(int b=a+1;b<3;++b){GpuReal x=v.gradU[a][b]+v.gradU[b][a];s2+=x*x;}}bool branch=in.model.sstCoefficients.a1*v.omega>in.model.sstCoefficients.b1*src.f2*std::sqrt(s2);if(slot<0)currentNutBranches.push_back(branch);else{s.nutSlotValue[slot]=branch;s.nutSlotUsed[slot]=true;}}return ok;}
// Base-only assembly: same Hermite values and analytic derivatives, no post-fit operations.
UGKWP_GAS_HD physical::Primitive<GpuReal,2> basePhysical(Profile&p,GpuReal y){
 int l=0;while(l+1<p.cfg.nodes-1&&p.w.y[l+1]<y)++l;if(l==0)return p.physicalPrimitive(y);
 GpuReal h=p.w.y[l+1]-p.w.y[l],t=(y-p.w.y[l])/h,state[6],ds[6];
 for(int j=0;j<6;++j){state[j]=p.w.state[l][j]+t*(p.w.state[l+1][j]-p.w.state[l][j]);ds[j]=(p.w.state[l+1][j]-p.w.state[l][j])/h;if(p.hermiteField(j)){if(j<2){GpuReal a=p.w.state[l][j],b=p.w.state[l+1][j],ga=cachedUslopes[l][j],gb=cachedUslopes[l+1][j];state[j]=(2*t*t*t-3*t*t+1)*a+(-2*t*t*t+3*t*t)*b+(t*t*t-2*t*t+t)*h*ga+(t*t*t-t*t)*h*gb;ds[j]=6*t*(1-t)*(b-a)/h+(3*t*t-4*t+1)*ga+(3*t*t-2*t)*gb;}else{auto hv=p.hermite(l,j,t);state[j]=hv[0];ds[j]=hv[1];}}}
 detail::LayerPoint<GpuReal,2> q;detail::pointState(p.in,p.cfg,p.ctx,state,q);physical::Primitive<GpuReal,2>v;v.rho=q.rho;v.T=q.T;v.k=q.k;v.omega=q.omega;for(int j=0;j<2;++j)v.Y[j]=q.Y[j];
 GpuReal dy0=(p.ctx.independent[0]==0?1:-1)*ds[3],R=ugkwp::mixtureGasConstant(q.Y,p.in.model.thermo),dR=dy0*ugkwp::universalGasConstant<GpuReal>()*(1/p.in.model.thermo.species[0].molarMass-1/p.in.model.thermo.species[1].molarMass),drho=-q.rho*(ds[2]/q.T+dR/R),dv=-p.ctx.mass*drho/(q.rho*q.rho);
 for(int a=0;a<3;++a){v.U[a]=q.normalVelocity*p.in.normal[a];for(int d=0;d<2;++d)v.U[a]+=(p.ctx.wallU[d]+q.u[d])*p.ctx.tangent[d][a];v.gradT[a]=ds[2]*p.in.normal[a];v.gradK[a]=2*state[4]*ds[4]*p.in.normal[a];v.gradOmega[a]=(state[5]>0?-2*ds[5]/(state[5]*state[5]*state[5]):0)*p.in.normal[a];v.gradY[0][a]=dy0*p.in.normal[a];v.gradY[1][a]=-dy0*p.in.normal[a];GpuReal gu=dv*p.in.normal[a];for(int d=0;d<2;++d)gu+=ds[d]*p.ctx.tangent[d][a];for(int b=0;b<3;++b){v.gradU[a][b]=gu*p.in.normal[b];for(int lc=0;lc<3;++lc)for(int ld=0;ld<2;++ld)v.gradU[a][b]+=(lc==2?p.in.normal[a]:p.ctx.tangent[lc][a])*p.in.sstTangentialGradient[lc][ld]*p.ctx.tangent[ld][b];}}
 return v;
}
UGKWP_GAS_HD void copyActive(const Config&cfg,W&dest,const W&src){for(int i=0;i<cfg.nodes;++i){dest.y[i]=src.y[i];for(int j=0;j<6;++j){dest.state[i][j]=src.state[i][j];dest.residual[i][j]=src.residual[i][j];}}}
// Reduced energy is stored explicitly. Momentum Face remains absolute-frame;
// tangential reduced M subtracts m*wallU. Species rows retain old interval flux.
UGKWP_GAS_HD physical::Face<GpuReal,2> resistanceFace(Profile&p,int l,const GpuReal*speciesFlux,GpuReal*resistance=nullptr,GpuReal*stressIntegral=nullptr,bool*valid=nullptr,GpuReal*advectiveIntegral=nullptr){
 const auto&in=p.in;const auto&cfg=p.cfg;GpuReal ya=p.w.y[l],yb=p.w.y[l+1],y=(ya+yb)/2;
 auto v=basePhysical(p,y);physical::Face<GpuReal,2>f;physical::Source<GpuReal,2>src;bool pointOk=trackedEvaluate(in,cfg,v,y,in.normal,p.ctx.wallNormal,f,src);if(valid)*valid=pointOk&&*valid;
 GpuReal mu=wallMolecularViscosity(in,v.T,v.Y)+v.rho*src.nut,momentum[2]{},F[2]={speciesFlux[0],speciesFlux[1]};
 for(int d=0;d<2;++d){GpuReal cross=0;for(int a=0;a<3;++a)for(int b=0;b<3;++b)cross+=v.gradU[b][a]*in.normal[b]*p.ctx.tangent[d][a];for(int a=0;a<3;++a)f.momentum[a]+=mu*cross*p.ctx.tangent[d][a];momentum[d]=-p.ctx.mass*p.ctx.wallU[d];for(int a=0;a<3;++a)momentum[d]+=f.momentum[a]*p.ctx.tangent[d][a];}
 coherent_energy::Integral<GpuReal> integral;
 for(GpuReal g:{-.57735026918962576451f,.57735026918962576451f}){GpuReal yy=(ya+yb)/2+(yb-ya)/2*g;auto vv=basePhysical(p,yy);physical::Face<GpuReal,2>ff;physical::Source<GpuReal,2>ss;pointOk=trackedEvaluate(in,cfg,vv,yy,in.normal,p.ctx.wallNormal,ff,ss);if(valid)*valid=pointOk&&*valid;integral.add((yb-ya)/2,coherent_energy::terms(in,cfg,p.ctx,vv,ss.nut,momentum,F));}
 if(resistance)*resistance=integral.resistance;
 if(stressIntegral)*stressIntegral=integral.stress;
 if(advectiveIntegral)*advectiveIntegral=integral.advective;
 f.energy=integral.energy(p.w.state[l+1][2]-p.w.state[l][2]);for(int j=0;j<2;++j)f.species[j]=F[j];return f;
}

// Solve the SAME first-face constitutive fixed point, rather than truncating a
// Picard series. q and two tangential tractions are local auxiliary variables,
// not additional BVP nodes/unknowns. Always retain the last physical iterate.
UGKWP_GAS_HD bool closeFirstFace(Profile&p,const GpuReal*speciesFlux,GpuReal wallEnergy,WallOutput<GpuReal,2>&out,GpuReal&closure,GpuReal&scaledClosure){
 GpuReal x[3]={out.conductiveHeatFlux,0,0};for(int d=0;d<2;++d)for(int a=0;a<3;++a)x[d+1]+=out.traction[a]*p.ctx.tangent[d][a];
 const auto wallBasis=p.wallFluxBasis();
 auto evaluate=[&](const GpuReal*v,GpuReal*r){auto candidate=out;candidate.conductiveHeatFlux=v[0];for(int a=0;a<3;++a){candidate.traction[a]=0;for(int d=0;d<2;++d)candidate.traction[a]+=v[d+1]*p.ctx.tangent[d][a];}p.setWallFlux(candidate,wallBasis);bool valid=true;auto f=resistanceFace(p,0,speciesFlux,nullptr,nullptr,&valid);r[0]=f.energy-wallEnergy-v[0];for(int d=0;d<2;++d){GpuReal tau=p.ctx.mass*p.ctx.wallU[d];for(int a=0;a<3;++a)tau-=f.momentum[a]*p.ctx.tangent[d][a];r[d+1]=tau-v[d+1];}return valid;};
 auto magnitude=[](const GpuReal*r){return std::max(std::abs(r[0]),std::max(std::abs(r[1]),std::abs(r[2])));};
 GpuReal r[3];bool valid=evaluate(x,r);closure=magnitude(r);const GpuReal tolerance=p.cfg.absoluteTolerance+p.cfg.relativeTolerance;
 for(int iteration=0;valid&&closure>tolerance&&iteration<p.cfg.maxIterations;++iteration){GpuReal jac[3][3],step[3]={-r[0],-r[1],-r[2]};bool derivativeValid=true;
 for(int j=0;j<3;++j){GpuReal v[3]={x[0],x[1],x[2]},rr[3],h=std::sqrt(std::numeric_limits<GpuReal>::epsilon())*std::max(1.f,std::abs(x[j]));v[j]+=h;bool evaluated=evaluate(v,rr);if(!evaluated){h=-h;v[j]=x[j]+h;evaluated=evaluate(v,rr);}derivativeValid=derivativeValid&&evaluated;for(int i=0;i<3;++i)jac[i][j]=(rr[i]-r[i])/(v[j]-x[j]);}
 bool invertible=derivativeValid;for(int k=0;k<3&&invertible;++k){int pivot=k;for(int i=k+1;i<3;++i)if(std::abs(jac[i][k])>std::abs(jac[pivot][k]))pivot=i;if(jac[pivot][k]==0||!std::isfinite(jac[pivot][k])){invertible=false;break;}if(pivot!=k){for(int j=k;j<3;++j)swapValue(jac[pivot][j],jac[k][j]);swapValue(step[pivot],step[k]);}for(int i=k+1;i<3;++i){GpuReal t=jac[i][k]/jac[k][k];for(int j=k+1;j<3;++j)jac[i][j]-=t*jac[k][j];step[i]-=t*step[k];}}
 if(invertible)for(int i=2;i>=0;--i){for(int j=i+1;j<3;++j)step[i]-=jac[i][j]*step[j];step[i]/=jac[i][i];}
 bool accepted=false;for(int direction=0;direction<2&&!accepted;++direction){if(direction==0&&!invertible)continue;if(direction==1)for(int i=0;i<3;++i)step[i]=r[i];GpuReal alpha=1;for(int line=0;line<p.cfg.numerics.maximumLineSearchIterations;++line){GpuReal v[3],rr[3];for(int i=0;i<3;++i)v[i]=x[i]+alpha*step[i];bool evaluated=evaluate(v,rr);GpuReal next=magnitude(rr);if(evaluated&&std::isfinite(next)&&next<closure){for(int i=0;i<3;++i){x[i]=v[i];r[i]=rr[i];}closure=next;accepted=true;break;}alpha*=p.cfg.numerics.lineSearchReduction;}}
 if(!accepted)break;
 }
 // No new input rejection: expose remaining local residual to the caller's
 // actual equation norm instead of falsely claiming a closed first interval.
 out.conductiveHeatFlux=x[0];for(int a=0;a<3;++a){out.traction[a]=0;for(int d=0;d<2;++d)out.traction[a]+=x[d+1]*p.ctx.tangent[d][a];}p.setWallFlux(out,wallBasis);scaledClosure=std::max(std::abs(r[0])/p.ctx.scale[2],std::max(std::abs(r[1])/p.ctx.scale[0],std::abs(r[2])/p.ctx.scale[1]));return valid;
}

UGKWP_GAS_HD void trackOldInterval(const Input&in,const Config&cfg,const detail::LayerPoint<GpuReal,2>&left,const detail::LayerPoint<GpuReal,2>&right,const detail::LayerPoint<GpuReal,2>&p,GpuReal length,GpuReal y){
 int slot=observerSlot<0?-1:observerSlot++;if(!collectNutBranches||!cfg.enableSst||p.z<=0||y<=0)return;
 GpuReal du0=(right.u[0]-left.u[0])/length,du1=(right.u[1]-left.u[1])/length;
 GpuReal ln=(p.normalCoefficient+left.normalCoefficient)/2*(p.theta-left.theta)+(p.T+left.T)/2*(p.normalCoefficient-left.normalCoefficient);
 GpuReal rn=(right.normalCoefficient+p.normalCoefficient)/2*(right.theta-p.theta)+(right.T+p.T)/2*(right.normalCoefficient-p.normalCoefficient);
 GpuReal dv=(ln+rn)/length;const auto&t=in.sstTangentialGradient;
 GpuReal s2=2*(t[0][0]*t[0][0]+t[1][1]*t[1][1]+dv*dv)+(t[0][1]+t[1][0])*(t[0][1]+t[1][0])+(du0+t[2][0])*(du0+t[2][0])+(du1+t[2][1])*(du1+t[2][1]);
 bool branch=in.model.sstCoefficients.a1*p.omega>in.model.sstCoefficients.b1*p.f2*std::sqrt(s2);if(slot<0)currentNutBranches.push_back(branch);else{s.nutSlotValue[slot]=branch;s.nutSlotUsed[slot]=true;}
}

UGKWP_GAS_HD bool refresh(const Input&in,const Config&cfg,W&w,const Ctx&ctx,GpuReal&norm,WallOutput<GpuReal,2>&out){currentNutBranches.clear();nutBranchPhase=false;auto& low=s.low;auto&points=s.points;auto&oldFlux=s.oldFlux;
 for(int i=0;i<cfg.nodes;++i){points[i]={};oldFlux[i]={};if(!detail::pointState(in,cfg,ctx,w.state[i],points[i]))return false;for(int j=0;j<6;++j)low[i][j]=0;}
 for(int i=0;i<cfg.nodes-1;++i){auto a=points[i],b=points[i+1];detail::pointSst(in,cfg,w.y[i],points[i],points[i+1],w.y[i+1]-w.y[i],a);detail::pointSst(in,cfg,w.y[i+1],points[i],points[i+1],w.y[i+1]-w.y[i],b);trackOldInterval(in,cfg,points[i],points[i+1],a,w.y[i+1]-w.y[i],w.y[i]);trackOldInterval(in,cfg,points[i],points[i+1],b,w.y[i+1]-w.y[i],w.y[i+1]);if(!detail::layerFlux(in,cfg,ctx,a,b,w.y[i+1]-w.y[i],oldFlux[i],w.y[i],w.y[i+1]))return false;}
 for(int i=0;i<cfg.nodes-1;++i){auto p=points[i];if(!detail::pointReaction(in,p))return false;GpuReal volume=i==0?w.y[1]/2:(w.y[i+1]-w.y[i-1])/2;int sp=ctx.independent[0];low[i][3]=(oldFlux[i].species[sp]-(i==0?in.massFlux[sp]:oldFlux[i-1].species[sp])-volume*p.source[sp])/ctx.scale[3];}
 for(int j:{0,1,2,4,5})low[0][j]=w.state[0][j];
 if(!cfg.enableSst)low[0][5]-=1;
 int edge=cfg.nodes-1;low[edge][0]=w.state[edge][0]-ctx.edgeU[0];low[edge][1]=w.state[edge][1]-ctx.edgeU[1];low[edge][2]=w.state[edge][2]-(in.matching.temperature-in.temperature);low[edge][3]=w.state[edge][3]-in.matching.massFraction[ctx.independent[0]];low[edge][4]=w.state[edge][4]-(cfg.enableSst?std::sqrt(in.matching.k):0);low[edge][5]=w.state[edge][5]-(cfg.enableSst?1/std::sqrt(in.matching.omega):1);
 if(cfg.nodes>2&&cfg.enableSst)low[1][5]=points[1].z-points[0].z-w.y[1]*std::sqrt(points[0].rho*in.model.sstCoefficients.beta1/(6*points[0].mu));
 GpuReal wallEnergy=coherent_energy::wallTransport(in,ctx,points[0]);out={};out.conductiveHeatFlux=oldFlux[0].energy-wallEnergy;for(int d=0;d<2;++d)for(int a=0;a<3;++a)out.traction[a]-=oldFlux[0].momentum[d]*ctx.tangent[d][a];
 Profile p(in,cfg,w,s.transformSlope);for(int i=1;i<cfg.nodes;++i)for(int j=0;j<2;++j)cachedUslopes[i][j]=p.nodeSlope(i,j);p.setWallFlux(out);Segment seg;seg.lo=0;seg.hi=w.y[cfg.nodes-1];p.segments.push_back(seg);auto& faces=s.faces;
 // Zero-blowing/constant-composition diagnostic only. Derived first flux and wall gradient share a fixed point.
 GpuReal closure=0,scaledClosure=0;bool physicalValid=closeFirstFace(p,oldFlux[0].species,wallEnergy,out,closure,scaledClosure);
 nutBranchPhase=true;for(int i=0;i<cfg.nodes-1;++i)faces[i]=resistanceFace(p,i,oldFlux[i].species,nullptr,nullptr,&physicalValid);

 if(cfg.enableSst&&ctx.mass>0){GpuReal tau2=0;for(int d=0;d<2;++d){GpuReal tau=ctx.mass*ctx.wallU[d];for(int a=0;a<3;++a)tau-=faces[0].momentum[a]*ctx.tangent[d][a];tau2+=tau*tau;}GpuReal uTau=std::sqrt(std::sqrt(tau2)/points[0].rho),v=ctx.mass/points[0].rho;if(!(uTau>0))return detail::wallFatal(in,ugkp::diagnostics::Code::ZeroDenominator,10012,uTau);GpuReal vp=v/uTau;low[0][5]=w.state[0][5]-std::sqrt((points[0].mu/points[0].rho)/(uTau*uTau)*vp*(1+5*vp)/25);}
 if(in.adiabatic)low[0][2]=out.conductiveHeatFlux/ctx.scale[2];
 norm=scaledClosure;for(int i=0;i<cfg.nodes;++i){GpuReal high[6];for(int j=0;j<6;++j)high[j]=low[i][j];if(i>0&&i<cfg.nodes-1){GpuReal integralK=0,integralO=0;for(int side=0;side<2;++side){int l=i-1+side;GpuReal lo=side?w.y[i]:(w.y[i-1]+w.y[i])/2,hi=side?(w.y[i]+w.y[i+1])/2:w.y[i];const GpuReal g[2]={-.57735026918962576451f,.57735026918962576451f},wt[2]={1,1};for(int q=0;q<2;++q){GpuReal y=(hi+lo)/2+(hi-lo)/2*g[q];auto v=basePhysical(p,y);physical::Face<GpuReal,2>f;physical::Source<GpuReal,2>s;if(!trackedEvaluate(in,cfg,v,y,in.normal,ctx.wallNormal,f,s))return false;integralK+=(hi-lo)/2*wt[q]*s.k;integralO+=(hi-lo)/2*wt[q]*s.omega;}}
 auto&a=faces[i-1];auto&b=faces[i];GpuReal mom[2]{};for(int d=0;d<2;++d){for(int k=0;k<3;++k)mom[d]+=(b.momentum[k]-a.momentum[k])*ctx.tangent[d][k];high[d]=mom[d]/ctx.scale[d];}GpuReal energy=b.energy-a.energy;high[2]=energy/ctx.scale[2];high[4]=(b.k-a.k-integralK)/ctx.scale[4];if(i>1){GpuReal volume=(w.y[i+1]-w.y[i-1])/2,z=w.state[i][5];high[5]=-(b.omega-a.omega-integralO)*z*z*z*z/(2*volume*ctx.scale[5]);}}
 if(!cfg.enableSst){high[4]=w.state[i][4];high[5]=w.state[i][5]-1;}for(int j=0;j<6;++j){w.residual[i][j]=high[j];norm=std::max(norm,std::abs(high[j]));}}
 return physicalValid;}
UGKWP_GAS_HD BoundedVector<int,2*M> hermiteBranches(const Config&cfg,const W&w){BoundedVector<int,2*M>b;for(int j:{4,5}){for(int i=1;i<cfg.nodes-1;++i){GpuReal dl=w.state[i][j]-w.state[i-1][j],dr=w.state[i+1][j]-w.state[i][j];b.push_back((dl>0?1:dl<0?-1:0)*3+(dr>0?1:dr<0?-1:0));}int i=cfg.nodes-1;if(i==1){b.push_back(2);continue;}GpuReal h0=w.y[i]-w.y[i-1],h1=w.y[i-1]-w.y[i-2],d0=(w.state[i][j]-w.state[i-1][j])/h0,d1=(w.state[i-1][j]-w.state[i-2][j])/h1,sl=((2*h0+h1)*d0-h0*d1)/(h0+h1);b.push_back(sl*d0<=0?0:d0*d1<0&&std::abs(sl)>3*std::abs(d0)?1:2);}return b;}
// Publication retains old species-row semantics and replaces every changed high-operator field.
UGKWP_GAS_HD bool completePublication(const Input&in,const Config&cfg,W&w,const Ctx&ctx,Profile&p,int iterations,GpuReal norm,WallOutput<GpuReal,2>&out,GpuReal totalR,GpuReal totalStress,GpuReal totalAdvective,bool physicalValid){
 auto& faces=s.faces;
 out.heatConduction=(out.traceTemperature-in.matching.temperature)/totalR;out.heatViscous=totalStress/totalR;detail::LayerPoint<GpuReal,2> wallPoint;if(!detail::pointState(in,cfg,ctx,w.state[0],wallPoint))return false;out.heatTransport=totalAdvective/totalR-coherent_energy::wallTransport(in,ctx,wallPoint);out.heatBalanceResidual=out.heatConduction+out.heatViscous+out.heatTransport-out.conductiveHeatFlux;
 auto wall=p.physicalPrimitive(0);physical::Face<GpuReal,2>wallFace;physical::Source<GpuReal,2>wallSource;physicalValid=trackedEvaluate(in,cfg,wall,0.f,in.normal,ctx.wallNormal,wallFace,wallSource)&&physicalValid;out.wallKFlux=wallFace.k;
 if(cfg.enableSst&&in.volumeSstClosure){out.kFluxCount=cfg.nodes+1;out.kFluxHeight[0]=0;out.kTransportFlux[0]=out.wallKFlux;out.omegaDiffusiveFlux[0]=0;
 for(int i=0;i<cfg.nodes-1;++i){GpuReal y=(w.y[i]+w.y[i+1])/2;out.kFluxHeight[i+1]=y;out.kTransportFlux[i+1]=faces[i].k;out.omegaDiffusiveFlux[i+1]=faces[i].omega-ctx.mass*p.base(y).omega;}
 GpuReal lo=(w.y[cfg.nodes-2]+w.y[cfg.nodes-1])/2,hi=w.y[cfg.nodes-1],sk=0,so=0;for(GpuReal g:{-.57735026918962576451f,.57735026918962576451f}){GpuReal y=(lo+hi)/2+(hi-lo)/2*g;auto v=basePhysical(p,y);physical::Face<GpuReal,2>f;physical::Source<GpuReal,2>src;physicalValid=trackedEvaluate(in,cfg,v,y,in.normal,ctx.wallNormal,f,src)&&physicalValid;sk+=(hi-lo)/2*src.k;so+=(hi-lo)/2*src.omega;}
 out.kFluxHeight[cfg.nodes]=hi;out.kTransportFlux[cfg.nodes]=faces[cfg.nodes-2].k+sk;out.omegaDiffusiveFlux[cfg.nodes]=faces[cfg.nodes-2].omega+so-ctx.mass*p.base(hi).omega;
 }
 if(cfg.enableSst&&in.integrateOwnerKSource){out.integratedKSource=out.ownerKContent=out.ownerKDestruction=out.ownerKBalanceResidual=0;const auto&q=in.quadrature;for(int j=0;j<q.count;++j){GpuReal y=q.distance[j],weight=q.volumeWeight[j];auto v=basePhysical(p,y);physical::Face<GpuReal,2>f;physical::Source<GpuReal,2>src;physicalValid=trackedEvaluate(in,cfg,v,y,in.normal,ctx.wallNormal,f,src)&&physicalValid;out.integratedKSource+=weight*src.k;out.ownerKContent+=weight*v.rho*v.k;out.ownerKDestruction+=weight*v.rho*in.model.sstCoefficients.betaStar*v.k*v.omega;if(in.volumeSstClosure){int control=0;while(control+1<cfg.nodes&&out.kFluxHeight[control+1]<y)++control;out.ownerKBalanceResidual-=weight*(out.kTransportFlux[control+1]-out.kTransportFlux[control])/(out.kFluxHeight[control+1]-out.kFluxHeight[control]);}}out.ownerKBalanceResidual+=out.integratedKSource;}
 out.iterations=iterations;out.residual=norm;return physicalValid;
}
UGKWP_GAS_HD bool publish(const Input&in,const Config&cfg,W&w,const Ctx&ctx,int iterations,GpuReal norm,WallOutput<GpuReal,2>&out){
 auto high=out;if(!detail::publishLayer(in,cfg,ctx,w,iterations,norm,out))return false;
 out.conductiveHeatFlux=high.conductiveHeatFlux;for(int a=0;a<3;++a)out.traction[a]=high.traction[a];
 Profile p(in,cfg,w,s.transformSlope);for(int i=1;i<cfg.nodes;++i)for(int j=0;j<2;++j)cachedUslopes[i][j]=p.nodeSlope(i,j);p.setWallFlux(out);Segment seg;seg.lo=0;seg.hi=w.y[cfg.nodes-1];p.segments.push_back(seg);
 if(cfg.enableSst)out.ownerOmega=p.base(in.ownerDistance).omega;
 bool physicalValid=true;GpuReal totalR=0,totalStress=0,totalAdvective=0;auto& faces=s.faces;for(int i=0;i<cfg.nodes-1;++i){GpuReal R,W,A;detail::LayerFlux<GpuReal,2> old;if(!detail::intervalFlux(in,cfg,ctx,w.state[i],w.state[i+1],w.y[i],w.y[i+1],old))return false;faces[i]=resistanceFace(p,i,old.species,&R,&W,&physicalValid,&A);totalR+=R;totalStress+=W;totalAdvective+=A;}
 return completePublication(in,cfg,w,ctx,p,iterations,norm,out,totalR,totalStress,totalAdvective,physicalValid);
}
template<class Stats> UGKWP_GAS_HD bool solve(const Input&in,const Config&originalConfig,W&w,Cache<M>&cache,WallOutput<GpuReal,2>&out,Stats&stats){
 auto cfg=originalConfig;cfg.coherentDiscretization=true;bool ok=true;int seedIts=0;if(!cache.haveState){WallStatus status;ok=evaluateWallModel(in,cfg,w,out,status);seedIts=out.iterations;}Ctx ctx;detail::makeContext(in,cfg,ctx);{int e=cfg.nodes-1;w.state[e][0]=ctx.edgeU[0];w.state[e][1]=ctx.edgeU[1];w.state[e][2]=in.matching.temperature-in.temperature;w.state[e][3]=in.matching.massFraction[ctx.independent[0]];w.state[e][4]=cfg.enableSst?std::sqrt(in.matching.k):0;w.state[e][5]=cfg.enableSst?1/std::sqrt(in.matching.omega):1;}for(int i=0;i<cfg.nodes;++i)w.y[i]=in.matchingDistance*std::pow(GpuReal(i)/(cfg.nodes-1),cfg.stretch);GpuReal norm=0;int evaluations=0,its=0;ok=refresh(in,cfg,w,ctx,norm,out)&&ok;++evaluations;GpuReal maxOffBand=0;
 const int n=cfg.nodes*6;auto&lu=cache.factors;auto&cachedScale=cache.scale;int jacobianBuilds=0;auto&base=s.base;auto&res=s.res;auto&a=s.matrix;auto&rhs=s.rhs;auto&scale=s.scale;auto&plus=s.plus;auto&probe=s.probe;auto&trialOut=s.trialOut;base.resize(n);res.resize(n);rhs.resize(n);scale.resize(n);plus.resize(n);
 while(ok&&norm>cfg.absoluteTolerance+cfg.relativeTolerance&&its<cfg.maxIterations){
 for(int j=0;j<n;++j){base[j]=w.state[j/6][j%6];res[j]=w.residual[j/6][j%6];rhs[j]=-res[j];}
 auto branches=hermiteBranches(cfg,w);const int mask=(cfg.enableSst&&ctx.mass>0?1:0)|(in.adiabatic?2:0);bool fresh=!lu.valid||lu.n!=n||cache.dirichletMask!=mask||!cache.haveState||branches!=cache.branches;if(fresh){collectNutBranches=true;bool branchBaseValid=refresh(in,cfg,w,ctx,norm,out);++evaluations;if(!branchBaseValid){ok=false;collectNutBranches=false;break;}auto&baseNutBranches=s.baseNutBranches;baseNutBranches=currentNutBranches;++jacobianBuilds;a.assign(n*n,0.f);auto&steps=s.steps;steps.resize(n);
 for(int col=0;col<6&&ok;++col)for(int color=0;color<5&&ok;++color){copyActive(cfg,probe,w);
 for(int node=color;node<cfg.nodes;node+=5){int j=node*6+col;GpuReal old=base[j],eps=std::sqrt(std::numeric_limits<GpuReal>::epsilon()),vs=col==2?100:1;if(col==4)vs=std::sqrt(std::max(1.f,in.matching.k))*eps;if(col==5)vs=eps/std::sqrt(in.matching.omega);scale[j]=std::max(std::abs(old),vs);GpuReal h=eps*scale[j];if(col==3&&old>=.5f)h=-h;probe.state[node][col]=old+h;if((col==4||col==5)&&!pchipFdCompatible(cfg,w,probe,node,col)){probe.state[node][col]=old-h;if(!pchipFdCompatible(cfg,w,probe,node,col))probe.state[node][col]=old+h;}steps[j]=probe.state[node][col]-old;}
 GpuReal nn;bool valid=refresh(in,cfg,probe,ctx,nn,trialOut);++evaluations;
 // Try an equally sized opposite batch before shrinking unrelated columns.
 bool mirrorAdmissible=true;
 for(int node=color;node<cfg.nodes;node+=5){int j=node*6+col;if(!fdProbeAdmissible(in,col,base[j]-steps[j]))mirrorAdmissible=false;}
 if(valid&&mirrorAdmissible&&(currentNutBranches!=baseNutBranches||!pchipFdCompatible(cfg,w,probe))){
  copyActive(cfg,probe,w);
  for(int node=color;node<cfg.nodes;node+=5){int j=node*6+col;s.originalSteps[j]=steps[j];probe.state[node][col]=base[j]-s.originalSteps[j];steps[j]=probe.state[node][col]-base[j];}
  bool mirroredValid=refresh(in,cfg,probe,ctx,nn,trialOut);++evaluations;
  if(mirroredValid&&currentNutBranches==baseNutBranches&&pchipFdCompatible(cfg,w,probe))valid=true;
  else{
   copyActive(cfg,probe,w);
   for(int node=color;node<cfg.nodes;node+=5){int j=node*6+col;steps[j]=s.originalSteps[j];probe.state[node][col]=base[j]+steps[j];}
   valid=refresh(in,cfg,probe,ctx,nn,trialOut);++evaluations;
  }
 }
 int branchHalves=0;while(valid&&(currentNutBranches!=baseNutBranches||!pchipFdCompatible(cfg,w,probe))){copyActive(cfg,probe,w);bool changed=false;for(int node=color;node<cfg.nodes;node+=5){int j=node*6+col;GpuReal h=steps[j]/2,old=base[j];probe.state[node][col]=old+h;GpuReal actual=probe.state[node][col]-old;if(actual!=0&&actual!=steps[j]){steps[j]=actual;changed=true;}else probe.state[node][col]=old+steps[j];}if(!changed){++s.fdRoundoff;break;}valid=refresh(in,cfg,probe,ctx,nn,trialOut);++evaluations;++branchHalves;}s.fdBranchHalves+=branchHalves;
 bool fallback=!valid||currentNutBranches!=baseNutBranches||!pchipFdCompatible(cfg,w,probe);
 for(int node=color;node<cfg.nodes;node+=5){int j=node*6+col;bool zero=col==4&&base[j]!=0;int lo=std::max(0,node-2),hi=std::min(cfg.nodes-1,node+2);for(int i=lo;i<=hi;++i)if(probe.residual[i][4]!=res[i*6+4])zero=false;
 bool fullZero=true;for(int i=lo*6;i<(hi+1)*6;++i)if(probe.residual[i/6][i%6]!=res[i])fullZero=false;
 if(zero||fullZero)fallback=true;
 for(int i=lo*6;i<(hi+1)*6;++i)a[i*n+j]=(probe.residual[i/6][i%6]-res[i])/steps[j]*scale[j];} if(fallback){ for(int node=color;node<cfg.nodes&&ok;node+=5){int j=node*6+col;GpuReal old=base[j],eps=std::sqrt(std::numeric_limits<GpuReal>::epsilon()),vs=col==2?100:1;if(col==4)vs=std::sqrt(std::max(1.f,in.matching.k))*eps;if(col==5)vs=eps/std::sqrt(in.matching.omega);GpuReal h=eps*std::max(std::abs(old),vs);if(col==3&&old>=.5f)h=-h;GpuReal nn;bool valid=false;
 copyActive(cfg,probe,w);probe.state[node][col]=old+h;if((col==4||col==5)&&!pchipFdCompatible(cfg,w,probe,node,col)){probe.state[node][col]=old-h;if(pchipFdCompatible(cfg,w,probe,node,col))h=-h;}
 // Prefer ordinary-sized per-column directions after a colored obstruction.
 // Only the newly added opposite direction gets the alternative-domain check.
 GpuReal directionH=h;
 int direction=fdSelectDirection([&](int sign){GpuReal candidate=old+sign*directionH;if(sign<0&&!fdProbeAdmissible(in,col,candidate))return false;copyActive(cfg,probe,w);probe.state[node][col]=candidate;valid=refresh(in,cfg,probe,ctx,nn,trialOut);++evaluations;return valid&&currentNutBranches==baseNutBranches&&pchipFdCompatible(cfg,w,probe);});
 if(direction)h=direction*directionH;
 else{
  h=directionH;valid=false;
 for(int attempt=0;attempt<2&&!valid;++attempt){copyActive(cfg,probe,w);probe.state[node][col]=old+h;valid=refresh(in,cfg,probe,ctx,nn,trialOut);++evaluations;if(!valid)h=col==3?h/2:-h;}
 }
 while(valid&&(currentNutBranches!=baseNutBranches||!pchipFdCompatible(cfg,w,probe))){GpuReal nextH=h/2;if(old+nextH==old||old+nextH==old+h){++s.fdRoundoff;break;}h=nextH;copyActive(cfg,probe,w);probe.state[node][col]=old+h;valid=refresh(in,cfg,probe,ctx,nn,trialOut);++evaluations;}
 if(!valid){ok=false;++s.fdFailures;break;}GpuReal actual=probe.state[node][col]-old;bool zero=col==4&&old!=0;for(int i=0;i<cfg.nodes;++i)if(probe.residual[i][4]!=res[i*6+4])zero=false;
 for(int i=0;i<n;++i)plus[i]=probe.residual[i/6][i%6];
 if(zero){++s.centralDirections;h=std::cbrt(std::numeric_limits<GpuReal>::epsilon())*std::max(std::abs(old),std::sqrt(std::max(1.f,in.matching.k)));while(true){copyActive(cfg,probe,w);probe.state[node][col]=old+h;valid=refresh(in,cfg,probe,ctx,nn,trialOut);++evaluations;bool samePlus=currentNutBranches==baseNutBranches&&pchipFdCompatible(cfg,w,probe);for(int i=0;i<n;++i)plus[i]=probe.residual[i/6][i%6];GpuReal hp=probe.state[node][col]-old;copyActive(cfg,probe,w);probe.state[node][col]=old-h;valid=refresh(in,cfg,probe,ctx,nn,trialOut)&&valid;++evaluations;actual=hp-(probe.state[node][col]-old);if(!valid){ok=false;break;}if(samePlus&&currentNutBranches==baseNutBranches&&pchipFdCompatible(cfg,w,probe))break;GpuReal nextH=h/2;if(old+nextH==old||old-nextH==old||old+nextH==old+h){++s.fdRoundoff;break;}h=nextH;}if(!ok)break;}

 scale[j]=std::max(std::abs(old),vs);for(int i=0;i<n;++i){a[i*n+j]=(plus[i]-(zero?probe.residual[i/6][i%6]:res[i]))/actual*scale[j];if(std::abs(i/6-node)>2)maxOffBand=std::max(maxOffBand,std::abs(a[i*n+j]));}}
}
 }
 cachedScale=scale;cache.branches=branches;cache.dirichletMask=mask;collectNutBranches=false;
 cache.boundary.clear();
 // Store only exact Dirichlet elimination transforms, never a second raw J.
 // Nonzero boundary RHS still receives the original interior coupling.
 for(int node:{0,cfg.nodes-1})for(int col=0;col<6;++col){if(node==0&&(col==3||(col==2&&in.adiabatic)||(col==5&&cfg.enableSst&&ctx.mass>0)))continue;int j=node*6+col;
 BoundaryTransform t;t.column=j;t.lo=std::max(0,j-17);t.hi=std::min(n,j+18);t.diagonal=a[j*n+j];for(int i=t.lo;i<t.hi;++i)t.values[i-t.lo]=a[i*n+j];cache.boundary.push_back(t);
 for(int i=0;i<n;++i)if(i!=j)a[i*n+j]=0;
 for(int k=0;k<n;++k)a[j*n+k]=0;
 a[j*n+j]=1;}
 if(!ok||!lu.factor(a,n)){ok=false;++s.factorFailures;break;}
 }else scale=cachedScale;
 for(const auto&t:cache.boundary){int j=t.column;GpuReal step=rhs[j]/t.diagonal;for(int i=t.lo;i<t.hi;++i)if(i!=j)rhs[i]-=t.values[i-t.lo]*step;rhs[j]=step;}
 lu.solve(rhs);
 GpuReal previousNorm=norm;GpuReal alpha=1,next=norm;bool accepted=false;for(int line=0;line<cfg.numerics.maximumLineSearchIterations;++line){copyActive(cfg,probe,w);bool feasible=true;for(int i=0;i<n;++i)probe.state[i/6][i%6]=base[i]+alpha*rhs[i]*scale[i];for(int i=0;i<cfg.nodes;++i)feasible=detail::feasibleLayerComposition<GpuReal,2>(probe.state[i])&&feasible;
 bool valid=false;if(feasible){valid=refresh(in,cfg,probe,ctx,next,trialOut);++evaluations;}if(valid&&std::isfinite(next)&&next<norm){copyActive(cfg,w,probe);out=trialOut;norm=next;accepted=true;break;}alpha*=cfg.numerics.lineSearchReduction;}
 ++its;if(!accepted){if(fresh)break;lu.valid=false;}else if(alpha<1||norm>previousNorm*cfg.numerics.jacobianRefreshReduction)lu.valid=false;
 }

 cache.haveState=true;stats.iterations=seedIts+its;stats.residualEvaluations=evaluations;stats.factorizations=jacobianBuilds;stats.physicalResidual=norm;if(ok)ok=publish(in,cfg,w,ctx,seedIts+its,norm,out);stats.seconds=0;return ok;
}

};
// Every lane enters the same control flow and collectives. Residual rows,
// interval fluxes and colored FD columns have one writer; ordered reductions,
// local first-face closure, LU and publication retain their scalar owner.
template<int M,class Team=detail::SerialWallTeam> struct Engine : ScalarEngine<M> {
 using Base=ScalarEngine<M>;using W=typename Base::W;using Ctx=typename Base::Ctx;
 using Base::s;using Base::cachedUslopes;using Base::collectNutBranches;
 using Base::nutBranchPhase;using Base::currentNutBranches;using Base::observerSlot;
 using Base::trackedEvaluate;using Base::trackOldInterval;using Base::basePhysical;
 using Base::resistanceFace;using Base::closeFirstFace;using Base::hermiteBranches;using Base::publish;using Base::completePublication;
 Team team;
 UGKWP_GAS_HD explicit Engine(Scratch<M>&scratch,const Team&t=Team{}):Base(scratch),team(t){}
 UGKWP_GAS_HD void copyActive(const Config&cfg,W&dest,const W&src){
  for(int i=team.rank();i<cfg.nodes;i+=team.size()){dest.y[i]=src.y[i];for(int j=0;j<6;++j){dest.state[i][j]=src.state[i][j];dest.residual[i][j]=src.residual[i][j];}}team.sync();
 }
 // Independent interval work is distributed, while the final R/W/A sums
 // retain exactly the scalar interval order and the same publication helper.
 UGKWP_GAS_HD bool publishTeam(const Input&in,const Config&cfg,W&w,const Ctx&ctx,int iterations,GpuReal norm,WallOutput<GpuReal,2>&out){
  bool valid=false;
  if(team.leader()){
   auto high=out;valid=detail::publishLayer(in,cfg,ctx,w,iterations,norm,out);
   if(valid){out.conductiveHeatFlux=high.conductiveHeatFlux;for(int a=0;a<3;++a)out.traction[a]=high.traction[a];}
  }
  if(!team.broadcast(valid))return false;
  Profile p(in,cfg,w,s.transformSlope,team);
  for(int i=1+team.rank();i<cfg.nodes;i+=team.size())for(int j=0;j<2;++j)cachedUslopes[i][j]=p.nodeSlope(i,j);
  team.sync();p.setWallFlux(out);Segment seg;seg.lo=0;seg.hi=w.y[cfg.nodes-1];p.segments.push_back(seg);
  if(team.leader()&&cfg.enableSst)out.ownerOmega=p.base(in.ownerDistance).omega;
  team.sync();bool intervalValid=true,physicalValid=true;
  for(int i=team.rank();i<cfg.nodes-1;i+=team.size()){
   GpuReal R,W,A;detail::LayerFlux<GpuReal,2> old;
   bool intervalOk=detail::intervalFlux(in,cfg,ctx,w.state[i],w.state[i+1],w.y[i],w.y[i+1],old);intervalValid=intervalOk&&intervalValid;
   if(intervalOk){s.faces[i]=resistanceFace(p,i,old.species,&R,&W,&physicalValid,&A);s.publicationIntegral[i][0]=R;s.publicationIntegral[i][1]=W;s.publicationIntegral[i][2]=A;}
  }
  if(!team.all(intervalValid))return false;
  physicalValid=team.all(physicalValid);bool published=false;
  if(team.leader()){
   GpuReal totalR=0,totalStress=0,totalAdvective=0;
   for(int i=0;i<cfg.nodes-1;++i){totalR+=s.publicationIntegral[i][0];totalStress+=s.publicationIntegral[i][1];totalAdvective+=s.publicationIntegral[i][2];}
   published=completePublication(in,cfg,w,ctx,p,iterations,norm,out,totalR,totalStress,totalAdvective,physicalValid);
  }
  return team.broadcast(published);
 }
 // Only the leader inspects a shared probe before a collective permits writes.
 UGKWP_GAS_HD bool sameBranches(const Config&cfg,const W&w,const W&probe){bool same=false;if(team.leader())same=currentNutBranches==s.baseNutBranches&&pchipFdCompatible(cfg,w,probe);return team.broadcast(same);}
 UGKWP_GAS_HD bool refresh(const Input&in,const Config&cfg,W&w,const Ctx&ctx,GpuReal&norm,WallOutput<GpuReal,2>&out){
  const int nodes=cfg.nodes,edge=nodes-1;
  if(team.leader()){currentNutBranches.clear();nutBranchPhase=false;}
  for(int i=team.rank();i<9*nodes;i+=team.size())s.nutSlotUsed[i]=false;
  team.sync();auto&low=s.low;auto&points=s.points;auto&oldFlux=s.oldFlux;auto&faces=s.faces;
  bool valid=true;
  for(int i=team.rank();i<nodes;i+=team.size()){points[i]={};oldFlux[i]={};bool pointOk=detail::pointState(in,cfg,ctx,w.state[i],points[i]);valid=pointOk&&valid;for(int j=0;j<6;++j)low[i][j]=0;}
  if(!team.all(valid))return false;
  for(int i=team.rank();i<edge;i+=team.size()){
   auto a=points[i],b=points[i+1];detail::pointSst(in,cfg,w.y[i],points[i],points[i+1],w.y[i+1]-w.y[i],a);detail::pointSst(in,cfg,w.y[i+1],points[i],points[i+1],w.y[i+1]-w.y[i],b);
   observerSlot=2*i;trackOldInterval(in,cfg,points[i],points[i+1],a,w.y[i+1]-w.y[i],w.y[i]);trackOldInterval(in,cfg,points[i],points[i+1],b,w.y[i+1]-w.y[i],w.y[i+1]);
   bool intervalOk=detail::layerFlux(in,cfg,ctx,a,b,w.y[i+1]-w.y[i],oldFlux[i],w.y[i],w.y[i+1]);valid=intervalOk&&valid;
  }
  if(!team.all(valid))return false;
  for(int i=team.rank();i<edge;i+=team.size()){auto p=points[i];bool reactionOk=detail::pointReaction(in,p);valid=reactionOk&&valid;GpuReal volume=i==0?w.y[1]/2:(w.y[i+1]-w.y[i-1])/2;int sp=ctx.independent[0];low[i][3]=(oldFlux[i].species[sp]-(i==0?in.massFlux[sp]:oldFlux[i-1].species[sp])-volume*p.source[sp])/ctx.scale[3];}
  if(!team.all(valid))return false;
  GpuReal wallEnergy=coherent_energy::wallTransport(in,ctx,points[0]);
  if(team.leader()){
   for(int j:{0,1,2,4,5})low[0][j]=w.state[0][j];
   if(!cfg.enableSst)low[0][5]-=1;
   low[edge][0]=w.state[edge][0]-ctx.edgeU[0];low[edge][1]=w.state[edge][1]-ctx.edgeU[1];low[edge][2]=w.state[edge][2]-(in.matching.temperature-in.temperature);low[edge][3]=w.state[edge][3]-in.matching.massFraction[ctx.independent[0]];low[edge][4]=w.state[edge][4]-(cfg.enableSst?std::sqrt(in.matching.k):0);low[edge][5]=w.state[edge][5]-(cfg.enableSst?1/std::sqrt(in.matching.omega):1);
   if(nodes>2&&cfg.enableSst)low[1][5]=points[1].z-points[0].z-w.y[1]*std::sqrt(points[0].rho*in.model.sstCoefficients.beta1/(6*points[0].mu));
   out={};out.conductiveHeatFlux=oldFlux[0].energy-wallEnergy;for(int d=0;d<2;++d)for(int a=0;a<3;++a)out.traction[a]-=oldFlux[0].momentum[d]*ctx.tangent[d][a];
  }
  team.sync();Profile p(in,cfg,w,s.transformSlope,team);
  for(int i=1+team.rank();i<nodes;i+=team.size())for(int j=0;j<2;++j)cachedUslopes[i][j]=p.nodeSlope(i,j);
  team.sync();p.setWallFlux(out);Segment seg;seg.lo=0;seg.hi=w.y[edge];p.segments.push_back(seg);team.sync();
  GpuReal closure=0,scaledClosure=0;bool physicalValid=true;observerSlot=-1;
  if(team.leader())physicalValid=closeFirstFace(p,oldFlux[0].species,wallEnergy,out,closure,scaledClosure);
  if(team.leader()&&in.adiabatic)low[0][2]=out.conductiveHeatFlux/ctx.scale[2];
  physicalValid=team.broadcast(physicalValid);scaledClosure=team.broadcast(scaledClosure);
  // Each lane owns its Profile correction; propagate the leader's closed flux.
  p.setWallFlux(out);if(team.leader())nutBranchPhase=true;team.sync();
  for(int i=team.rank();i<edge;i+=team.size()){observerSlot=2*edge+3*i;faces[i]=resistanceFace(p,i,oldFlux[i].species,nullptr,nullptr,&physicalValid);}
  physicalValid=team.all(physicalValid);
  bool wallValid=true;
  if(team.leader()&&cfg.enableSst&&ctx.mass>0){GpuReal tau2=0;for(int d=0;d<2;++d){GpuReal tau=ctx.mass*ctx.wallU[d];for(int a=0;a<3;++a)tau-=faces[0].momentum[a]*ctx.tangent[d][a];tau2+=tau*tau;}GpuReal uTau=std::sqrt(std::sqrt(tau2)/points[0].rho),v=ctx.mass/points[0].rho;if(!(uTau>0))wallValid=detail::wallFatal(in,ugkp::diagnostics::Code::ZeroDenominator,10012,uTau);else{GpuReal vp=v/uTau;low[0][5]=w.state[0][5]-std::sqrt((points[0].mu/points[0].rho)/(uTau*uTau)*vp*(1+5*vp)/25);}}
  if(!team.broadcast(wallValid))return false;
  valid=true;
  for(int i=team.rank();i<nodes;i+=team.size()){
   GpuReal high[6];for(int j=0;j<6;++j)high[j]=low[i][j];
   if(i>0&&i<edge){GpuReal integralK=0,integralO=0;observerSlot=5*edge+4*(i-1);
    for(int side=0;side<2;++side){GpuReal lo=side?w.y[i]:(w.y[i-1]+w.y[i])/2,hi=side?(w.y[i]+w.y[i+1])/2:w.y[i];const GpuReal g[2]={-.57735026918962576451f,.57735026918962576451f},wt[2]={1,1};for(int q=0;q<2;++q){GpuReal y=(hi+lo)/2+(hi-lo)/2*g[q];auto v=basePhysical(p,y);physical::Face<GpuReal,2>f;physical::Source<GpuReal,2>src;bool evaluated=trackedEvaluate(in,cfg,v,y,in.normal,ctx.wallNormal,f,src);valid=evaluated&&valid;integralK+=(hi-lo)/2*wt[q]*src.k;integralO+=(hi-lo)/2*wt[q]*src.omega;}}
    auto&a=faces[i-1];auto&b=faces[i];GpuReal mom[2]{};for(int d=0;d<2;++d){for(int k=0;k<3;++k)mom[d]+=(b.momentum[k]-a.momentum[k])*ctx.tangent[d][k];high[d]=mom[d]/ctx.scale[d];}GpuReal energy=b.energy-a.energy;high[2]=energy/ctx.scale[2];high[4]=(b.k-a.k-integralK)/ctx.scale[4];if(i>1){GpuReal volume=(w.y[i+1]-w.y[i-1])/2,z=w.state[i][5];high[5]=-(b.omega-a.omega-integralO)*z*z*z*z/(2*volume*ctx.scale[5]);}
   }
   if(!cfg.enableSst){high[4]=w.state[i][4];high[5]=w.state[i][5]-1;}for(int j=0;j<6;++j)w.residual[i][j]=high[j];
  }
  if(!team.all(valid))return false;
  if(team.leader()){
   norm=scaledClosure;for(int i=0;i<nodes;++i)for(int j=0;j<6;++j)norm=std::max(norm,std::abs(w.residual[i][j]));
   for(int i=0;i<9*nodes-13;++i)if(s.nutSlotUsed[i])currentNutBranches.push_back(s.nutSlotValue[i]);
  }
  norm=team.broadcast(norm);observerSlot=-1;return physicalValid;
 }
 template<class Stats> UGKWP_GAS_HD bool solve(const Input&in,const Config&originalConfig,W&w,Cache<M>&cache,WallOutput<GpuReal,2>&out,Stats&stats){
  auto cfg=originalConfig;cfg.coherentDiscretization=true;bool ok=true;int seedIts=0;
  if(!cache.haveState){WallStatus status;if(cfg.model==BoundaryLayerModel::ReactingSst)ok=evaluateReactingWallLayerImpl(in,cfg,w,out,status,team);else{if(team.leader())ok=evaluateWallModel(in,cfg,w,out,status);ok=team.broadcast(ok);}seedIts=out.iterations;}
  Ctx ctx;detail::makeContext(in,cfg,ctx);
  if(team.leader()){int e=cfg.nodes-1;w.state[e][0]=ctx.edgeU[0];w.state[e][1]=ctx.edgeU[1];w.state[e][2]=in.matching.temperature-in.temperature;w.state[e][3]=in.matching.massFraction[ctx.independent[0]];w.state[e][4]=cfg.enableSst?std::sqrt(in.matching.k):0;w.state[e][5]=cfg.enableSst?1/std::sqrt(in.matching.omega):1;}
  for(int i=team.rank();i<cfg.nodes;i+=team.size())w.y[i]=in.matchingDistance*std::pow(GpuReal(i)/(cfg.nodes-1),cfg.stretch);
  team.sync();GpuReal norm=0;int evaluations=0,its=0;ok=refresh(in,cfg,w,ctx,norm,out)&&ok;++evaluations;
  const int n=cfg.nodes*6;auto&lu=cache.factors;auto&cachedScale=cache.scale;int jacobianBuilds=0;auto&base=s.base;auto&res=s.res;auto&a=s.matrix;auto&rhs=s.rhs;auto&scale=s.scale;auto&plus=s.plus;auto&probe=s.probe;auto&trialOut=s.trialOut;
  if(team.leader()){base.resize(n);res.resize(n);rhs.resize(n);scale.resize(n);plus.resize(n);}team.sync();
  while(ok&&norm>cfg.absoluteTolerance+cfg.relativeTolerance&&its<cfg.maxIterations){
   for(int j=team.rank();j<n;j+=team.size()){base[j]=w.state[j/6][j%6];res[j]=w.residual[j/6][j%6];rhs[j]=-res[j];}team.sync();
   auto branches=hermiteBranches(cfg,w);const int mask=(cfg.enableSst&&ctx.mass>0?1:0)|(in.adiabatic?2:0);
   bool fresh=!lu.valid||lu.n!=n||cache.dirichletMask!=mask||!cache.haveState||branches!=cache.branches;
   if(fresh){
    if(team.leader())collectNutBranches=true;
    team.sync();bool branchBaseValid=refresh(in,cfg,w,ctx,norm,out);++evaluations;
    if(!branchBaseValid){ok=false;if(team.leader())collectNutBranches=false;team.sync();break;}
    auto&baseNutBranches=s.baseNutBranches;auto&steps=s.steps;
    if(team.leader()){baseNutBranches=currentNutBranches;a.resize(n*n);steps.resize(n);}++jacobianBuilds;
    for(int i=team.rank();i<n*n;i+=team.size())a[i]=0.f;
    team.sync();
    for(int col=0;col<6&&ok;++col)for(int color=0;color<5&&ok;++color){
     copyActive(cfg,probe,w);
     for(int node=color+5*team.rank();node<cfg.nodes;node+=5*team.size()){
      int j=node*6+col;GpuReal old=base[j],eps=std::sqrt(std::numeric_limits<GpuReal>::epsilon()),vs=col==2?100:1;
      if(col==4)vs=std::sqrt(std::max(1.f,in.matching.k))*eps;
      if(col==5)vs=eps/std::sqrt(in.matching.omega);
      scale[j]=std::max(std::abs(old),vs);GpuReal h=eps*scale[j];if(col==3&&old>=.5f)h=-h;probe.state[node][col]=old+h;
      if((col==4||col==5)&&!pchipFdCompatible(cfg,w,probe,node,col)){probe.state[node][col]=old-h;if(!pchipFdCompatible(cfg,w,probe,node,col))probe.state[node][col]=old+h;}steps[j]=probe.state[node][col]-old;
     }
     team.sync();GpuReal nn;bool valid=refresh(in,cfg,probe,ctx,nn,trialOut);++evaluations;
     bool mirrorAdmissible=true;
     for(int node=color+5*team.rank();node<cfg.nodes;node+=5*team.size()){
      int j=node*6+col;if(!fdProbeAdmissible(in,col,base[j]-steps[j]))mirrorAdmissible=false;
     }
     mirrorAdmissible=team.all(mirrorAdmissible);
     if(valid&&mirrorAdmissible&&!sameBranches(cfg,w,probe)){
      copyActive(cfg,probe,w);
      for(int node=color+5*team.rank();node<cfg.nodes;node+=5*team.size()){
       int j=node*6+col;s.originalSteps[j]=steps[j];probe.state[node][col]=base[j]-s.originalSteps[j];steps[j]=probe.state[node][col]-base[j];
      }
      team.sync();bool mirroredValid=refresh(in,cfg,probe,ctx,nn,trialOut);++evaluations;
      if(mirroredValid&&sameBranches(cfg,w,probe))valid=true;
      else{
       copyActive(cfg,probe,w);
       for(int node=color+5*team.rank();node<cfg.nodes;node+=5*team.size()){
        int j=node*6+col;steps[j]=s.originalSteps[j];probe.state[node][col]=base[j]+steps[j];
       }
       team.sync();valid=refresh(in,cfg,probe,ctx,nn,trialOut);++evaluations;
      }
     }
     int branchHalves=0;
     while(valid&&!sameBranches(cfg,w,probe)){
      copyActive(cfg,probe,w);bool changed=false;
      for(int node=color+5*team.rank();node<cfg.nodes;node+=5*team.size()){int j=node*6+col;GpuReal h=steps[j]/2,old=base[j];probe.state[node][col]=old+h;GpuReal actual=probe.state[node][col]-old;if(actual!=0&&actual!=steps[j]){steps[j]=actual;changed=true;}else probe.state[node][col]=old+steps[j];}
      changed=!team.all(!changed);if(!changed){if(team.leader())++s.fdRoundoff;team.sync();break;}
      valid=refresh(in,cfg,probe,ctx,nn,trialOut);++evaluations;++branchHalves;
     }
     if(team.leader())s.fdBranchHalves+=branchHalves;
     bool fallback=!valid||!sameBranches(cfg,w,probe);
     for(int node=color+5*team.rank();node<cfg.nodes;node+=5*team.size()){
      int j=node*6+col;bool zero=col==4&&base[j]!=0;int lo=std::max(0,node-2),hi=std::min(cfg.nodes-1,node+2);for(int i=lo;i<=hi;++i)if(probe.residual[i][4]!=res[i*6+4])zero=false;
      bool fullZero=true;for(int i=lo*6;i<(hi+1)*6;++i)if(probe.residual[i/6][i%6]!=res[i])fullZero=false;
      if(zero||fullZero)fallback=true;
      for(int i=lo*6;i<(hi+1)*6;++i)a[i*n+j]=(probe.residual[i/6][i%6]-res[i])/steps[j]*scale[j];
     }
     fallback=!team.all(!fallback);
     if(fallback){
      for(int node=color;node<cfg.nodes&&ok;node+=5){
       int j=node*6+col;GpuReal old=base[j],eps=std::sqrt(std::numeric_limits<GpuReal>::epsilon()),vs=col==2?100:1;
       if(col==4)vs=std::sqrt(std::max(1.f,in.matching.k))*eps;
       if(col==5)vs=eps/std::sqrt(in.matching.omega);
       GpuReal h=eps*std::max(std::abs(old),vs);if(col==3&&old>=.5f)h=-h;GpuReal nn;bool valid=false;
       copyActive(cfg,probe,w);
       if(team.leader()){probe.state[node][col]=old+h;if((col==4||col==5)&&!pchipFdCompatible(cfg,w,probe,node,col)){probe.state[node][col]=old-h;if(pchipFdCompatible(cfg,w,probe,node,col))h=-h;}}
       h=team.broadcast(h);
       const GpuReal directionH=h;
       // Every lane invokes this callback: refresh and branch inspection contain
       // full-team collectives. All candidate decisions are lane-uniform.
       int direction=fdSelectDirection([&](int sign){
        GpuReal candidate=old+sign*directionH;
        if(sign<0&&!fdProbeAdmissible(in,col,candidate))return false;
        copyActive(cfg,probe,w);if(team.leader())probe.state[node][col]=candidate;
        team.sync();valid=refresh(in,cfg,probe,ctx,nn,trialOut);++evaluations;
        return valid&&sameBranches(cfg,w,probe);
       });
       if(direction)h=direction*directionH;
       else{
        h=directionH;valid=false;
       for(int attempt=0;attempt<2&&!valid;++attempt){copyActive(cfg,probe,w);if(team.leader())probe.state[node][col]=old+h;team.sync();valid=refresh(in,cfg,probe,ctx,nn,trialOut);++evaluations;if(!valid)h=col==3?h/2:-h;}
       }
       while(valid&&!sameBranches(cfg,w,probe)){
        GpuReal nextH=h/2;if(old+nextH==old||old+nextH==old+h){if(team.leader())++s.fdRoundoff;team.sync();break;}
        h=nextH;copyActive(cfg,probe,w);if(team.leader())probe.state[node][col]=old+h;team.sync();valid=refresh(in,cfg,probe,ctx,nn,trialOut);++evaluations;
       }
       if(!valid){ok=false;if(team.leader())++s.fdFailures;team.sync();break;}
       GpuReal actual=probe.state[node][col]-old;bool zero=col==4&&old!=0;
       for(int i=0;i<cfg.nodes;++i)if(probe.residual[i][4]!=res[i*6+4])zero=false;
       for(int i=team.rank();i<n;i+=team.size())plus[i]=probe.residual[i/6][i%6];
       team.sync();
       if(zero){
        if(team.leader())++s.centralDirections;
        h=std::cbrt(std::numeric_limits<GpuReal>::epsilon())*std::max(std::abs(old),std::sqrt(std::max(1.f,in.matching.k)));
        while(true){
         copyActive(cfg,probe,w);if(team.leader())probe.state[node][col]=old+h;team.sync();valid=refresh(in,cfg,probe,ctx,nn,trialOut);++evaluations;
         bool samePlus=sameBranches(cfg,w,probe);
         for(int i=team.rank();i<n;i+=team.size())plus[i]=probe.residual[i/6][i%6];
         team.sync();GpuReal hp=0;if(team.leader())hp=probe.state[node][col]-old;hp=team.broadcast(hp);
         copyActive(cfg,probe,w);if(team.leader())probe.state[node][col]=old-h;team.sync();valid=refresh(in,cfg,probe,ctx,nn,trialOut)&&valid;++evaluations;if(team.leader())actual=hp-(probe.state[node][col]-old);actual=team.broadcast(actual);
         if(!valid){ok=false;break;}if(samePlus&&sameBranches(cfg,w,probe))break;
         GpuReal nextH=h/2;if(old+nextH==old||old-nextH==old||old+nextH==old+h){if(team.leader())++s.fdRoundoff;team.sync();break;}h=nextH;
        }
        if(!ok)break;
       }
       if(team.leader())scale[j]=std::max(std::abs(old),vs);
       team.sync();
       for(int i=team.rank();i<n;i+=team.size())a[i*n+j]=(plus[i]-(zero?probe.residual[i/6][i%6]:res[i]))/actual*scale[j];
       team.sync();
      }
     }
    }
    bool factored=false;
    if(team.leader()){
     cachedScale=scale;cache.branches=branches;cache.dirichletMask=mask;collectNutBranches=false;cache.boundary.clear();
     for(int node:{0,cfg.nodes-1})for(int col=0;col<6;++col){if(node==0&&(col==3||(col==2&&in.adiabatic)||(col==5&&cfg.enableSst&&ctx.mass>0)))continue;int j=node*6+col;
      BoundaryTransform t;t.column=j;t.lo=std::max(0,j-17);t.hi=std::min(n,j+18);t.diagonal=a[j*n+j];for(int i=t.lo;i<t.hi;++i)t.values[i-t.lo]=a[i*n+j];cache.boundary.push_back(t);
      for(int i=0;i<n;++i)if(i!=j)a[i*n+j]=0;
      for(int k=0;k<n;++k)a[j*n+k]=0;
      a[j*n+j]=1;
     }
     factored=ok&&lu.factor(a,n);if(!factored)++s.factorFailures;
    }
    if(!team.broadcast(factored)){ok=false;break;}
   }else{if(team.leader())scale=cachedScale;team.sync();}
   if(team.leader()){
    for(const auto&t:cache.boundary){int j=t.column;GpuReal step=rhs[j]/t.diagonal;for(int i=t.lo;i<t.hi;++i)if(i!=j)rhs[i]-=t.values[i-t.lo]*step;rhs[j]=step;}lu.solve(rhs);
   }
   team.sync();GpuReal previousNorm=norm;GpuReal alpha=1,next=norm;bool accepted=false;
   for(int line=0;line<cfg.numerics.maximumLineSearchIterations;++line){
    copyActive(cfg,probe,w);bool feasible=true;
    for(int i=team.rank();i<n;i+=team.size())probe.state[i/6][i%6]=base[i]+alpha*rhs[i]*scale[i];
    team.sync();
    for(int i=team.rank();i<cfg.nodes;i+=team.size())feasible=detail::feasibleLayerComposition<GpuReal,2>(probe.state[i])&&feasible;
    feasible=team.all(feasible);bool valid=false;if(feasible){valid=refresh(in,cfg,probe,ctx,next,trialOut);++evaluations;}
    if(valid&&std::isfinite(next)&&next<norm){copyActive(cfg,w,probe);if(team.leader())out=trialOut;norm=next;accepted=true;team.sync();break;}alpha*=cfg.numerics.lineSearchReduction;
   }
   ++its;if(!accepted){if(fresh)break;if(team.leader())lu.valid=false;}else if(alpha<1||norm>previousNorm*cfg.numerics.jacobianRefreshReduction){if(team.leader())lu.valid=false;}team.sync();
  }
  if(team.leader()){
   cache.haveState=true;stats.iterations=seedIts+its;stats.residualEvaluations=evaluations;stats.factorizations=jacobianBuilds;stats.physicalResidual=norm;
   stats.seconds=0;
  }
  observerSlot=-1;
  if(team.broadcast(ok))ok=publishTeam(in,cfg,w,ctx,seedIts+its,norm,out);
  return team.broadcast(ok);
 }
};
// Keep scalar order/arithmetic exactly, without artificial collective overhead.
template<int M> struct Engine<M,detail::SerialWallTeam> : ScalarEngine<M> {
 UGKWP_GAS_HD explicit Engine(Scratch<M>&scratch,const detail::SerialWallTeam& = detail::SerialWallTeam{}):ScalarEngine<M>(scratch){}
};

}}} // ugkwp::gaswall::coherent
