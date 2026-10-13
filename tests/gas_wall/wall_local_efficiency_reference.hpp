// Frozen test-only layerRow from d6110c02745a8ad726208a426c4bb3a28233c1fd.
namespace ugkwp { namespace gaswall { namespace detail {
template<class R,int N,int M,class TableReal> UGKWP_GAS_HD inline bool referenceLayerRow(const WallInput<R,N,TableReal>& in,
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
}}}
