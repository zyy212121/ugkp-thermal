#pragma once
#include "GasStateView.H"
#include "../gasWall/WallModelTypes.H"
namespace ugkwp {
// Deconvolve conservative FV averages, then recover the point thermodynamics.
// A single analytic convex scaling enforces the simplex and positive thermal
// energy. It does not evaluate candidates or change any macro inventory.
template<class State,class R,int N> UGKWP_GAS_HD inline bool recoverWallMatching(
 const State& s,const int* cells,const R* weights,int count,bool sst,
 gaswall::WallMatchingState<R,N>& out,R& pressure){
    using E=typename std::common_type<double,GasEnergyScalar<State>>::type;
    constexpr int M=N+6; // species, momentum, energy, rho*k, rho*omega
    E anchor[M]{},value[M]{},lower[2]={std::numeric_limits<E>::max(),std::numeric_limits<E>::max()},upper[2]{};
    for(int j=0;j<count;++j){const int cell=cells[j];E q[M]{};
        for(int a=0;a<N;++a)q[a]=s.gasSpecies.rho[a*s.nCells+cell];
        // Physical finite fractions use the common clipping/warning rule.
        // Close the temporary matching composition on its largest carrier;
        // absent species remain absent. Macro species inventories are untouched.
        E density=s.rho[cell],fraction[N];
        if(!mixtureMassFractionsFromDensities<E,N>(q,density,E(s.gasSpecies.densityClosureTolerance),fraction,s.diagnostics))return false;
        E sum=0;int carrier=0;for(int a=0;a<N;++a){q[a]=density*fraction[a];sum+=q[a];if(q[a]>q[carrier])carrier=a;}
        if(!(sum>0))return false;
        if(q[carrier]+density-sum>=0)q[carrier]+=density-sum;
        else for(int a=0;a<N;++a)q[a]*=density/sum;
        q[N]=s.rhoUx[cell];q[N+1]=s.rhoUy[cell];q[N+2]=s.rhoUz[cell];q[N+3]=s.rhoE[cell];
        if(sst){q[N+4]=s.rhoK[cell];q[N+5]=s.rhoOmega[cell];
            if(!(q[N+4]>=0&&q[N+5]>0&&density>0))return false;
            for(int k=0;k<2;++k){const E v=q[N+4+k]/density;
                lower[k]=gasThermoDetail::minValue(lower[k],v);upper[k]=gasThermoDetail::maxValue(upper[k],v);}}
        for(int k=0;k<M;++k){if(!gasFinite(q[k]))return false;value[k]+=E(weights[j])*q[k];if(j==0)anchor[k]=q[k];}
    }
    if(count<=0)return false;
    E alpha=1;
    for(int k=0;k<N;++k){if(anchor[k]<0)return false;
        if(value[k]<0)alpha=gasThermoDetail::minValue(alpha,anchor[k]/(anchor[k]-value[k]));}
    if(sst)for(int k=N+4;k<M;++k){if(anchor[k]<0)return false;
        if(value[k]<=0)alpha=gasThermoDetail::minValue(alpha,anchor[k]/(anchor[k]-value[k]));}
    E density0=0,ddensity=0,thermal0=anchor[N+3],dthermal=value[N+3]-anchor[N+3];
    for(int a=0;a<N;++a){const E delta=value[a]-anchor[a];
        const E reference=speciesE(a,E(0),s.gasSpecies.thermo);
        if(!gasFinite(reference))return false;
        density0+=anchor[a];ddensity+=delta;thermal0-=anchor[a]*reference;dthermal-=delta*reference;}
    E aa=2*ddensity*dthermal,bb=2*(density0*dthermal+ddensity*thermal0),cc=2*density0*thermal0;
    for(int d=0;d<3;++d){const E v=anchor[N+d],dv=value[N+d]-v;aa-=dv*dv;bb-=2*v*dv;cc-=v*v;}
    if(!(density0>0&&cc>0))return false;
    // Parameter-free local bounds for transported turbulence ratios. A
    // high-order overshoot is limited to the donor data range, rather than
    // generating a nearly zero matching k/omega and a singular edge Jacobian.
    // These are four linear constraints in conservative variables; all fields
    // share the same scale, and no mathematical or physical trial is repeated.
    if(sst)for(int k=0;k<2;++k){
        const E q0=anchor[N+4+k],q1=value[N+4+k];
        const E lo0=q0-lower[k]*density0,lo1=q1-lower[k]*(density0+ddensity);
        const E hi0=upper[k]*density0-q0,hi1=upper[k]*(density0+ddensity)-q1;
        if(lo1<0)alpha=gasThermoDetail::minValue(alpha,gasThermoDetail::maxValue(E(0),lo0)/(lo0-lo1));
        if(hi1<0)alpha=gasThermoDetail::minValue(alpha,gasThermoDetail::maxValue(E(0),hi0)/(hi0-hi1));
    }

    // The admissible thermal-energy cone is convex. Its first boundary on
    // this segment is the positive root, obtained once with stable algebra.
    if((aa*alpha+bb)*alpha+cc<=0){
        E root;
        if(aa==0)root=-cc/bb;
        else {const E disc=::sqrt(bb*bb-4*aa*cc),q=-E(.5)*(bb+(bb>=0?disc:-disc));
            const E x=q/aa,y=cc/q;root=x>0?x:y;if(y>0&&y<root)root=y;}
        alpha=::nextafter(root,E(0));
    }
    // An active cone/simplex boundary is approached from the interior once.
    // FMA avoids a second rounding of anchor + alpha*delta that can otherwise
    // make a mathematically nonnegative k or positive omega slightly negative.
    if(alpha<1)alpha=::nextafter(alpha,E(0));
    else if(sst&&value[N+5]==0)alpha=::nextafter(E(1),E(0));
    // Keep exact zeros (including absent species); no independent clipping or
    // renormalisation can make residual and accepted matching states differ.
    E rho=0;for(int k=0;k<M;++k)value[k]=::fma(alpha,value[k]-anchor[k],anchor[k]);
    for(int a=0;a<N;++a){if(value[a]<0&&alpha<1)value[a]=0;rho+=value[a];}
    if(!(rho>0))return false;
    E kinetic=0;for(int d=0;d<3;++d){out.velocity[d]=R(value[N+d]/rho);kinetic+=value[N+d]*value[N+d]/(2*rho);}
    E temperature=s.Tgas[cells[0]],energy=value[N+3]-kinetic;
    const auto controls=castThermoControls<E>(s.gasSpecies.thermoControls);
    bool available=false;
    if(invertMixtureEnergy(value,energy,s.gasSpecies.thermo,controls,temperature,&available)!=ThermoStatus::Success||!available)return false;
    E gasR=0;int carrier=0;E sum=0;
    for(int a=0;a<N;++a){out.massFraction[a]=R(value[a]/rho);sum+=out.massFraction[a];
        if(value[a]>value[carrier])carrier=a;gasR+=value[a]*universalGasConstant<E>()/s.gasSpecies.thermo.species[a].molarMass;}
    out.massFraction[carrier]+=R(1-sum);
    out.temperature=R(temperature);pressure=R(gasR*temperature);
    if(sst){out.k=R(value[N+4]/rho);out.omega=R(value[N+5]/rho);}
    return gasFinite(pressure)&&gasFinite(temperature)&&temperature>0;
}
}
