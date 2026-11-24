#pragma once
#include "GasStateView.H"
#include "../gasWall/WallModelTypes.H"
namespace ugkwp {



template<class State,class Model>
inline bool configureSingleGasWallThermo(const State& state,Model& model){
    using Real=decltype(model.config.relativeTolerance);
    if(!(gasFinite(state.Rgas)&&state.Rgas>0&&gasFinite(state.gasCp)&&state.gasCp>state.Rgas
      &&gasFinite(state.TgasMin)&&state.TgasMin>0))return false;
    model.singleThermo.molarMass=universalGasConstant<Real>()/Real(state.Rgas);
    model.singleThermo.minTemperature=Real(state.TgasMin);
    model.singleThermo.maxTemperature=std::numeric_limits<Real>::max();
    model.singleCoefficients[0]=Real(state.gasCp);
    return true;
}

template<bool Single=false,class State,class R,int N> UGKWP_GAS_HD inline bool recoverWallMatching(
 const State& s,const int* cells,const R* weights,int count,bool sst,
 gaswall::WallMatchingState<R,N>& out,R& pressure){
    using E=std::conditional_t<Single,R,typename std::common_type<double,GasEnergyScalar<State>>::type>;
    constexpr int M=N+6;
    E anchor[M]{},value[M]{};
    for(int j=0;j<count;++j){const int cell=cells[j];E q[M]{};
        const E density=s.rho[cell];
        if constexpr(Single) {
            static_assert(N==1);
            q[0]=density;
        } else {
            for(int a=0;a<N;++a)q[a]=s.gasSpecies.rho[a*s.nCells+cell];



            E fraction[N];
            if(!mixtureMassFractionsFromDensities<E,N>(q,density,E(s.gasSpecies.densityClosureTolerance),fraction,s.diagnostics))return false;
            E sum=0;int carrier=0;for(int a=0;a<N;++a){q[a]=density*fraction[a];sum+=q[a];if(q[a]>q[carrier])carrier=a;}
            if(!(sum>0))return false;
            if(q[carrier]+density-sum>=0)q[carrier]+=density-sum;
            else for(int a=0;a<N;++a)q[a]*=density/sum;
        }
        q[N]=s.rhoUx[cell];q[N+1]=s.rhoUy[cell];q[N+2]=s.rhoUz[cell];q[N+3]=s.rhoE[cell];
        if(sst){q[N+4]=s.rhoK[cell];q[N+5]=s.rhoOmega[cell];
            if(!(q[N+4]>=0&&q[N+5]>0&&density>0))return false;}
        for(int k=0;k<M;++k){
            if(!gasFinite(q[k]))return false;
            if(j==0)anchor[k]=q[k];
            value[k]+=E(weights[j])*(q[k]-anchor[k]);
        }
    }
    if(count<=0)return false;
    for(int k=0;k<M;++k)value[k]+=anchor[k];
    const auto towardZero=[](E value){
        if constexpr(std::is_same<E,float>::value)return ::nextafterf(value,0.0f);
        else return ::nextafter(value,0.0);
    };
    E alpha=1;
    for(int k=0;k<N;++k){if(anchor[k]<0)return false;
        if(value[k]<0)alpha=gasThermoDetail::minValue(alpha,anchor[k]/(anchor[k]-value[k]));}


    if(sst)for(int k=N+4;k<M;++k){if(anchor[k]<0)return false;
        if(value[k]<0)alpha=gasThermoDetail::minValue(alpha,anchor[k]/(anchor[k]-value[k]));}
    const auto controls=castThermoControls<E>(s.gasSpecies.thermoControls);
    E minimumTemperature,maximumTemperature;
    if constexpr(Single)minimumTemperature=E(s.TgasMin);
    else mixtureTemperatureBounds(s.gasSpecies.thermo,controls,minimumTemperature,maximumTemperature);




    E density0=0,ddensity=0,thermal0=anchor[N+3],dthermal=value[N+3]-anchor[N+3];
    for(int a=0;a<N;++a){const E delta=value[a]-anchor[a];
        E reference;
        if constexpr(Single)reference=minimumTemperature*(E(s.gasCp)-E(s.Rgas));
        else reference=speciesE(a,minimumTemperature,s.gasSpecies.thermo);
        if(!gasFinite(reference))return false;
        density0+=anchor[a];ddensity+=delta;thermal0-=anchor[a]*reference;dthermal-=delta*reference;}
    E aa=2*ddensity*dthermal,bb=2*(density0*dthermal+ddensity*thermal0),cc=2*density0*thermal0;
    for(int d=0;d<3;++d){const E v=anchor[N+d],dv=value[N+d]-v;aa-=dv*dv;bb-=2*v*dv;cc-=v*v;}
    if(!(density0>0&&cc>=0))return false;







    if((aa*alpha+bb)*alpha+cc<=0){
        E root;
        if(cc==0)root=bb>0&&aa<0?-bb/aa:E(0);
        else if(aa==0)root=-cc/bb;
        else {const E disc=::sqrt(bb*bb-4*aa*cc),q=-E(.5)*(bb+(bb>=0?disc:-disc));
            const E x=q/aa,y=cc/q;root=x>0?x:y;if(y>0&&y<root)root=y;}
        alpha=towardZero(root);
    }



    if(alpha<1)alpha=towardZero(alpha);
    else if(sst&&value[N+5]==0)alpha=towardZero(E(1));


    E rho=0;for(int k=0;k<M;++k)value[k]=::fma(alpha,value[k]-anchor[k],anchor[k]);
    for(int a=0;a<N;++a){if(value[a]<0&&alpha<1)value[a]=0;rho+=value[a];}
    if(!(rho>0))return false;
    E kinetic=0;for(int d=0;d<3;++d){out.velocity[d]=R(value[N+d]/rho);kinetic+=value[N+d]*value[N+d]/(2*rho);}
    E temperature=s.Tgas[cells[0]],energy=value[N+3]-kinetic;
    if constexpr(Single){
        temperature=energy/(rho*(E(s.gasCp)-E(s.Rgas)));
        pressure=R(rho*E(s.Rgas)*temperature);
        out.massFraction[0]=R(1);
    }else{
        bool available=false;
        if(invertMixtureEnergy(value,energy,s.gasSpecies.thermo,controls,temperature,&available)!=ThermoStatus::Success||!available)return false;
        E gasR=0;int carrier=0;E sum=0;
        for(int a=0;a<N;++a){out.massFraction[a]=R(value[a]/rho);sum+=out.massFraction[a];
            if(value[a]>value[carrier])carrier=a;
            gasR+=value[a]*universalGasConstant<E>()/s.gasSpecies.thermo.species[a].molarMass;}
        out.massFraction[carrier]+=R(1-sum);
        pressure=R(gasR*temperature);
    }
    out.temperature=R(temperature);
    if(sst){out.k=R(value[N+4]/rho);out.omega=R(value[N+5]/rho);}
    return gasFinite(pressure)&&gasFinite(temperature)&&temperature>0;
}
}
