#pragma once
// Included in mixtureThermalParticles after its state adapters. This common
// reduction changes only response evaluation, never parcel publication.
template<class State,class Team>
struct Response {
    const State& state;int cell;double dt,volume;bool csr,split;Team team;
    mutable double lower=0.,upper=0.,reference=0.,heat=0.,slope=0.,curvature=0.;
    __device__ ugkwp::ThermoStatus operator()(double T,double& q,double& derivative)const noexcept {
        if(T>lower&&T<upper){
            const double change=T-reference;
            q=heat+change*(slope+.5*curvature*change);
            derivative=slope+curvature*change;
            return ugkwp::gasFinite(q)&&ugkwp::gasFinite(derivative)?
                ugkwp::ThermoStatus::Success:ugkwp::ThermoStatus::NonFiniteInput;
        }
        q=0.;derivative=0.;double second=0.;
        double lo=-1.7976931348623157e308,hi=1.7976931348623157e308;
        const int n=mixtureParticles::count(state,cell,csr,split);
        for(int pos=team.rank();pos<n;pos+=team.width()) {
            const int i=mixtureParticles::parcel(state,cell,pos,csr,split);
            if(!member(state,i,cell))continue;
            const auto response=cachedResponse(state,i);
            double h,dh;response(T,h,dh);const auto interval=response.segment(T);
            q+=state.pm[i]*h/volume;derivative+=state.pm[i]*dh/volume;
            second+=state.pm[i]*interval.curvature/volume;
            lo=fmax(lo,interval.lower);hi=fmin(hi,interval.upper);
        }
        q=team.sum(q);derivative=team.sum(derivative);
        curvature=team.sum(second);lower=team.maximum(lo);upper=team.minimum(hi);
        // A reference on a phase/clipping cut carries the original one-sided
        // derivative. It must not seed the polynomial on the other side.
        if(!(T>lower&&T<upper)||!ugkwp::gasFinite(curvature))lower=upper=T;
        reference=T;heat=q;slope=derivative;
        return ugkwp::gasFinite(q)&&ugkwp::gasFinite(derivative)?
            ugkwp::ThermoStatus::Success:ugkwp::ThermoStatus::NonFiniteInput;
    }
};
