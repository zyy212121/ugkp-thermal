#pragma once
// Included in mixtureThermalParticles after its state adapters. This common
// reduction changes only response evaluation, never parcel publication.
template<class State,class Team>
struct Response {
    const State& state;int cell;GpuTime dt;float volume;bool csr,split;Team team;
    mutable float lower=0.f,upper=0.f,reference=0.f,heat=0.f,slope=0.f,curvature=0.f;
    __device__ ugkwp::ThermoStatus operator()(float T,float& q,float& derivative)const noexcept {
        if(T>lower&&T<upper){
            const float change=T-reference;
            q=heat+change*(slope+.5f*curvature*change);
            derivative=slope+curvature*change;
            return ugkwp::gasFinite(q)&&ugkwp::gasFinite(derivative)?
                ugkwp::ThermoStatus::Success:ugkwp::ThermoStatus::NonFiniteInput;
        }
        q=0.f;derivative=0.f;float second=0.f;
        float lo=-3.4028234663852886e38f,hi=3.4028234663852886e38f;
        const int n=mixtureParticles::count(state,cell,csr,split);
        for(int pos=team.rank();pos<n;pos+=team.width()) {
            const int i=mixtureParticles::parcel(state,cell,pos,csr,split);
            if(!member(state,i,cell))continue;
            const auto response=cachedResponse(state,i);
            float h,dh;response(T,h,dh);const auto interval=response.segment(T);
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
