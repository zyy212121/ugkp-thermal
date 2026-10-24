// Offline prescribed-temperature benchmark, not a production material configuration.
#include "materials/dry/BulkResponse.H"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
using namespace chmt::dry;
int main(int argc,char**argv) {
    const double dt=argc>1?std::atof(argv[1]):.125;
    const double volume=argc>2?std::atof(argv[2]):1.;
    const bool endpointTemperature=argc>3&&std::atoi(argv[3]);
    const double cap[3]={30,22.8,7.2}, A[3]={12000,4.97777e8,4.97777e8};
    const double E[3]={71130.89,169975,169975}, onset[3]={333.3,555.6,555.6};
    constexpr double Rp=8.314471469, Rc=8.31446261815324;
    const double yields[3][7]={{.69,.02,.29,0,0,0,0},{0,0,0,.09,.33,.58,0},{0,0,0,0,0,0,1}};
    MaterialState<5,7> q; q.condensedMass={{30*volume,22.8*volume,7.2*volume,160*volume,60*volume}};
    std::array<CondensedThermo,5> solids{};
    for(auto& s:solids) {s.density=1200;} solids[3].density=1600;
    std::vector<Reaction<5,7>> base(3);
    for(int j=0;j<3;++j) {
        auto& r=base[j];r.preExponential=A[j]/(cap[j]*cap[j]);
        r.activationEnergy=E[j]*Rc/Rp;r.condensedOrder[j]=3;r.production.condensed[j]=-1;
        for(int s=0;s<7;++s)r.production.gas[s]=yields[j][s];
    }
    auto source=[&](double t) {
        auto active=base; const double T=300+3.1*t;
        for(int j=0;j<3;++j)if(T<onset[j])active[j].preExponential=0;
        return reactionSource(q,active,volume,solids,T);
    };
    auto density=[&](){return std::accumulate(q.condensedMass.begin(),q.condensedMass.end(),0.)/volume;};
    double minFraction=1,maxMassError=0,t=0,previousDensity=280;
    std::cout<<std::setprecision(17)<<"time_s,temperature_K,rho_s,pi_instant,pi_1s_average,volatile1,volatile2,volatile3,H2O,A1,A1OH,CO2,CO,CH4,H2\n";
    for(int output=0;output<=351;++output) {
        while(t<output-1e-12) {
            double next=std::min(t+dt,double(output));
            // Split known material onset discontinuities; no production gate.
            if(!endpointTemperature)for(double threshold:onset) {
                const double event=(threshold-300)/3.1;
                if(event>t+1e-12&&event<next)next=event;
            }
            const auto src=source(endpointTemperature?next:t);
            minFraction=std::min(minFraction,reactionEuler(q,src,next-t));t=next;
            const double total=density()+std::accumulate(q.poreMass.begin(),q.poreMass.end(),0.)/volume;
            maxMassError=std::max(maxMassError,std::abs(total-280));
        }
        const auto src=source(t); const double rho=density();
        const double pi=std::accumulate(src.poreMass.begin(),src.poreMass.end(),0.)/volume;
        std::cout<<t<<','<<300+3.1*t<<','<<rho<<','<<pi<<','<<(output?previousDensity-rho:0);
        for(int j=0;j<3;++j)std::cout<<','<<q.condensedMass[j]/volume;
        for(double m:q.poreMass)std::cout<<','<<m/volume;
        std::cout<<'\n'; previousDensity=rho;
    }
    std::cerr<<std::setprecision(17)<<"dt="<<dt<<" volume="<<volume<<" endpointTemperature="<<endpointTemperature
        <<" min_extent_fraction="<<minFraction<<" max_mass_residual_kg_m3="<<maxMassError<<'\n';
}
