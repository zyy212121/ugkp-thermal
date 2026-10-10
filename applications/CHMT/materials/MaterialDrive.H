#ifndef CHMT_MATERIALS_MATERIALDRIVE_H
#define CHMT_MATERIALS_MATERIALDRIVE_H
#include "coupling/IntervalContracts.H"
namespace chmt {
inline Real materialDriveRelative(Real a,Real b,Real floor){return absValue(a-b)/maxValue(floor,maxValue(absValue(a),absValue(b)));}
// A LOCAL first-order lag control, not a bound on global PDE error. Only
// interface records are scanned; no volume RHS or implicit solve occurs here.
inline bool materialDriveSlabEnd(const IntervalHistory& history,Real begin,Real requested,
    Real tolerance,Real tractionScale,Real& output,std::uint64_t& samples,std::string& error){
    if(!history.complete()||begin<history.interval().begin||requested>history.interval().end||requested<=begin
        ||!finite(tolerance)||tolerance<0||!finite(tractionScale)||tractionScale<=0){error="invalid material driving control";return false;}
    const GasIntervalRecord* held=nullptr;for(const auto& r:history.records())if(begin<r.end){held=&r;break;}
    if(!held){error="no driving sample at material slab start";return false;}Real end=requested;
    for(const auto& r:history.records()){if(r.begin<=begin||r.begin>=end)continue;
        if(r.gasTrace.size()!=held->gasTrace.size()||r.gasTraction.size()!=held->gasTraction.size()||r.gasGradient.size()!=held->gasGradient.size()||r.gasWallMatching.size()!=held->gasWallMatching.size()){
            error="gas drive layout changed";return false;}
        Real change=0;++samples;
        for(std::size_t f=0;f<r.gasTrace.size();++f){const auto& a=held->gasTrace[f];const auto& b=r.gasTrace[f];
            change=maxValue(change,materialDriveRelative(a.temperature,b.temperature,1));change=maxValue(change,materialDriveRelative(a.pressure,b.pressure,1));
            change=maxValue(change,materialDriveRelative(a.rho,b.rho,1e-30));
            change=maxValue(change,mag(a.velocity-b.velocity)/maxValue(1,maxValue(a.soundSpeed,b.soundSpeed)));
            for(int s=0;s<Ns;++s)change=maxValue(change,absValue(a.Y[s]-b.Y[s]));
        }
        // New-family material and film constitutive data are driven by this
        // actual matching snapshot, which may change while the owner is steady.
        for(std::size_t f=0;f<r.gasWallMatching.size();++f){const auto& a=held->gasWallMatching[f];const auto& b=r.gasWallMatching[f];
            change=maxValue(change,materialDriveRelative(a.pressure,b.pressure,1));
            change=maxValue(change,materialDriveRelative(a.state.temperature,b.state.temperature,1));
            const Real speedScale=f<r.gasTrace.size()?maxValue(1,maxValue(r.gasTrace[f].soundSpeed,held->gasTrace[f].soundSpeed)):1;
            Real velocityChange=0;for(int d=0;d<3;++d){const Real delta=a.state.velocity[d]-b.state.velocity[d];velocityChange+=delta*delta;}
            change=maxValue(change,::sqrt(velocityChange)/speedScale);
            for(int s=0;s<Ns;++s)change=maxValue(change,absValue(a.state.massFraction[s]-b.state.massFraction[s]));
            change=maxValue(change,materialDriveRelative(a.state.k,b.state.k,1e-12));
            change=maxValue(change,materialDriveRelative(a.state.omega,b.state.omega,1e-12));
        }
        for(std::size_t f=0;f<r.gasTraction.size();++f)change=maxValue(change,mag(r.gasTraction[f]-held->gasTraction[f])/maxValue(tractionScale,maxValue(mag(r.gasTraction[f]),mag(held->gasTraction[f]))));
        for(std::size_t f=0;f<r.gasGradient.size();++f){for(int j=0;j<3;++j)change=maxValue(change,mag(r.gasGradient[f].velocity[j]-held->gasGradient[f].velocity[j])/maxValue(1,maxValue(mag(r.gasGradient[f].velocity[j]),mag(held->gasGradient[f].velocity[j]))));}
        if(!finite(change)){error="nonfinite gas driving change";return false;}
        if(change>tolerance||r.radiationReceiver!=held->radiationReceiver){end=r.begin;break;}
    }
    if(!(end>begin)){error="driving slab does not advance";return false;}output=end;error.clear();return true;
}
}
#endif
