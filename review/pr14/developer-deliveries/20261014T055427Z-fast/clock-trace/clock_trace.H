// Diagnostic-only host observer for the actual native CHMT CUDA backend.
// Compile and run the real solver; this file contains no replacement kernels.
#include "coupling/IntervalContracts.H"
#include <cstdio>
namespace chmt { namespace fp64_clock_trace {
inline void value(const CouplingInterval& w, std::uint64_t micro,
                  const char* event, std::size_t index, const char* field, double x) {
    std::fprintf(stderr, "F003,%llu,%llu,%llu,%s,%zu,%s,%a\n",
        static_cast<unsigned long long>(w.identity.sequence),
        static_cast<unsigned long long>(w.identity.epoch),
        static_cast<unsigned long long>(micro),event,index,field,x);
}
inline void begin(const CouplingInterval& w,std::uint64_t micro,double stamp,double dt) {
    value(w,micro,"begin",0,"window_begin",w.begin);
    value(w,micro,"begin",0,"window_end",w.end);
    value(w,micro,"begin",0,"micro_time",stamp);
    value(w,micro,"begin",0,"micro_dt",dt);
}
inline void wall(const CouplingInterval& w,std::uint64_t micro,double argumentStamp,
                 double sampledStamp,double dt,const WallKnot& wall) {
    value(w,micro,"stage",0,"argument_stamp",argumentStamp);
    value(w,micro,"stage",0,"sampled_stamp",sampledStamp);
    value(w,micro,"stage",0,"packet_dt",dt);
    for(std::size_t i=0;i<wall.faces.size();++i) {
        const auto& q=wall.faces[i];
        value(w,micro,"wall",i,"temperature",q.temperature);
        value(w,micro,"wall",i,"normal_velocity",q.normalVelocity);
        value(w,micro,"wall",i,"solid_normal_velocity",q.solidNormalVelocity);
        value(w,micro,"wall",i,"velocity_x",q.velocity.x);
        value(w,micro,"wall",i,"velocity_y",q.velocity.y);
        value(w,micro,"wall",i,"velocity_z",q.velocity.z);
        value(w,micro,"wall",i,"radiation_flux",q.radiationFlux);
        for(int s=0;s<Ns;++s) {
            const std::size_t item=i*Ns+s;
            value(w,micro,"wall_species",item,"species_rate",q.speciesRate[s]);
            value(w,micro,"wall_species",item,"pore_rate",q.poreRate[s]);
            value(w,micro,"wall_species",item,"pore_sweep_rate",q.poreSweepRate[s]);
        }
    }
}
inline void accepted(const CouplingInterval& w,const GasIntervalRecord& r,const HostState& state) {
    const auto micro=r.microSequence;
    value(w,micro,"accepted",0,"begin",r.begin);
    value(w,micro,"accepted",0,"end",r.end);
    value(w,micro,"accepted",0,"gas_trace_time",r.gasTraceTime);
    for(std::size_t c=0;c<state.gas.size();++c) {
        const auto& q=state.gas[c];
        value(w,micro,"gas",c,"mass",q.mass);
        value(w,micro,"gas",c,"energy",q.energy);
        value(w,micro,"gas",c,"momentum_x",q.momentum.x);
        value(w,micro,"gas",c,"momentum_y",q.momentum.y);
        value(w,micro,"gas",c,"momentum_z",q.momentum.z);
        for(int s=0;s<Ns;++s)value(w,micro,"gas_species",c*Ns+s,"mass",q.species[s]);
    }
    for(std::size_t i=0;i<r.packets.size();++i) {
        const auto& p=r.packets[i];
        value(w,micro,"packet",i,"face",double(p.face));
        value(w,micro,"packet",i,"stage",double(p.stage));
        value(w,micro,"packet",i,"kind",double(int(p.kind)));
        value(w,micro,"packet",i,"mass",p.mass);
        value(w,micro,"packet",i,"energy",p.energy);
        value(w,micro,"packet",i,"conductive",p.conductive);
        value(w,micro,"packet",i,"advective",p.advective);
        value(w,micro,"packet",i,"pressure_work",p.pressureWork);
        value(w,micro,"packet",i,"viscous_work",p.viscousWork);
        value(w,micro,"packet",i,"momentum_x",p.momentum.x);
        value(w,micro,"packet",i,"momentum_y",p.momentum.y);
        value(w,micro,"packet",i,"momentum_z",p.momentum.z);
    }
    std::fflush(stderr);
}
} }
