#ifndef CHMT_IO_COUPLEDOUTPUT_H
#define CHMT_IO_COUPLEDOUTPUT_H
#include "core/WallHeatIntegral.H"
#include "core/HostState.H"
#include "configuration/CouplingEntry.H"
#include "coupling/ExchangeLedger.H"
#include "materials/Thermodynamics.H"
#include <ostream>
#include <iomanip>
namespace chmt { namespace io {



struct AcceptedSummary {
    Real time;
    std::uint64_t step,sequence;
    std::size_t cells,solidCells,faces;
};
inline AcceptedSummary acceptedSummary(const HostState& state) {
    return {state.time,state.acceptedSteps,state.commitSequence,state.gas.size(),state.solid.size(),state.surface.area.size()};
}
inline bool writeAcceptedCoupledSummary(std::ostream& stream,const AcceptedSummary& state,
    const ModelConfig& model,std::string& error) {
    stream<<std::setprecision(17)<<"time,accepted_steps,gas_cells,material_cells,interface_faces,gas_mode,thermo_hash,commit_sequence,sst_chemistry_integral_audit_collection\n"
        <<state.time<<','<<state.step<<','<<state.cells<<','<<state.solidCells<<','
        <<state.faces<<','<<static_cast<int>(model.physics.gasMode)<<','<<model.physics.gasThermoFingerprint<<','<<state.sequence<<",NOT_COLLECTED_LEGACY_CHECKPOINT_SLOTS_ONLY\n";
    if(!stream){error="coupled summary write failed";return false;}
    error.clear();return true;
}
inline bool writeAcceptedCoupledSummary(std::ostream& stream,const HostState& state,
    const ModelConfig& model,std::string& error) {
    return writeAcceptedCoupledSummary(stream,acceptedSummary(state),model,error);
}
inline bool writeAcceptedGasFields(std::ostream& stream,const HostState& state,
    const ModelConfig& model,std::string& error){
    std::vector<GasPrimitive> primitive(state.gas.size());
    for(std::size_t c=0;c<state.gas.size();++c)if(!recoverGas(state.gas[c],gasPhaseVolume(state,c),model.physics,primitive[c],nullptr,static_cast<int>(c))){
        error="invalid accepted gas output";return false;}
    stream<<std::setprecision(17)<<"cell,volume,mass,momentum_x,momentum_y,momentum_z,total_energy,temperature,pressure";
    for(const auto& name:model.speciesNames){stream<<','<<std::quoted("mass_"+name);}
    stream<<'\n';
    for(std::size_t c=0;c<state.gas.size();++c){const auto&q=state.gas[c];const auto&w=primitive[c];
        stream<<c<<','<<state.gasMesh.volumes[c]<<','<<q.mass<<','<<q.momentum.x<<','<<q.momentum.y<<','<<q.momentum.z<<','<<q.energy<<','<<w.temperature<<','<<w.pressure;
        for(int s=0;s<Ns;++s){stream<<','<<q.species[s];}
        stream<<'\n';
    }
    if(!stream){error="gas field output write failed";return false;}error.clear();return true;
}
inline bool writeAcceptedParticleFields(std::ostream& stream,const HostState& state,std::string& error){
    stream<<std::setprecision(17)<<"id,status,state,cell,face,x,y,z,ux,uy,uz,mass,diameter,temperature,thermal_energy,theta,thermal_carry,contact_age,contact_duration,contact_area,contact_mechanical_energy\n";
    const auto carry=[](const auto& thermal){
        if constexpr(std::is_same_v<PhysicalReal,float>)return 0.0;
        else return thermal.specificEnthalpyResidual;
    };
    for(const auto& p:state.particles){
        stream<<p.id<<','<<p.thermal.status<<','<<p.state<<','<<p.cell<<','<<p.face<<','
            <<p.position.x<<','<<p.position.y<<','<<p.position.z<<','<<p.velocity.x<<','<<p.velocity.y<<','<<p.velocity.z<<','
            <<p.mass<<','<<p.diameter<<','<<p.thermal.temperature<<','<<p.energy<<','<<p.thermal.theta<<','<<carry(p.thermal)<<','
            <<p.contactAge<<','<<p.contactDuration<<','<<p.contactArea<<','<<p.contactMechanicalEnergy<<'\n';
    }
    if(!stream){error="particle field output write failed";return false;}error.clear();return true;
}
inline bool writeAcceptedMaterialFields(std::ostream& stream,const HostState& state,
    const ModelConfig& model,std::string& error){
    std::vector<PhysicalReal> temperature(state.solid.size());
    for(std::size_t c=0;c<state.solid.size();++c)if(!recoverSolid(state.solid[c],model.physics,temperature[c])){error="invalid accepted material output";return false;}
    stream<<std::setprecision(17)<<"cell,volume,total_energy,temperature,porosity";
    for(const auto& name:model.condensedNames)stream<<','<<std::quoted("mass_"+name);
    for(const auto& name:model.speciesNames)stream<<','<<std::quoted("mass_pore_"+name);
    stream<<'\n';
    for(std::size_t c=0;c<state.solid.size();++c){const auto& q=state.solid[c];
        stream<<c<<','<<state.solidMesh.volumes[c]<<','<<q.energy<<','<<temperature[c]<<','<<q.porosity;
        for(int s=0;s<Nc;++s)stream<<','<<q.condensed[s];
        for(int s=0;s<Ns;++s)stream<<','<<q.pore[s];
        stream<<'\n';
    }
    if(!stream){error="material field output write failed";return false;}error.clear();return true;
}
inline bool writeAcceptedWallHeatFields(std::ostream& stream,const HostState& state,double interval,
    const std::vector<WallHeatIntegral>& heat,std::string& error){
    if(!std::isfinite(interval)||interval<=0||heat.size()!=state.surface.gasFace.size()
        ||heat.size()!=state.surface.persistentId.size()){
        error="accepted wall heat interval or faces are incomplete";return false;
    }
    stream<<std::setprecision(17)<<"time,interval_s,gas_face,face,convection_J,gas_transport_J,particle_contact_J\n";
    for(std::size_t f=0;f<heat.size();++f){
        const auto& h=heat[f];
        if(!finite(h.convection)||!finite(h.gasTransport)||!finite(h.particleContact)){
            error="nonfinite accepted wall heat";return false;
        }
        stream<<state.time<<','<<interval<<','<<state.surface.gasFace[f]<<','<<state.surface.persistentId[f]
            <<','<<h.convection<<','<<h.gasTransport<<','<<h.particleContact<<'\n';
    }
    if(!stream){error="wall heat field output write failed";return false;}
    error.clear();return true;
}
inline bool writeAcceptedWallRadiationFields(std::ostream& stream,const HostState& state,
    double interval,const std::vector<double>& energy,std::string& error){
    if(energy.size()!=state.surface.gasFace.size()||energy.size()!=state.surface.persistentId.size()){
        error="accepted radiation face data is incomplete";return false;
    }
    stream<<std::setprecision(17)<<"radiation_time,interval_s,gas_face,face,energy_J\n";
    for(std::size_t f=0;f<energy.size();++f){
        if(!std::isfinite(energy[f])){error="nonfinite accepted wall radiation energy";return false;}
        stream<<state.radiationTime<<','<<interval<<','<<state.surface.gasFace[f]<<','
            <<state.surface.persistentId[f]<<','<<energy[f]<<'\n';
    }
    if(!stream){error="wall radiation field output write failed";return false;}
    error.clear();return true;
}
}}
#endif
