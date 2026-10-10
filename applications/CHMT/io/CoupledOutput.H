#ifndef CHMT_IO_COUPLEDOUTPUT_H
#define CHMT_IO_COUPLEDOUTPUT_H
#include "core/HostState.H"
#include "configuration/CouplingEntry.H"
#include "coupling/ExchangeLedger.H"
#include "materials/Thermodynamics.H"
#include <ostream>
#include <iomanip>
namespace chmt { namespace io {
// Caller supplies the accepted synchronized macro endpoint. No rejected gas
// trial or partial CPU candidate may acquire output/restart authority.
inline bool validateAcceptedOutput(const HostState& state,
    const ModelConfig& model,std::string& error) {
    if(!validateCouplingEntry("Multirate",state,error))return false;
    if(!finite(state.time)||state.time<0||state.commitSequence!=state.acceptedSteps){
        error="output requires a synchronized accepted macro endpoint";return false;}
    for(const auto& packet:state.ledger)if(!validatePacketMath(packet,model.physics.tolerances)
        ||packet.consumerMask!=requiredConsumers(packet.kind)){
        error="output refuses pending or invalid coupling exchange";return false;}
    for(const auto& update:state.filmStorage)if(!update.consumed){error="output refuses pending film storage";return false;}
    error.clear();return true;
}
inline bool writeAcceptedCoupledSummary(std::ostream& stream,const HostState& state,
    const ModelConfig& model,std::string& error) {
    if(!validateAcceptedOutput(state,model,error))return false;
    stream<<std::setprecision(17)<<"time,accepted_steps,gas_cells,material_cells,interface_faces,gas_mode,thermo_hash,commit_sequence\n"
        <<state.time<<','<<state.acceptedSteps<<','<<state.gas.size()<<','<<state.solid.size()<<','
        <<state.surface.area.size()<<','<<static_cast<int>(model.physics.gasMode)<<','<<model.physics.gasThermoFingerprint<<','<<state.commitSequence<<'\n';
    if(!stream){error="coupled summary write failed";return false;}
    error.clear();return true;
}
inline bool writeAcceptedGasFields(std::ostream& stream,const HostState& state,
    const ModelConfig& model,std::string& error){
    if(!validateAcceptedOutput(state,model,error))return false;
    std::vector<GasPrimitive> primitive(state.gas.size());
    for(std::size_t c=0;c<state.gas.size();++c)if(!recoverGas(state.gas[c],state.gasMesh.volumes[c],model.physics,primitive[c])){
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
inline bool writeAcceptedMaterialFields(std::ostream& stream,const HostState& state,
    const ModelConfig& model,std::string& error){
    if(!validateAcceptedOutput(state,model,error))return false;
    std::vector<Real> temperature(state.solid.size());
    for(std::size_t c=0;c<state.solid.size();++c)if(!finite(state.solidMesh.volumes[c])||state.solidMesh.volumes[c]<=0
        ||!recoverSolid(state.solid[c],model.physics,temperature[c])){error="invalid accepted material output";return false;}
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
inline bool writeAcceptedExchangeFields(std::ostream& stream,const HostState& state,
    const ModelConfig& model,std::string& error){
    if(!validateAcceptedOutput(state,model,error))return false;
    stream<<std::setprecision(17)<<"packet,step,geometry,face,stage,kind,gas_cell,solid_cell,mass,energy,conductive,advective,pressure_work,viscous_work,radiation,liquid_kinetic_advection,consumer_mask\n";
    for(std::size_t i=0;i<state.ledger.size();++i){const auto& q=state.ledger[i];
        stream<<i<<','<<q.step<<','<<q.geometry<<','<<q.face<<','<<q.stage<<','<<static_cast<int>(q.kind)<<','
            <<q.gasCell<<','<<q.solidCell<<','<<q.mass<<','<<q.energy<<','<<q.conductive<<','<<q.advective<<','
            <<q.pressureWork<<','<<q.viscousWork<<','<<q.radiation<<','<<q.liquidKineticAdvection<<','<<q.consumerMask<<'\n';
    }
    if(!stream){error="exchange field output write failed";return false;}error.clear();return true;
}
}}
#endif
