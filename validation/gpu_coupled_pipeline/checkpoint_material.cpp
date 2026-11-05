// Offline inspection only: compile against the frozen source used by the run.
// The production codec supplies field layout/validation, not numerical answers.
#include "restart/Checkpoint.C"
#include <iostream>
#include <iomanip>
#include <cmath>

int main(int argc,char** argv) {
    try {
        if(argc!=2)throw std::runtime_error("usage: checkpoint_material CHECKPOINT_DIRECTORY");
        std::ifstream input(std::string(argv[1])+"/state.chmt",std::ios::binary|std::ios::ate);
        if(!input)throw std::runtime_error("cannot open checkpoint");
        const auto size=input.tellg();
        if(size<72)throw std::runtime_error("truncated checkpoint");
        chmt::Bytes bytes(static_cast<std::size_t>(size));input.seekg(0);
        input.read(reinterpret_cast<char*>(bytes.data()),size);
        if(!input)throw std::runtime_error("truncated checkpoint read");
        chmt::Decoder decoder(bytes);decoder.offset=8;
        const auto schema=decoder.integer(8);
        if(schema!=9)throw std::runtime_error("this inspector supports the frozen schema 9 only");
        decoder.offset=72;chmt::ModelConfig config;decoder(config);
        chmt::HostState state;std::string error;
        if(!chmt::readCheckpoint(argv[1],config,state,error))throw std::runtime_error(error);
        if(state.solid.size()!=state.solidMesh.volumes.size())throw std::runtime_error("material volume count mismatch");
        std::cout<<std::setprecision(17)<<"cell,volume,energy,porosity";
        for(int s=0;s<chmt::Nc;++s)std::cout<<",condensed_"<<s;
        for(int s=0;s<chmt::Nr;++s)std::cout<<",progress_"<<s;
        for(int s=0;s<chmt::Ns;++s)std::cout<<",pore_"<<s;
        std::cout<<'\n';
        auto emit=[](double x){if(!std::isfinite(x))throw std::runtime_error("nonfinite checkpoint material state");std::cout<<','<<x;};
        for(std::size_t i=0;i<state.solid.size();++i){
            const auto& q=state.solid[i];std::cout<<i;emit(state.solidMesh.volumes[i]);emit(q.energy);emit(q.porosity);
            for(double x:q.condensed)emit(x);
            for(double x:q.progress)emit(x);
            for(double x:q.pore)emit(x);
            std::cout<<'\n';
        }
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}