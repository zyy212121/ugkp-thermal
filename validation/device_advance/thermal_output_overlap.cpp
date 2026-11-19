#include "thermal/FrozenThermalOutput.H"
#include <chrono>
#include <iostream>
#include <thread>

namespace fs=std::filesystem;
using namespace Foam::gpuThermal;

static void require(bool condition,const char* message){
    if(!condition)throw std::runtime_error(message);
}
static bool staged(const fs::path& dir){
    for(const auto& entry:fs::directory_iterator(dir)){
        if(entry.is_directory()&&entry.path().filename().string().find(".cht-snapshot-")==0
           &&fs::is_regular_file(entry.path()/"gpuResidentStrictSourceResidual.dat"))return true;
    }
    return false;
}
static void exercise(const fs::path& dir,bool commit){
    require(fs::create_directory(dir),"test output directory already exists");
    ChtRestartSnapshot restart;
    std::future<void> writer;
    {
        std::promise<FrozenThermalOutput> fields;
        writer=writeFrozenThermalOutput(dir.string(),restart,nullptr,fields.get_future());
        const auto limit=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        while(!staged(dir)&&std::chrono::steady_clock::now()<limit){
            if(writer.wait_for(std::chrono::milliseconds(2))==std::future_status::ready)writer.get();
        }
        require(staged(dir),"writer did not stage the native restart while ordinary fields were unavailable");
        require(!fs::exists(dir/"gpuResidentStrictParticles.dat")&&!fs::exists(dir/"thermalExchangeState"),
            "partial output was published early");
        if(commit){
            FrozenThermalOutput output;
            output.fields.push_back({(dir/"T").string(),"temperature",false});
            output.manifest={(dir/"thermalExchangeState").string(),"accepted",false};
            output.temporaryLedger=(dir/"pending-ledger").string();
            output.ledger=(dir/"ledger").string();
            std::ofstream(output.temporaryLedger)<<"energy";
            fields.set_value(std::move(output));
        }
    }
    if(commit){
        writer.get();
        const auto manifest=fs::last_write_time(dir/"thermalExchangeState");
        for(const char* field:{"T","ledger","gpuResidentStrictParticles.dat","gpuResidentStrictSourceResidual.dat","gpuResidentStrictEpsGPrev.dat"})
            require(fs::is_regular_file(dir/field)&&fs::last_write_time(dir/field)<=manifest,"incomplete accepted snapshot");
        require(!staged(dir)&&!fs::exists(dir/"pending-ledger"),"staging files survived a commit");
    }else{
        try{writer.get();throw std::runtime_error("cancelled output reported success");}
        catch(const std::future_error& e){require(e.code()==std::make_error_code(std::future_errc::broken_promise),"wrong producer error");}
        require(fs::is_empty(dir),"cancelled staging changed the published endpoint");
    }
}
int main(int argc,char** argv){
    if(argc!=2)return 2;
    exercise(fs::path(argv[1])/"commit",true);
    exercise(fs::path(argv[1])/"cancel",false);
    std::cout<<"PASS native staging overlaps ordinary field production, commit order and cancellation\n";
}
