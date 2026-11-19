#pragma once
#include "FrozenFoamFields.H"
#include "../gpu/ChtRestartSnapshot.H"
#include "../../../common/gpu/RadiationWarmStart.H"
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <future>
#include <unistd.h>

namespace Foam { namespace gpuThermal {
struct FrozenThermalOutput {
    FrozenFoamFiles fields;
    FrozenFoamFile manifest;
    std::string temporaryLedger, ledger;
};



class StagedThermalRestart {
    std::filesystem::path directory_;
public:
    explicit StagedThermalRestart(const std::string& destination) {
        auto pattern=destination+"/.cht-snapshot-XXXXXX";
        if(!::mkdtemp(pattern.data()))
            throw std::runtime_error("cannot stage thermal restart in "+destination+": "+std::strerror(errno));
        directory_=pattern;
    }
    StagedThermalRestart(const StagedThermalRestart&)=delete;
    ~StagedThermalRestart() {
        std::error_code ignored;
        std::filesystem::remove_all(directory_,ignored);
    }
    const std::filesystem::path& directory() const {return directory_;}
    void publish(bool withRadiation) const {
        const auto moveFile=[this](const char* name){
            std::filesystem::rename(directory_/name,directory_.parent_path()/name);
        };
        moveFile("gpuResidentStrictParticles.dat");
        moveFile("gpuResidentStrictEpsGPrev.dat");
        moveFile("gpuResidentStrictSourceResidual.dat");
        if(withRadiation)moveFile("domWarmStart.bin");
    }
};

inline std::future<void> writeFrozenThermalOutput(
    const std::string& directory,ChtRestartSnapshot& restart,ugkp::gpu::RadiationWarmStart* radiation,
    std::future<FrozenThermalOutput> fieldsReady)
{
    return std::async(std::launch::async,[directory,&restart,radiation,fieldsReady=std::move(fieldsReady)]() mutable {
        StagedThermalRestart staged(directory);
        restart.write(staged.directory().string());
        std::string error;
        if(radiation&&!radiation->write((staged.directory()/"domWarmStart.bin").string(),error))
            throw std::runtime_error(error);

        auto output=fieldsReady.get();
        for(const auto& field:output.fields)field.publish();
        staged.publish(radiation!=nullptr);
        if(::link(output.temporaryLedger.c_str(),output.ledger.c_str())||::unlink(output.temporaryLedger.c_str()))
            throw std::runtime_error("cannot commit wall energy ledger "+output.ledger+": "+std::strerror(errno));
        output.manifest.publish();
    });
}
}}
