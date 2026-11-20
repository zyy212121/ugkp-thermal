#pragma once
#include "io/CoupledOutput.H"
#include "io/CoupledRadiation.H"
#include "gpu/AcceptedStateSnapshot.H"
#include "gpu/Backend.H"
#include "restart/Checkpoint.H"
#include "../../../common/DiagnosticsCatalogue.H"
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <optional>
#include <fcntl.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/fs.h>

namespace chmt { namespace io {
struct OutputSelection { bool summary=false,fields=false,checkpoint=false; };

class AcceptedOutputWriter {
    const std::string directory_;
    const std::optional<std::uint64_t> resumedStep_;
    ModelConfig model_;
    ugkp::diagnostics::HostSink sink_;
    AcceptedStateSnapshot fieldTransfer_;
    ugkp::gpu::RadiationWarmStart warm_;

    std::future<ugkp::diagnostics::Snapshot> pending_;

    template<class Write>
    static void csv(const std::string& path,bool replace,Write write){
        std::string staging=path+".pending.XXXXXX";
        if(!::mkdtemp(staging.data()))throw std::runtime_error("cannot stage "+path+": "+std::strerror(errno));
        const auto payload=staging+"/data";
        std::ofstream file(payload,std::ios::trunc);
        if (!file) throw std::runtime_error("cannot open "+payload);
        std::string reason;
        if (!write(file,reason)) throw std::runtime_error(path+": "+reason);
        file.close();
        if (!file) throw std::runtime_error("cannot finish writing "+path);
        if(::syscall(SYS_renameat2,AT_FDCWD,payload.c_str(),AT_FDCWD,path.c_str(),replace?0:RENAME_NOREPLACE))
            throw std::runtime_error("cannot publish "+path+": "+std::strerror(errno));
        std::filesystem::remove(staging);
    }

    static void syncPath(const std::string& path) {
        const int fd=::open(path.c_str(),O_RDONLY);
        if (fd<0) throw std::runtime_error("open for sync "+path+": "+std::strerror(errno));
        const int result=::fsync(fd);
        const int saved=errno;
        const int closed=::close(fd);
        if (result || closed) throw std::runtime_error("sync "+path+": "+std::strerror(result?saved:errno));
    }
    void checkpoint(const std::string& suffix,const HostState& state,const ModelConfig& model,bool radiation,bool replace) {
        const auto path=directory_+"/checkpoint-"+suffix;


        namespace fs=std::filesystem;
        auto staging=path+".pending.XXXXXX";
        if (!::mkdtemp(staging.data())) throw std::runtime_error("cannot stage checkpoint at "+path);


        const bool replacing=replace&&fs::is_directory(fs::symlink_status(path));
        if(replacing){
            for(const auto& item:fs::directory_iterator(path)){
                const auto name=item.path().filename();
                if(name=="state.chmt"||name=="ugkpPrecision"||name=="domWarmStart.bin")continue;
                fs::copy(item.path(),fs::path(staging)/name,fs::copy_options::recursive|
                    fs::copy_options::create_hard_links|fs::copy_options::copy_symlinks);
            }
        }
        std::string error;
        if (!writeCheckpoint(staging,model,state,error)) throw std::runtime_error(error);
        if (radiation) {
            const auto cache=staging+"/domWarmStart.bin";
            if (!warm_.write(cache,error)) throw std::runtime_error(error);
            syncPath(cache);
        }
        const auto precision=staging+"/ugkpPrecision";
        if (std::filesystem::exists(precision)) syncPath(precision);
        syncPath(staging);
        const auto operation=replacing?RENAME_EXCHANGE:RENAME_NOREPLACE;
        if(::syscall(SYS_renameat2,AT_FDCWD,staging.c_str(),AT_FDCWD,path.c_str(),operation))
            throw std::runtime_error("cannot publish "+path+": "+std::strerror(errno));
        syncPath(directory_);
        if(replacing)fs::remove_all(staging);
    }

    void fields(const std::string& suffix,const HostState& state,const ModelConfig& model,bool replace) {
        const auto path=[&](const char* name){return directory_+"/"+name+"-"+suffix+".csv";};
        csv(path("gas"),replace,[&](auto& stream,auto& e){return writeAcceptedGasFields(stream,state,model,e);});
        if (model.physics.enableParticles)
            csv(path("particles"),replace,[&](auto& stream,auto& e){return writeAcceptedParticleFields(stream,state,e);});
        csv(path("material"),replace,[&](auto& stream,auto& e){return writeAcceptedMaterialFields(stream,state,model,e);});
        csv(path("exchange"),replace,[&](auto& stream,auto& e){return writeAcceptedExchangeFields(stream,state,model,e);});
        if (model.physics.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer)
            csv(path("wall-layer"),replace,[&](auto& stream,auto& e){return writeAcceptedWallLayerFields(stream,state,model,e);});
    }

    template<class Write>
    ugkp::diagnostics::Snapshot withReport(const AcceptedSummary& accepted,Write write) {
        using namespace ugkp::diagnostics;
        DeviceContext context{};
        reset(context,accepted.sequence,0,accepted.step,accepted.time);
        std::vector<SourceDescriptor> captured;
        ScopedHostWarningCapture warnings(context,captured);
        try {
            write(context);
        } catch (const RecordedFailure& failure) {
            if (!hasFatal(&context)) {
                captured.push_back(failure.source);
                recordFatal(&context,failure.record);
            }
        } catch (const std::exception& failure) {
            if (!hasFatal(&context)) {
                captured.push_back({42001,__FILE__,"AcceptedOutputWriter",__LINE__,directory_+": "+failure.what()});
                recordFatal(&context,makeRecord(Code::OutputData,42001,42));
            }
        }
        auto report=takeSnapshot(context,sourceCatalogue(),"ugkp-diagnostics-v1");
        report.sources.insert(report.sources.end(),captured.begin(),captured.end());
        return report;
    }
public:
    AcceptedOutputWriter(std::string directory,std::optional<std::uint64_t> resumedStep,const ModelConfig& model,ugkp::diagnostics::HostSink sink)
        :directory_(std::move(directory)),resumedStep_(resumedStep),model_(model),sink_(sink) {
        model_.physics.diagnostics=nullptr;
        model_.physics.gasTransportCoefficients=molecularTransportView(model_.physics);
    }
    void collectReady() {
        if (pending_.valid() && pending_.wait_for(std::chrono::seconds(0))==std::future_status::ready) finish();
    }
    ugkp::diagnostics::Snapshot collect() {
        return pending_.valid() ? pending_.get() : ugkp::diagnostics::Snapshot{};
    }
    void finish() {
        ugkp::diagnostics::dispatch(collect(),sink_);
    }
    void submit(const OutputSelection& selected,Backend& backend,const HostState& state,
        CoupledRadiation* radiation,double radiationInterval) {
        collectReady();
        if (!selected.summary && !selected.fields && !selected.checkpoint) return;
        finish();
        const auto summary=acceptedSummary(state);
        const bool saveWarm=selected.checkpoint && radiation;
        const bool captureRadiation=selected.fields && radiationInterval>0;
        std::unique_ptr<HostState> copy;
        if (selected.fields || selected.checkpoint) {
            auto report=withReport(summary,[&](auto&) {
                copy=std::make_unique<HostState>(state);
                fieldTransfer_.freeze(backend,state,captureRadiation);
                if (saveWarm) radiation->freezeWarmStart(warm_,gasWindowDeviceView(backend).producerStream);
            });
            ugkp::diagnostics::dispatch(report,sink_);
        }
        pending_=std::async(std::launch::async,[this,summary,selected,saveWarm,radiationInterval,copy=std::move(copy)]() mutable {
            return withReport(summary,[&](ugkp::diagnostics::DeviceContext& context) {
                auto model=model_;
                model.physics.diagnostics=&context;
                model.physics.gasTransportCoefficients=molecularTransportView(model.physics);
                const auto suffix=std::to_string(summary.step);
                const bool replace=resumedStep_&&*resumedStep_==summary.step;
                if (selected.summary)
                    csv(directory_+"/accepted-"+suffix+".csv",replace,[&](auto& out,auto& e){return writeAcceptedCoupledSummary(out,summary,model,e);});
                std::vector<double> wallRadiation;
                if (copy) fieldTransfer_.collect(*copy,wallRadiation);
                if (selected.fields) fields(suffix,*copy,model,replace);
                if (!wallRadiation.empty())
                    csv(directory_+"/wall-radiation-"+suffix+".csv",replace,[&](auto& out,auto& e){
                        return writeAcceptedWallRadiationFields(out,*copy,radiationInterval,wallRadiation,e);
                    });
                if (selected.checkpoint && !ugkp::diagnostics::hasFatal(&context))
                    checkpoint(suffix,*copy,model,saveWarm,replace);
            });
        });
    }
};
}}
