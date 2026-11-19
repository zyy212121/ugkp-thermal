#pragma once
#include "ExecutionGraph.H"
#include "AdvanceExtension.H"
#include <vector>
#include "ThermalEventState.H"
#include "DiscreteOrdinateRadiation.H"
#include "ParticleRadiationMaterial.H"
#include "SolidThermalProperties.H"
#include <functional>
#include <future>
#include <memory>
#include <cstdint>
namespace ugkp { namespace gpu {
ExecutionGraphNode appendThermalOperation(ExecutionGraph,ExecutionGraphNode,void*,const std::function<void()>&);
using ThermalEventAppender=std::function<ExecutionGraphNode
    (ExecutionGraph,ExecutionGraphNode,void*)>;
struct ThermalEventBindings {
    DiscreteOrdinateRadiation* radiation=nullptr;
    ParticleRadiationMaterial* material=nullptr;
    SolidThermalIterationGraph* solid=nullptr;
    SolidContactIterationGraph* contact=nullptr;
    diagnostics::DeviceContext* diagnostics=nullptr;


    ThermalEventAppender beforeRadiation,prepareInterface,publishParticles,commit;
    std::vector<void*> preparedInputStreams;
};
class ThermalEventGraph : public AdvanceExtension {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    explicit ThermalEventGraph(const ThermalEventBindings&);
    ~ThermalEventGraph();
    ThermalEventGraph(const ThermalEventGraph&)=delete;
    ThermalEventGraph& operator=(const ThermalEventGraph&)=delete;
    ThermalEventState* deviceState()const noexcept;
    void setEvent(const ThermalEventState&);
    void bindRadiation(DiscreteOrdinateRadiation&,ParticleRadiationMaterial&);
    void orderPreparedInputs(void* stream) override;
    ExecutionGraphNode appendTo(ExecutionGraph,ExecutionGraphNode,
        AdvanceState*,const AdvanceRequest*,void* stream) override;
    ExecutionGraphNode appendTo(ExecutionGraph,ExecutionGraphNode dependency=nullptr);
    bool servicePendingCapacity();
    void acceptExternalCompletion(const std::shared_future<void>&);
    void acceptExternalExecution(void* producerStream);
    void awaitCompletion()const;
    ThermalEventState collect()const;
};
}}
