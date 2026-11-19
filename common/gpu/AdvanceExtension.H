#pragma once
#include "AdvanceState.H"
#include "ExecutionGraph.H"
#include <future>
#include <memory>
namespace ugkp { namespace gpu {


class AdvanceExtension {
public:
    virtual ~AdvanceExtension()=default;
    virtual ExecutionGraphNode appendTo(ExecutionGraph,ExecutionGraphNode,
        AdvanceState*,const AdvanceRequest*,void* stream)=0;
    virtual void orderPreparedInputs(void* stream)=0;
    virtual bool servicePendingCapacity()=0;
    virtual void acceptExternalCompletion(const std::shared_future<void>&)=0;
    virtual void acceptExternalExecution(void* stream)=0;
};
int bindAdvanceExtension(void* handle,std::shared_ptr<AdvanceExtension>);
}}
