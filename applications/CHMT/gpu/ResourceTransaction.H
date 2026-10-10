#ifndef CHMT_GPU_RESOURCETRANSACTION_H
#define CHMT_GPU_RESOURCETRANSACTION_H
#include <memory>
#include <string>
namespace chmt {
// Resource must provide a noexcept fault-pointer rebind. All fallible allocation,
// transfer and validation happens under the candidate owner BEFORE this helper.
// No accepted status or inventory changes when candidate preparation failed.
template<class Resource,class Fault>
bool publishResourceReplacement(std::unique_ptr<Resource>& active0,std::unique_ptr<Resource>& active1,
    std::unique_ptr<Resource>& active2,std::unique_ptr<Resource>& candidate0,
    std::unique_ptr<Resource>& candidate1,std::unique_ptr<Resource>& candidate2,
    Fault& activeFault,const Fault& candidateFault,std::string& error){
    if(!candidateFault.message.empty()||!candidate0||!candidate1||!candidate2){
        error=candidateFault.message.empty()?"incomplete replacement resources":candidateFault.message;return false;
    }
    if(!activeFault.message.empty()){error=activeFault.message;return false;}
    candidate0->rebindFault(activeFault);candidate1->rebindFault(activeFault);candidate2->rebindFault(activeFault);
    active0.swap(candidate0);active1.swap(candidate1);active2.swap(candidate2);
    error.clear();return true;
}
}
#endif
