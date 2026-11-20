#pragma once
#include <memory>
#include <vector>
namespace chmt {
struct Backend;
struct HostState;
class AcceptedStateSnapshot {
    struct Storage;
    std::unique_ptr<Storage> data_;
public:
    AcceptedStateSnapshot();
    ~AcceptedStateSnapshot();

    void freeze(Backend&,const HostState&,bool captureRadiation);

    void collect(HostState&,std::vector<double>& radiationEnergy);
};
}
