#pragma once
#include <memory>
namespace chmt {
struct Backend;
struct HostState;
class AcceptedStateSnapshot {
    struct Storage;
    std::unique_ptr<Storage> data_;
public:
    AcceptedStateSnapshot();
    ~AcceptedStateSnapshot();

    void freeze(Backend&,const HostState&);

    void collect(HostState&);
};
}
