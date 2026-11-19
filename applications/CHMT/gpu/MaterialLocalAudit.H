#pragma once
#include "core/Types.H"
#include "../../../common/CompensatedTime.H"
#include <cstddef>
namespace chmt {



struct MaterialLocalAudit {
    std::size_t record=0;
    ugkp::CompensatedTime time{},heldBegin{},heldEnd{};
    Real dt=0;
    bool ready=false;
};
}
