#ifndef CHMT_GPU_COURANTREDUCTIONSTORAGE_CUH
#define CHMT_GPU_COURANTREDUCTIONSTORAGE_CUH
#include "gpu/Buffer.H"
#include <cstddef>
#include <stdexcept>

namespace chmt {




class CoupledCourantStorage {
    struct Block {Buffer<char> bytes;bool inUse=false;};
    CudaFault fault_;
    std::vector<Block> blocks_;
public:
    using value_type=char;
    CoupledCourantStorage()=default;
    CoupledCourantStorage(const CoupledCourantStorage&)=delete;
    CoupledCourantStorage& operator=(const CoupledCourantStorage&)=delete;
    CoupledCourantStorage(CoupledCourantStorage&&)=delete;
    CoupledCourantStorage& operator=(CoupledCourantStorage&&)=delete;

    char* allocate(std::ptrdiff_t count){
        const auto bytes=static_cast<std::size_t>(count);
        Block* idle=nullptr;
        for(auto& block:blocks_){
            if(block.inUse)continue;
            if(block.bytes.size()>=bytes){block.inUse=true;return block.bytes.data();}
            if(!idle||block.bytes.size()>idle->bytes.size())idle=&block;
        }
        if(!idle){blocks_.emplace_back();idle=&blocks_.back();}
        fault_.message.clear();fault_.runtimeFailure=false;
        if(!idle->bytes.resize(bytes,fault_))throw std::runtime_error(fault_.message);
        idle->inUse=true;
        return idle->bytes.data();
    }
    void deallocate(char* pointer,std::size_t){
        for(auto& block:blocks_)if(block.bytes.data()==pointer){block.inUse=false;return;}
    }
};
}
#endif
