#pragma once
#include <memory>
#include <string>

namespace Foam { namespace gpuThermal {
class ChtRestartSnapshot {
    struct Storage;
    std::unique_ptr<Storage> storage_;
public:
    ChtRestartSnapshot();
    ~ChtRestartSnapshot();
    int freeze(void* owner);
    void* producerStream() const;
    void write(const std::string& directory);
};
}}
