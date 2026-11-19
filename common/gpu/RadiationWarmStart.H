#pragma once
#include "RadiationPrecision.H"
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ugkp { namespace gpu {


class RadiationWarmStart {
    friend class DiscreteOrdinateRadiation;
    struct Storage;
    std::unique_ptr<Storage> storage_;
    std::array<std::uint64_t,7> header_{};
public:
    RadiationWarmStart();
    ~RadiationWarmStart();
    void describe(std::uint64_t cells,std::uint64_t directions,std::uint64_t layout,bool initialized);
    bool write(const std::string& path,std::string& error);
};

struct ReadWarmStart {
    bool initialized = false, metadataChanged = false;
    std::vector<double> values;
};
bool readWarmStartFile(const std::string& path, std::size_t cells, std::size_t directions,
                       std::uint64_t layout, ReadWarmStart& result, std::string& error);
}}
