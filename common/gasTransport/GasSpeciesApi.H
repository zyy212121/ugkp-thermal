#pragma once
#include <cstdint>
namespace ugkwpGpu {
static constexpr std::uint32_t gasModelApiVersion = 1;
static constexpr std::uint64_t maxGasModelTextBytes = 8ULL*1024ULL*1024ULL;
struct GasModelCapabilitiesV1
{
    std::uint32_t version;
    std::uint32_t compiledSpecies;
    std::uint32_t modeMask;
    std::uint32_t reserved;
};
struct GasModelConfigureArgsV1
{
    std::uint32_t version;
    std::uint32_t mode;
    std::uint32_t speciesCount;
    std::uint32_t reserved;
    std::uint64_t speciesOrderHash;
    std::uint64_t thermoHash;
    std::uint64_t mechanismHash;
    std::uint64_t modelBytes;
    std::uint64_t mechanismBytes;
};
struct GasSpeciesIdentityV1
{
    std::uint32_t version;
    std::uint32_t speciesCount;
    std::uint64_t speciesOrderHash;
    std::uint64_t thermoHash;
    std::uint64_t mechanismHash;
};
inline bool sameGasSpeciesIdentity(const GasSpeciesIdentityV1& a, const GasSpeciesIdentityV1& b)
{
    return a.version==b.version && a.speciesCount==b.speciesCount
        && a.speciesOrderHash==b.speciesOrderHash && a.thermoHash==b.thermoHash
        && a.mechanismHash==b.mechanismHash;
}

}
extern "C" int ugkwpGpuResidentStrictQueryGasModelCapabilitiesV1
(void* handle, ugkwpGpu::GasModelCapabilitiesV1* capabilities);
extern "C" int ugkwpGpuResidentStrictConfigureGasModelV1
(void* handle, const ugkwpGpu::GasModelConfigureArgsV1* configuration,
 const char* modelText, const char* mechanismText);
extern "C" int ugkwpGpuResidentStrictUploadSpeciesV1
(void* handle, const ugkwpGpu::GasSpeciesIdentityV1* identity, const double* speciesDensity);
extern "C" int ugkwpGpuResidentStrictDownloadSpeciesV1
(void* handle, const ugkwpGpu::GasSpeciesIdentityV1* identity, double* speciesDensity);
extern "C" int ugkwpGpuResidentStrictUploadSpeciesBoundaryV1
(void* handle, const ugkwpGpu::GasSpeciesIdentityV1* identity,
 const int* compositionFixed, const double* massFraction);
