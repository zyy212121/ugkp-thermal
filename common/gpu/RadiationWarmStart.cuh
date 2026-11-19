#pragma once
#include "RadiationWarmStart.H"
#include "FrozenReadback.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>

namespace ugkp { namespace gpu {
struct RadiationWarmStart::Storage {
    FrozenReadback transfer;
    SnapshotRange<RadiationReal> range;
};
RadiationWarmStart::RadiationWarmStart() : storage_(new Storage) {}
RadiationWarmStart::~RadiationWarmStart() = default;
void RadiationWarmStart::describe(std::uint64_t cells, std::uint64_t directions,
                                 std::uint64_t layout, bool initialized) {
    header_ = {0x55474b50444f4d31ULL,2,sizeof(RadiationReal),cells,directions,layout,initialized};
}
bool RadiationWarmStart::write(const std::string& path, std::string& error) {
    const auto* host = storage_->transfer.collect();
    const std::string temporary = path + ".tmp";
    std::ofstream out(temporary,std::ios::binary|std::ios::trunc);
    out.write(reinterpret_cast<const char*>(header_.data()),sizeof(header_));
    if (header_[6] && storage_->range.count)
        out.write(reinterpret_cast<const char*>(storage_->range.at(host)),
                  static_cast<std::streamsize>(storage_->range.count*sizeof(RadiationReal)));
    out.close();
    if (!out || std::rename(temporary.c_str(),path.c_str()) != 0) {
        std::remove(temporary.c_str());
        error = "could not publish DOM warm start at " + path;
        return false;
    }
    error.clear();
    return true;
}

template<class Saved>
static bool readWarmValues(std::istream& in, std::vector<double>& values) {
    std::array<Saved,4096> chunk;
    for (std::size_t first=0;first<values.size();) {
        const auto count=std::min(chunk.size(),values.size()-first);
        if (!in.read(reinterpret_cast<char*>(chunk.data()),count*sizeof(Saved))) return false;
        for (std::size_t i=0;i<count;++i) {
            const double value=chunk[i];
            if (!std::isfinite(value) || std::abs(value)>std::numeric_limits<RadiationReal>::max()) return false;
            values[first+i]=value;
        }
        first+=count;
    }
    return true;
}

bool readWarmStartFile(const std::string& path, std::size_t cells, std::size_t directions,
                       std::uint64_t layout, ReadWarmStart& result, std::string& error) {
    std::ifstream in(path,std::ios::binary);
    std::array<std::uint64_t,7> header{};
    if (!in.read(reinterpret_cast<char*>(header.data()),sizeof(header)) ||
        header[0]!=0x55474b50444f4d31ULL || (header[2]!=4 && header[2]!=8) ||
        header[3]!=cells || header[4]!=directions || header[6]>1 ||
        (directions && cells>std::numeric_limits<std::size_t>::max()/directions)) {
        error="DOM warm start missing or cells, directions or field encoding cannot be read at " + path;
        return false;
    }
    ReadWarmStart candidate;
    candidate.initialized=bool(header[6]);
    candidate.metadataChanged=header[1]!=2 || header[2]!=sizeof(RadiationReal) || header[5]!=layout;
    if (candidate.initialized) candidate.values.resize(cells*directions);
    const bool read=header[2]==4 ? readWarmValues<float>(in,candidate.values) : readWarmValues<double>(in,candidate.values);
    if (!read || in.peek()!=std::char_traits<char>::eof() || in.bad()) {
        error="unusable DOM warm-start intensity payload at " + path;
        return false;
    }
    result=std::move(candidate);
    error.clear();
    return true;
}
}}
