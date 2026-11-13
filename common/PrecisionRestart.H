#pragma once
#include "PrecisionBuild.H"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

namespace ugkp { namespace precision {
inline std::string metadataPath(const std::string& directory)
{
    return (std::filesystem::path(directory) / "ugkpPrecision").string();
}

struct RestartPrecision
{
    int schema = 0;
    int computeBits = 0;
    int restartBits = 0;
    bool known = false;
};

inline RestartPrecision readRestartPrecision(const std::string& directory)
{
    std::ifstream input(metadataPath(directory));
    RestartPrecision result;
    std::string line;
    while (std::getline(input, line))
    {
        std::istringstream row(line);
        std::string key;
        if (!(row >> key)) continue;
        if (key == "schema") row >> result.schema;
        else if (key == "computePrecision") row >> result.computeBits;
        else if (key == "restartPrecision") row >> result.restartBits;
    }
    result.known = result.schema == 1
        && (result.computeBits == 32 || result.computeBits == 64)
        && (result.restartBits == 32 || result.restartBits == 64)
        && result.restartBits <= result.computeBits;
    return result;
}

inline bool checkRestartPrecision(const std::string& directory, std::string& error)
{
    const auto saved = readRestartPrecision(directory);
    if (!saved.known)
    {
        std::cerr << "Warning: input state precision is unknown at " << directory
                  << " (missing or unrecognized ugkpPrecision metadata); "
                  << "continuing with the existing reader at precision " << compiledBits
                  << ". Legacy wire widths do not prove the producer precision.\n";
    }
    else if (saved.restartBits < compiledBits)
    {
        error = "restart precision " + std::to_string(saved.restartBits)
            + " is lower than selected precision " + std::to_string(compiledBits)
            + " at " + directory + "; restart stopped.";
        return false;
    }
    else if (saved.restartBits > compiledBits)
    {
        std::cerr << "Warning: restart precision " << saved.restartBits
                  << " is higher than selected precision " << compiledBits
                  << " at " << directory << "; continuing at precision "
                  << compiledBits << ".\n";
    }
    error.clear();
    return true;
}

// Invalidate the old publication record before replacing any constituent
// restart file, so a partial write is never labelled by an earlier record.
inline void beginPrecisionWrite(const std::string& directory)
{
    const auto target = metadataPath(directory);
    if (std::remove(target.c_str()) != 0 && errno != ENOENT)
        std::cerr << "Warning: could not invalidate precision metadata " << target
                  << ": " << std::strerror(errno) << '\n';
}

// Called only after the complete solver state has been written successfully.
// File representation remains owned by its existing codecs. FP32 values
// widened into a binary64 or 17-digit wire still have effective precision 32.
inline void writeRestartPrecision(const std::string& directory)
{
    const auto target = metadataPath(directory);
    const auto temporary = target + ".tmp." + std::to_string(::getpid());
    std::ofstream output(temporary, std::ios::trunc);
    output << "schema 1\n"
           << "computePrecision " << compiledBits << '\n'
           << "restartPrecision " << compiledBits << '\n'
           << "storagePolicy " << policy << '\n'
           << "wireEncoding existingFieldSchemas\n";
    output.close();
    if (!output || std::rename(temporary.c_str(), target.c_str()) != 0)
    {
        std::cerr << "Warning: could not publish precision metadata; "
                  << "stale metadata may remain at " << target << ".\n";
        std::remove(temporary.c_str());
    }
}
}}
