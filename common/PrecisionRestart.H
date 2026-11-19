#pragma once
#include "PrecisionBuild.H"
#include "DiagnosticsHost.H"
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
        diagnostics::hostWarning(diagnostics::Code::RestartMetadata,
            {36001,__FILE__,__func__,__LINE__,"unknown producer precision at "+directory
                +"; field layouts select decoding; continuing at selected precision"},0,compiledBits);
    }
    else if (saved.restartBits != compiledBits)
    {
        diagnostics::hostWarning(diagnostics::Code::RestartMetadata,
            {36001,__FILE__,__func__,__LINE__,"precision conversion at "+directory
                +"; conversion does not recover previously lost precision"},saved.restartBits,compiledBits);
    }
    error.clear();
    return true;
}



inline void beginPrecisionWrite(const std::string& directory)
{
    const auto target = metadataPath(directory);
    if (std::remove(target.c_str()) != 0 && errno != ENOENT)
        diagnostics::hostWarning(diagnostics::Code::RestartMetadata,
            {36002,__FILE__,__func__,__LINE__,"could not invalidate "+target+": "+std::strerror(errno)});
}




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
        diagnostics::hostWarning(diagnostics::Code::RestartMetadata,
            {36002,__FILE__,__func__,__LINE__,"could not publish precision metadata at "+target});
        std::remove(temporary.c_str());
    }
}
}}
