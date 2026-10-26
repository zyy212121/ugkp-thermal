#pragma once
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <string>

// Statistics windows restart afresh; physical sampling time and the frozen
// external boundary clock are independent of OpenFOAM's output-folder labels.
struct SteadyRestartState
{
    double boundaryTime=0;
    double samplingTime=0;
    std::uint64_t sampleCount=0;
    double samplingDeltaT=0;
};
inline bool readSteadyRestartState(const std::string& path,SteadyRestartState& state)
{
    std::ifstream input(path);if(!input)return false;
    SteadyRestartState candidate;
    std::string key;
    while(input>>key)
    {
        if(key=="boundaryTime")input>>candidate.boundaryTime;
        else if(key=="samplingTime")input>>candidate.samplingTime;
        else if(key=="sampleCount")input>>candidate.sampleCount;
        else if(key=="samplingDeltaT")input>>candidate.samplingDeltaT;
        else {std::string rest;std::getline(input,rest);}
        if(!input)return false;
    }
    state=candidate;return true;
}
inline bool writeSteadyRestartState(const std::string& path,const SteadyRestartState& state)
{
    std::ofstream output(path);output<<std::setprecision(17)
        <<"boundaryTime "<<state.boundaryTime<<'\n'
        <<"samplingTime "<<state.samplingTime<<'\n'
        <<"sampleCount "<<state.sampleCount<<'\n'
        <<"samplingDeltaT "<<state.samplingDeltaT<<'\n';
    return bool(output);
}
