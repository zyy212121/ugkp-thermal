#include "Ostream.H"
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <ostream>

namespace {
long long monotonicNanoseconds()
{
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

int boundary(const char* name)
{
    const char* value=std::getenv(name);
    if (value && std::atoi(value)>0) return std::atoi(value);
    std::abort();
}

void check(cudaError_t status)
{
    if (status!=cudaSuccess) {
        std::fprintf(stderr,"Progress sampling failed: %s\n",cudaGetErrorString(status));
        std::abort();
    }
}

std::FILE* openTimingFile()
{
    const char* path=std::getenv("UGKP_COMPLETED_PROGRESS");
    if (path) {
        if (auto* stream=std::fopen(path,"wx")) return stream;
    }
    std::fputs("Cannot create the completed progress timing file.\n",stderr);
    std::abort();
}

void sampleProgress()
{
    static std::FILE* samples=openTimingFile();
    static const int first=boundary("UGKP_PROGRESS_FIRST");
    static const int last=boundary("UGKP_PROGRESS_LAST");
    static const bool profile=std::getenv("UGKP_PROFILE_WINDOW")!=nullptr;
    static int index=0;
    ++index;
    const auto submitted=monotonicNanoseconds();
    long long completed=0;
    if (index==first || index==last) {
        check(cudaDeviceSynchronize());
        completed=monotonicNanoseconds();
        if (profile) check(index==first ? cudaProfilerStart() : cudaProfilerStop());
    }
    if (std::fprintf(samples,"%d,%lld,%lld\n",index,submitted,completed)<0) std::abort();
}
}

Foam::Ostream& Foam::operator<<(Foam::Ostream& stream,const char* text)
{
    using WriteFunction=Foam::Ostream& (*)(Foam::Ostream&,const char*);
    static const auto original=reinterpret_cast<WriteFunction>(dlsym(RTLD_NEXT,"_ZN4FoamlsERNS_7OstreamEPKc"));
    if (!original) std::abort();
    if (text && *text=='U' && std::strcmp(text,"UGKP_REPRO step=")==0) sampleProgress();
    return original(stream,text);
}

extern "C" std::ostream& insertProgress(std::ostream&,const char*,std::streamsize)
    asm("_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l");
extern "C" std::ostream& insertProgress(std::ostream& stream,const char* text,std::streamsize size)
{
    using Insert=std::ostream& (*)(std::ostream&,const char*,std::streamsize);
    static const auto original=reinterpret_cast<Insert>(dlsym(RTLD_NEXT,
        "_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l"));
    if (!original) std::abort();
    constexpr char prefix[]="CHMT_REPRO step=";
    if (size==sizeof(prefix)-1 && std::memcmp(text,prefix,sizeof(prefix)-1)==0) sampleProgress();
    return original(stream,text,size);
}
