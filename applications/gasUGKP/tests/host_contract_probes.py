from functools import lru_cache
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]


def _run(source, headers=None):
    with tempfile.TemporaryDirectory(prefix="gas-host-contract-") as directory:
        build = Path(directory)
        for name, contents in (headers or {}).items():
            (build / name).write_text(contents)
        cpp = build / "probe.cpp"
        cpp.write_text(source)
        binary = build / "probe"
        result = subprocess.run([
            "g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
            "-I" + str(build), "-I" + str(ROOT), "-I" + str(ROOT / "common"),
            str(cpp), "-o", str(binary),
        ], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        result = subprocess.run([str(binary)], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout


@lru_cache(maxsize=1)
def create_payload_size():
    output = _run(r'''
#include "applications/gasUGKP/gpu/GpuBackendProtocol.H"
#include <cassert>
#include <iostream>
int main(){
 using namespace ugkwpGpuIpc;
 CreateArgs input{};input.gammaGas=3.125;input.rngSeed=0x0102030405060708ULL;
 input.dragModel=7;input.dragParameter0=-3.5;input.gasRobustFallback=9;
 unsigned char bytes[WireCodec<CreateArgs>::size];WireCodec<CreateArgs>::encode(bytes,input);
 const unsigned char seed[]={8,7,6,5,4,3,2,1};
 const unsigned char gamma[]={0,0,0,0,0,0,9,64};
 for(int i=0;i<8;++i){assert(bytes[32+i]==seed[i]);assert(bytes[40+i]==gamma[i]);}
 CreateArgs output{};WireCodec<CreateArgs>::decode(bytes,output);
 assert(output.gammaGas==3.125&&output.rngSeed==input.rngSeed);
 assert(output.dragModel==7&&output.dragParameter0==-3.5&&output.gasRobustFallback==9);
 std::cout<<WireCodec<CreateArgs>::size<<"\n";
}
''')
    return int(output)


@lru_cache(maxsize=1)
def scheduling_rows():
    foam = r'''
#pragma once
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
namespace Foam {
using label=int;using scalar=double;using word=std::string;
struct dictionary {
 std::map<std::string,std::string> values;
 bool found(const char* name)const{return values.count(name)!=0;}
 const std::string& lookup(const char* name)const{return values.at(name);}
 template<class T>T lookupOrDefault(const char* name,T fallback)const{
  auto found=values.find(name);if(found==values.end())return fallback;
  std::istringstream stream(found->second);T value{};stream>>std::boolalpha>>value;
  if(!stream)throw std::runtime_error("invalid fixture value");
  return value;
 }
};
struct FatalStream {template<class T>FatalStream& operator<<(const T&){return *this;}};
inline FatalStream fatalStream;constexpr int FatalError=1;
inline int exit(int){throw std::runtime_error("fatal fixture input");}
}
#define FatalErrorInFunction Foam::fatalStream
'''
    output = _run(r'''
#include "common/GpuSchedulingConfiguration.H"
#include <cassert>
#include <iostream>
int main(){
 for(const char* level:{"L0","L1","L2","auto","future"}){
  Foam::dictionary properties;properties.values["gpuCsrLevel"]=level;
  properties.values["gpuCsrCellLocalPath"]="true";
  properties.values["gpuCsrSplitPreDirectory"]="true";
  const auto value=Foam::readGpuSchedulingConfiguration(properties);
  std::cout<<int(value.csrLevel)<<" "<<value.csrCellLocalPath<<" "<<int(value.heavyReductionMode)
   <<" "<<value.warpAggregatedBinning<<" "<<value.splitPreDirectory
   <<" "<<value.particleBlockThreads<<" "<<value.reductionBlockThreads<<"\n";
 }
}
''', {"fvCFD.H": foam})
    return [tuple(map(int, line.split())) for line in output.splitlines()]
