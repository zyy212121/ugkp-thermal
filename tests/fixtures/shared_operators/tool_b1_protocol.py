from pathlib import Path
import subprocess,sys,json
root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
owner=(root/'common/GpuToolB1.cuh').read_text()
owner='\n'.join(x for x in owner.splitlines() if not x.startswith('#include') and not x.startswith('#pragma'))
body=r'''
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cassert>
#include <cstdlib>
using GpuTime=double;
struct GasHostPolicy {using Time=double;};
using cudaError_t=int;constexpr int cudaSuccess=0;
using cudaEvent_t=int*;
struct DeviceState {int fixedWorkBlockTuned=0,fixedCellBlockThreads=0,fixedFaceBlockThreads=0,hostTurbulenceModel=0;};
int failure=0,created=0,alive=0,recorded=0,launched=0,currentBlock=0,queries=0;
constexpr int toolB1WarmupRuns=1,toolB1MeasuredRuns=3;
int recoverGasPrimitivesKernel=1;
template<bool Turbulent> int computeGasInternalFaceFluxKernel=2;
void setLastError(const char*,int){}
void setLastErrorText(const char*){}
int cudaEventCreate(cudaEvent_t* p){if(failure==1 && created==1)return 7;*p=new int(++created);++alive;return 0;}
int cudaEventDestroy(cudaEvent_t p){assert(p);delete p;--alive;return 0;}
int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* p,int,int block,int){++queries;if(failure==2 && block==64)return 7;*p=(block==32?0:1);return 0;}
int cudaDeviceSynchronize(){return failure==3?7:0;}
int cudaEventRecord(cudaEvent_t){++recorded;return (failure==4 && recorded%2==1)||(failure==5 && recorded%2==0)?7:0;}
int cudaEventSynchronize(cudaEvent_t){return failure==6?7:0;}
int cudaEventElapsedTime(float* p,cudaEvent_t,cudaEvent_t){*p=float(currentBlock);return failure==7?7:0;}
int launchToolB1CellBundle(DeviceState*,int block){++launched;currentBlock=block;return failure==8?1:0;}
int launchToolB1FaceBundle(DeviceState*,int block,double,double){++launched;currentBlock=block;return failure==9?1:0;}
'''+owner+r'''
int main(int argc,char** argv){failure=std::atoi(argv[1]);DeviceState s;s.hostTurbulenceModel=std::atoi(argv[2]);
 int result=tuneFixedWorkBlockThreads(&s,.01,.2);
 if(alive!=0){std::fprintf(stderr,"events leaked %d\n",alive);return 20;}
 if(failure){if(result==0||s.fixedWorkBlockTuned){std::fprintf(stderr,"failure %d was accepted\n",failure);return 21;}}
 else {if(result||s.fixedCellBlockThreads!=64||s.fixedFaceBlockThreads!=64||!s.fixedWorkBlockTuned)return 22;
  int old=queries;if(tuneFixedWorkBlockThreads(&s,.01,.3)||queries!=old)return 23;}
 return 0;
}
'''
source=out/'tool_b1.cpp';source.write_text(body);exe=out/'tool_b1'
build=subprocess.run(['g++','-std=c++17','-O2',str(source),'-o',str(exe)],capture_output=True,text=True)
(out/'build.log').write_text(build.stdout+build.stderr)
assert build.returncode==0,build.stderr
rows=[]
for turbulence in [0,1]:
 for failure in range(10):
  run=subprocess.run([str(exe),str(failure),str(turbulence)],capture_output=True,text=True)
  rows.append(dict(failure=failure,turbulence=turbulence,exit=run.returncode,output=run.stdout+run.stderr))
(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
bad=[r for r in rows if r['exit']]
print('TOOL_B1_PROTOCOL',len(rows)-len(bad),'PASS',len(bad),'FAIL')
for row in bad:print(row)
raise SystemExit(bool(bad))
