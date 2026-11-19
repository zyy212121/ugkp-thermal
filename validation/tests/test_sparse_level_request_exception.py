
#!/usr/bin/env python3

from pathlib import Path
import argparse, subprocess, re, json
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parents[2]
results=[]
for bits,source in [(64,root/'common/gpu/SparseConjugateGradient.cu'),(32,root/'precision/fp32/common/gpu/SparseConjugateGradient.cu')]:
 text=source.read_text()
 begin=text.index('   auto l=std::unique_ptr<Level>(new Level);')
 end=text.index('if(n<=32)break;',begin)
 body=text[begin:end]
 assert 'requestGuard.release();' in body
 assert body.index('requestGuard(')<body.index('new(l->request)')
 assert body.index('requestGuard.release();')>body.index('levels.push_back(std::move(l));')
 constants=sorted(set(re.findall(r'vcRow[A-Za-z]+',body)))
 fields=sorted(set(re.findall(r'l->([A-Za-z]+)\.(?:reserve|bind|ptr)',body)))
 pre=r'''
#include <memory>
#include <vector>
#include <set>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <iostream>
#include <cstring>
static int failAt=0,events=0,allocations=0,frees=0;
static std::set<void*> live;
static bool fail(){return ++events==failAt;}
static void checked(int code){if(code)throw std::runtime_error("injected CUDA error");}
enum {cudaHostAllocMapped=1,cudaMemcpyHostToDevice=2};
struct VcAllocationRequest {int state=0,error=0;size_t poolBytes=0;};
struct VcStorage {};
static int cudaHostAlloc(void** p,size_t bytes,int){if(fail())return 1;*p=std::malloc(bytes);if(!*p)throw std::bad_alloc();live.insert(*p);++allocations;return 0;}
static int cudaHostGetDevicePointer(void** p,void* h,int){if(fail())return 1;*p=h;return 0;}
static int cudaFreeHost(void* p) noexcept {if(live.erase(p)!=1)std::abort();++frees;std::free(p);return 0;}
static int cudaMemset(void*,int,size_t){return fail()?1:0;}
static int cudaMemcpy(void*,const void*,size_t,int){return fail()?1:0;}
struct Field {void* ptr=nullptr;void reserve(int){if(fail())throw std::bad_alloc();}void bind(void*,int){if(fail())throw std::bad_alloc();}};
'''
 pre+='enum {'+','.join(constants)+'};'+chr(10)
 pre+='struct Level {int n=0,nc=0;VcAllocationRequest *request=nullptr,*deviceRequest=nullptr;Field '+','.join(fields)+';};'+chr(10)
 pre+=r'''
struct Levels {
 std::vector<std::unique_ptr<Level>> values;
 void push_back(std::unique_ptr<Level>&& p){if(fail())throw std::bad_alloc();values.push_back(std::move(p));}
 void release(){for(auto& p:values)if(p->request)cudaFreeHost(p->request);values.clear();}
};
static void construct(Levels& levels){int n=64;
'''
 post=r'''
}
int main(){
 int total=0;
 for(int point=0;;++point){
  failAt=point;events=0;allocations=0;frees=0;Levels levels;bool threw=false;
  try{construct(levels);}catch(const std::exception&){threw=true;}
  if(point==0){total=events;if(threw||levels.values.size()!=1||live.size()!=1||frees!=0)return 10;}
  else if(!threw||!levels.values.empty()||!live.empty()||allocations!=frees)return 11;
  levels.release();if(!live.empty()||allocations!=frees)return 12;
  if(point==total)break;
 }
 std::cout<<"PASS injected_failure_points="<<total<<" success_ownership_transfer=1 no_double_free=1"<<std::endl;
}
'''
 cpp=args.output/f'level-request-{bits}.cpp';binary=args.output/f'level-request-{bits}'
 cpp.write_text(pre+body+post)
 command=['g++','-std=c++17','-Wall','-Wextra','-O0',str(cpp),'-o',str(binary)]
 built=subprocess.run(command,text=True,capture_output=True)
 (args.output/f'build-{bits}.log').write_text(built.stdout+built.stderr)
 if built.returncode:raise RuntimeError('host harness build failed')
 run=subprocess.run([str(binary)],text=True,capture_output=True)
 (args.output/f'run-{bits}.log').write_text(run.stdout+run.stderr)
 results.append(dict(precision=bits,exit=run.returncode,stdout=run.stdout,source=str(source),testScope='extracted construction slice, stubbed CUDA/Field/vector failure; no GPU'))
 if run.returncode:raise RuntimeError('host harness failed')
(args.output/'results.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
