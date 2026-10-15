"""Changed tooling/API boundaries, not solver admission or GPU performance tests."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess

ROOT = SOURCE_ROOT


def test_timer_samples_accumulate_api_float_values_in_double(tmp_path):
    source = (ROOT / 'common/GpuToolB1.cuh').read_text().split('int tuneFixedWorkBlockThreads', 1)[0]
    source = '\n'.join(line for line in source.splitlines() if not line.startswith(('#include', '#pragma')))
    body = r'''
#include <algorithm>
#include <vector>
#include <cassert>
using cudaEvent_t=int;using cudaError_t=int;constexpr int cudaSuccess=0;
constexpr int toolB1WarmupRuns=0,toolB1MeasuredRuns=3;
int sample=0;int cudaDeviceSynchronize(){return 0;}int cudaEventRecord(int){return 0;}
int cudaEventSynchronize(int){return 0;}
int cudaEventElapsedTime(float* p,int,int){const float values[3]={16777216.0f,1.0f,1.0f};*p=values[sample++];return 0;}
void setLastError(const char*,int){}
''' + source + r'''
int main(){double median=0,total=0;assert(measureToolB1Candidate([]{return 0;},0,0,median,total)==0);
assert(median==1.0);assert(total==16777218.0);assert(sample==3);}
'''
    cpp = tmp_path / 'timer.cpp'; cpp.write_text(body)
    exe = tmp_path / 'timer'
    build = subprocess.run(['g++', '-std=c++17', '-O2', str(cpp), '-o', str(exe)], capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    subprocess.run([str(exe)], check=True)


def test_native_probe_storage_is_double_with_cuda_float_temporaries():
    for app, leaf in [('CHT', 'gpu'), ('FSH', 'private_backend'), ('gasUGKP', 'private_backend')]:
        source = (ROOT / 'applications' / app / leaf / 'GpuResidentStrict.cu').read_text()
        assert 'double totalMs' in source and 'double stageMs[ProbeStageCount]' in source
    thermal = (ROOT / 'common/GpuDevelopmentThermalProbeSample.cuh').read_text()
    assert '&sample.totalMs' not in thermal
    assert 'sample.stageMs[stage] += static_cast<double>(elapsedMs)' in thermal


def test_gather_memset_byte_counts_match_actual_profile_owner():
    import re
    for owner in ('CHT', 'FSH'):
        text=(VALIDATION_ROOT / f'tests/fixtures/thermal_workers/gather_{owner}.cu.in').read_text()
        for field in ('pColdNodeSpecificEnthalpy','pCold2DNodeSpecificEnthalpy'):
            compact='compactP'+field[1:]
            scalar='float' if owner=='CHT' else 'double'
            assert re.search(r'\b'+scalar+r' \*'+field+r', \*'+compact+r';',text)
            line=next(line for line in text.splitlines() if 'cudaMemset' in line and 's->'+compact+',' in line)
            assert 'sizeof('+scalar+')' in line


def test_gather_actual_allocation_and_memset_expressions_stay_in_bounds(tmp_path):
    import re
    for owner in ('CHT', 'FSH'):
        text=(VALIDATION_ROOT / f'tests/fixtures/thermal_workers/gather_{owner}.cu.in').read_text()
        state=text[text.index('struct DeviceState {'):text.index('\n};')+3]
        allocations=re.findall(r'alloc\(s->\w+,[^;]*?\);',text)
        writes=re.findall(r'cudaMemset\(s->\w+,[^;]*?\);',text)
        assert len(writes)==32
        code=r'''
#include <unordered_map>
#include <cstring>
#include <cassert>
#include <cstddef>
using Real=double;
struct CsrReductionTask {int cell,begin,end,source;};
std::unordered_map<void*,size_t> bytes;
template<class T>void alloc(T*& p,int n){p=reinterpret_cast<T*>(new unsigned char[n*sizeof(T)]);bytes[p]=n*sizeof(T);}
void cudaMemset(void* p,int value,size_t n){assert(bytes.count(p)&&n<=bytes[p]);std::memset(p,value,n);}
'''+state+'\nint main(){const int capacity=25;DeviceState object{};auto* s=&object;\n'+'\n'.join(allocations+writes)+'\nfor(auto item:bytes)delete[] static_cast<unsigned char*>(item.first);}\n'
        cpp=tmp_path/(owner+'.cpp');exe=tmp_path/owner;cpp.write_text(code)
        subprocess.run(['g++','-std=c++17','-O0',str(cpp),'-o',str(exe)],check=True,capture_output=True,text=True)
        subprocess.run([str(exe)],check=True,capture_output=True,text=True)
