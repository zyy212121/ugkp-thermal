"""Execute the exact residual IO body with float native scalar and double ABI.

This isolates the production reader/transfer precision; it is not a complete
OpenFOAM SP application build or GPU execution.
"""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import pytest
ROOT=SOURCE_ROOT


def definition(text,start):
    begin=text.index(start);brace=text.index('{',begin);depth=1;end=brace+1
    while depth:
        if text[end]=='{':depth+=1
        elif text[end]=='}':depth-=1
        end+=1
    return text[begin:end]


@pytest.fixture(scope='module')
def residual_probe(tmp_path_factory):
    header=(ROOT/'applications/gasUGKP/gpu/GpuResidentStrict.H').read_text()
    reader=definition(header,'bool readSourceResidualRestartMirrorIfPresent(')
    finite=definition(header,'static bool finiteMirrorScalar(')
    root=tmp_path_factory.mktemp('residual_precision');source=root/'probe.cpp';binary=root/'probe'
    source.write_text(r'''
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using scalar=float;using label=int;using string=std::string;using fileName=std::string;
struct Time{std::string path;};
template<class T>struct List:std::vector<T>{explicit List(int n):std::vector<T>(n){} T* begin(){return this->data();}};
template<class T>using Field=List<T>;
template<class To,class From>To gpuHostValue(From x){return static_cast<To>(x);}
bool isFile(const fileName& f){return bool(std::ifstream(f));}
int fail(int){throw std::runtime_error("actual input error");}
#define FatalError 1
#define FatalErrorInFunction std::cerr
#define exit fail
static std::vector<double> received;
int ugkwpGpuResidentStrictUploadSourceResidualMass(void*,int count,const int* faces,const double* values){
 if(count!=2||faces[0]!=0||faces[1]!=1)return 1;
 received.assign(values,values+count);return 0;
}
struct Reader{
 void* handle_=nullptr;
 fileName sourceResidualRestartFile(const Time& t){return t.path;}
 void check(int code,const char*){if(code)throw std::runtime_error("actual ABI error");}
'''+finite+'\n'+reader+r'''
};
int main(int argc,char**argv){
 if(argc!=2)return 4;
 try{Reader reader;if(!reader.readSourceResidualRestartMirrorIfPresent(Time{argv[1]}))return 5;
 if(received.size()!=2||received[0]!=1e100||received[1]!=-1e100)return 6;
 std::cout<<"COMPLETE_FP64_RESIDUAL "<<received[0]<<" "<<received[1]<<"\n";
 }catch(const std::exception& e){std::cerr<<e.what();return 2;}
}
''')
    result=subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror',str(source),'-o',str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    return binary


def test_repair_float_native_scalar_does_not_narrow_double_residual_transfer(residual_probe,tmp_path):
    path=tmp_path/'residual';path.write_text('UGKP_SOURCE_RESIDUAL_SCHEMA99 2\n0 1e100\n1 -1e100\n')
    result=subprocess.run([str(residual_probe),str(path)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert result.stdout.strip()=='COMPLETE_FP64_RESIDUAL 1e+100 -1e+100'


def test_actual_residual_row_parse_failure_remains_specific(residual_probe,tmp_path):
    path=tmp_path/'residual';path.write_text('UGKP_SOURCE_RESIDUAL_SCHEMA99 2\n0 1e100\n1 malformed\n')
    result=subprocess.run([str(residual_probe),str(path)],capture_output=True,text=True)
    assert result.returncode==2 and 'Invalid source residual row 1' in result.stderr
