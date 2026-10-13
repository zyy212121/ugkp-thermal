"""Actual FSH reader tail preserves saved values while dropping its certificate."""
from pathlib import Path
import subprocess
import re
ROOT=Path(__file__).resolve().parents[1]
OWNER='applications/FSH/gpu/GpuResidentStrict.H'
BASE='1dc2947987aa47f9d12904a84bbd308c6e746e8b'

def test_fsh_saved_velocity_reader_tail(tmp_path):
    sources=[subprocess.check_output(['git','show',BASE+':'+OWNER],cwd=ROOT,text=True),(ROOT/OWNER).read_text()]
    outputs=[]
    for label,source in zip(['base','current'],sources):
        a=source.index('// Legacy files have no saved impact velocity.')
        b=source.index('\n        check\n',a)
        saved=source.index('        check(ugkwpGpuResidentStrictUploadParticleSavedVelocity',b)
        end=source.index(';',saved)+1
        body=source[a:end]
        fields=sorted(set(re.findall(r'(\w+)\.begin\(\)',body)) - {'puxOld','puyOld','puzOld'})
        declarations=';'.join('Field '+name+'(nParticles)' for name in fields)+';'
        legacy=re.search(r'scalarField puxOld\(nParticles,0.0\), puyOld\(nParticles,0.0\), puzOld\(nParticles,0.0\);',source).group(0)
        code=r'''
#include <vector>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <iostream>
struct Field:std::vector<double>{using std::vector<double>::vector;const double* cdata()const{return data();}double* begin(){return data();}};
unsigned copies=0,visits=0,mirrors=0;int failure=0;std::vector<double> values;
template<class... T>int ugkwpGpuResidentStrictUploadParticleRestartMirror(T...){++mirrors;return failure==1;}
int ugkwpGpuResidentStrictUploadParticleSavedVelocity(void*,int n,const double*x,const double*y,const double*z){++copies;if(failure==2)return 1;values.clear();for(auto p:{x,y,z})values.insert(values.end(),p,p+n);return 0;}
void check(int value,const char*){if(value)throw std::runtime_error("transfer");}
int fail(){throw std::runtime_error("policy");}
#define forAll(x,i) for(std::size_t i=0;i<x.size()&&(++visits,true);++i)
#define FatalErrorInFunction stream
#define FatalError 1
#define exit(x) fail()
void reader(Field puxOld,Field puyOld,Field puzOld){void*handle_=nullptr;int nParticles=puxOld.size();std::ostringstream stream;
DECLARATIONS
BODY
}
int main(){for(double value:{0.,-0.,1.2345678901234567,1e300,std::numeric_limits<double>::denorm_min(),std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}){
 copies=visits=0;Field x{value},y{-2.},z{3.};try{reader(x,y,z);if(values.size()!=3||std::memcmp(values.data(),x.data(),8))return 2;std::cout<<"uploaded ";}catch(const std::runtime_error&){std::cout<<"refused ";}std::cout<<copies<<' '<<visits<<'\n';}
 copies=visits=0;reader({}, {}, {});std::cout<<"empty "<<copies<<' '<<visits<<'\n';
 {using scalarField=Field;int nParticles=2;LEGACY;reader(puxOld,puyOld,puzOld);for(double x:values)if(x!=0)return 3;}
 for(int f:{1,2}){failure=f;copies=mirrors=0;bool caught=false;try{reader({1.},{2.},{3.});}catch(const std::runtime_error&){caught=true;}if(!caught||mirrors!=1||copies!=unsigned(f-1))return 4;}
 }
'''.replace('BODY',body).replace('DECLARATIONS',declarations).replace('LEGACY',legacy)
        src=tmp_path/(label+'.cpp');src.write_text(code);exe=tmp_path/label
        subprocess.run(['g++','-std=c++14','-O2',str(src),'-o',str(exe)],check=True)
        result=subprocess.check_output([str(exe)],text=True);(tmp_path/(label+'.out')).write_text(result);outputs.append(result.splitlines())
    assert outputs[0][:5]==['uploaded 1 1']*5
    assert outputs[0][5:7]==['refused 0 1']*2
    assert outputs[1][:7]==['uploaded 1 0']*7
    assert outputs[1][7]=='empty 1 0'
