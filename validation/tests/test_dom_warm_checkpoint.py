pass
from pathlib import Path
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]
@pytest.mark.parametrize("bits",[32,64])
def test_production_cache_io(tmp_path,bits):
    base=ROOT/("precision/fp32" if bits==32 else "")
    source=(base/"common/thermal/radiation/MieDoRadiationSolver.C").read_text()
    body=source[source.index("bool MieDoRadiationSolver::writeWarmStart"):source.index("MieDoResult MieDoRadiationSolver::solve(const MieDoSnapshot& snapshot) const")]
    harness=r'''#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cassert>
#define WarningInFunction std::cerr
using std::endl; static auto& Info=std::cout;
namespace ugkp { namespace gpu {using RadiationReal=REAL;}}
struct Mesh {int nCells()const{return 2;}};
struct Data {Mesh mesh;std::vector<int> directions{1,2,3};};
struct GPU {std::vector<double> values;void downloadIntensities(std::vector<double>& x){x=values;} void uploadIntensities(const std::vector<double>& x){values=x;}};
struct Scratch {bool intensitiesInitialized=false;std::unique_ptr<GPU> gpu{new GPU};void prepare(const std::shared_ptr<Data>&){}};
struct MieDoRadiationWorkspace {std::unique_ptr<Scratch> data_{new Scratch};};
struct MieDoRadiationSolver {std::shared_ptr<Data> data_{new Data};bool writeWarmStart(MieDoRadiationWorkspace&,const std::string&)const;bool readWarmStart(MieDoRadiationWorkspace&,const std::string&)const;};
std::uint64_t warmLayout(const Data& d){return 123+d.directions.size();}
'''.replace("REAL","float" if bits==32 else "double")+body+r'''
int main(int argc,char**argv){std::string p=argv[1];MieDoRadiationSolver solver;MieDoRadiationWorkspace a,b;
assert(!solver.readWarmStart(b,p));assert(!b.data_->intensitiesInitialized);
assert(solver.writeWarmStart(a,p));assert(solver.readWarmStart(b,p));assert(!b.data_->intensitiesInitialized);
a.data_->intensitiesInitialized=true;a.data_->gpu->values={0,1,2,3,4,5};
assert(solver.writeWarmStart(a,p));{std::ifstream size(p,std::ios::binary|std::ios::ate);assert(size.tellg()==56+6*sizeof(ugkp::gpu::RadiationReal));}assert(solver.readWarmStart(b,p));assert(b.data_->intensitiesInitialized);assert(a.data_->gpu->values==b.data_->gpu->values);
solver.data_->directions.push_back(4);assert(!solver.readWarmStart(b,p));assert(!b.data_->intensitiesInitialized);solver.data_->directions.pop_back();
{std::ofstream bad(p,std::ios::binary|std::ios::trunc);bad<<"bad";}assert(!solver.readWarmStart(b,p));assert(!b.data_->intensitiesInitialized);
assert(solver.writeWarmStart(a,p));{std::fstream f(p,std::ios::binary|std::ios::in|std::ios::out);std::uint64_t wrong=99;f.seekp(16);f.write((char*)&wrong,8);}assert(!solver.readWarmStart(b,p));
}
'''
    cpp=tmp_path/"cache.cpp";cpp.write_text(harness)
    subprocess.run(["g++","-std=c++17",str(cpp),"-o",str(tmp_path/"test")],check=True)
    subprocess.run([str(tmp_path/"test"),str(tmp_path/"cache.bin")],check=True)

def test_application_hooks():
    for prefix in ["","precision/fp32/"]:
        cht=(ROOT/(prefix+"applications/CHT/thermal/GpuSolidThermalCoupler.C")).read_text()
        assert "particleRadiationCoupler->readWarmStart" in cht
        assert "particleRadiationCoupler->writeWarmStart" in cht
        assert "runTime.timePath()/\"domWarmStart.bin\"" in cht
        chmt=(ROOT/(prefix+"applications/CHMT/io/CoupledRadiation.C")).read_text()
        assert "data_->coupler->readWarmStart(path)" in chmt
        assert "data_->coupler->writeWarmStart(path)" in chmt
    driver=(ROOT/"applications/CHMT/CHMT.C").read_text()
    assert 'radiation->readWarmStart((restart/"domWarmStart.bin")' in driver
    assert 'radiation->writeWarmStart((checkpointPath/"domWarmStart.bin")' in driver
