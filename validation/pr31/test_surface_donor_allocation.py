from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
def test_shared_cell_budget_is_not_duplicated(tmp_path):
    source=tmp_path/'donors.cpp'
    source.write_text(r'''#include "ablation/SurfaceDonorAllocation.H"
#include <cassert>
#include <cmath>
#include <algorithm>
int main(){
 chmt::HostState h;h.gasMesh.volumes={1,2,3};
 h.gasMesh.owner={0,0,1,1,2};h.gasMesh.areaVectors={{1,0,0},{0,3,0},{0,0,2},{-2,0,0},{0,-7,0}};
 h.surface.gasFace={0,1,2,3,4};h.surface.area={1,3,2,2,7};
 std::vector<double> w;std::string error;assert(chmt::surfaceGasDonorShares(h,w,error));
 assert(w[0]==.25&&w[1]==.75&&w[2]==.5&&w[3]==.5&&w[4]==1);
 for(int cell=0;cell<3;++cell){double budget=0;for(int i=0;i<5;++i)if(h.gasMesh.owner[h.surface.gasFace[i]]==cell)budget+=w[i];assert(budget==1);}
 std::reverse(h.surface.gasFace.begin(),h.surface.gasFace.end());assert(chmt::surfaceGasDonorShares(h,w,error));
 assert(w[0]==1&&w[3]==.75&&w[4]==.25);
 h.surface.gasFace={0};h.surface.area={1};assert(chmt::surfaceGasDonorShares(h,w,error)&&w[0]==1);
 h.surface.gasFace={-1};assert(chmt::surfaceGasDonorShares(h,w,error)&&w[0]==0);
 h.surface.gasFace={50};assert(!chmt::surfaceGasDonorShares(h,w,error));
}
''')
    exe=tmp_path/'donors'
    subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-I'+str(ROOT/'applications/CHMT'),str(source),'-o',str(exe)],check=True)
    subprocess.run([exe],check=True)
