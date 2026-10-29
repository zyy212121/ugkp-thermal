"""Physical wall anchoring under translation, rotation, shear and face reorder."""
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
SOURCE=r'''
#include "thermal/AttachedParticleGeometry.H"
#include <cassert>
#include <cstdio>
#include <initializer_list>
using namespace ugkp::thermal;using namespace ugkp::gpu::geometry;
void near(Point a,Point b){assert(magnitude(a-b)<2e-12);}
int main(){
 Point old[8]={{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,2},{1,0,2},{1,1,2},{0,1,2}},next[8];
 int offsets[3]={0,4,8},indices[8]={0,1,2,3,4,5,6,7},reorder[8]={6,7,4,5,2,3,0,1},owner[2]={8,9};
 ParticleSurfaceGeometry a{2,8,offsets,indices,owner,old},b{2,8,offsets,indices,owner,next};
 int cases=0;
 for(double x:{0.,.15,.4,1.})for(double y:{0.,.25,.8,1.})for(double offset:{-1e-9,0.,1e-9}) {
  Point p{x,y,offset};
  for(int mode=0;mode<4;++mode) {
   for(int k=0;k<8;++k){const auto q=old[k];
    next[k]=mode==0?q+Point{.3,-.2,-.1}:mode==1?Point{-q.y,q.x,q.z}:mode==2?Point{q.z,q.y,-q.x}:Point{2*q.x+.5*q.y,.75*q.y,q.z-.2};}
   const auto mapped=mapAttachedParticleGeometry(a,b,0,0,p);assert(mapped.valid&&mapped.cell==8&&mapped.face==0);
   const Point expected=mode==0?p+Point{.3,-.2,-.1}:mode==1?Point{-p.y,p.x,p.z}:mode==2?Point{p.z,p.y,-p.x}:Point{2*p.x+.5*p.y,.75*p.y,p.z-.2};
   near(mapped.position,expected);
   ++cases;
  }
 }
 for(int k=0;k<8;++k)next[k]=old[k]+Point{0,0,-.4};
 b.facePoints=reorder;
 auto mapped=mapAttachedParticleGeometry(a,b,0,1,{.2,.7,-1e-9});
 assert(mapped.valid&&mapped.face==1&&mapped.cell==9);near(mapped.position,{.2,.7,-.400000001});
 // Changed vertex membership and collapsed geometry are explicit mapping
 // failures; the caller cannot publish a fabricated old attachment as success.
 assert(!mapAttachedParticleGeometry(a,b,0,0,{.2,.7,0}).valid);
 b.facePoints=indices;for(int k=0;k<8;++k)next[k]={};
 assert(!mapAttachedParticleGeometry(a,b,0,0,{.2,.7,0}).valid);
 std::printf("%d affine/rigid wall mappings plus identity and invalid geometry checks passed\n",cases);
}
'''

def test_geometric_anchor(tmp_path):
    source=tmp_path/"mapping.cpp";source.write_text(SOURCE);binary=tmp_path/"mapping"
    build=subprocess.run(["g++","-std=c++17","-O2","-Wall","-Wextra","-I"+str(ROOT/"common"),str(source),"-o",str(binary)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([binary],capture_output=True,text=True);assert run.returncode==0,run.stdout+run.stderr
