"""Execute the bounded scratch policy against real workspace layouts."""
from test_mixture_state import compile_probe


def test_wall_workspace_auto_and_capacity_buckets(tmp_path):
    compile_probe(tmp_path,r'''
#include "gasTransport/GasBoundaryLayerWorkspace.H"
#include <cassert>
int main(){using namespace ugkwp;
const std::size_t GiB=std::size_t(1)<<30;
const std::size_t bytes=sizeof(gaswall::WallWorkspace<double,10,48>);
auto r=resolveBoundaryLayerWorkspace(100000,0,80,8*GiB,bytes);
assert(r.slots==2560&&r.bytes==2560*std::size_t(bytes)&&r.budget==GiB);
r=resolveBoundaryLayerWorkspace(100000,0,200,8*GiB,bytes);assert(r.slots==4096);
r=resolveBoundaryLayerWorkspace(17,0,80,8*GiB,bytes);assert(r.slots==17);
r=resolveBoundaryLayerWorkspace(1000,0,80,8*bytes,bytes);assert(r.slots==1);
assert(resolveBoundaryLayerWorkspace(1000,0,80,8*bytes-1,bytes).slots==0);
assert(resolveBoundaryLayerWorkspace(1000,0,0,8*GiB,bytes).slots==0);
assert(resolveBoundaryLayerWorkspace(1000,4097,80,8*GiB,bytes).slots==0);
r=resolveBoundaryLayerWorkspace(1000,32,0,32*bytes,bytes);assert(r.slots==32);
assert(resolveBoundaryLayerWorkspace(1000,32,80,32*bytes-1,bytes).slots==0);
assert(gasBoundaryLayerWorkspaceCapacity(24)==24&&gasBoundaryLayerWorkspaceCapacity(25)==48);
assert(gasBoundaryLayerWorkspaceCapacity(48)==48&&gasBoundaryLayerWorkspaceCapacity(49)==128);
assert(gasBoundaryLayerWorkspaceCapacity(128)==128&&gasBoundaryLayerWorkspaceCapacity(129)==0);
assert((gasBoundaryLayerWorkspaceBytes<double,2>(24)==sizeof(gaswall::WallWorkspace<double,2,24>)));
assert((gasBoundaryLayerWorkspaceBytes<double,10>(96)==sizeof(gaswall::WallWorkspace<double,10,128>)));
}
''')
