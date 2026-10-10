from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]

def test_boundary_layer_ipc_payload_preserves_legacy_layout_and_bounds(tmp_path):
    source=tmp_path/'probe.cpp';exe=tmp_path/'probe'
    source.write_text(r'''
#include "applications/gasUGKP/gpu/GpuBackendProtocol.H"
int main(){using namespace ugkwpGpuIpc;BoundaryLayerConfigV1 a{};a.version=1;a.speciesCount=2;a.model=1;a.nodes=24;a.maxIterations=60;a.wallCount=2;a.quadratureCount=6;a.matchingCount=3;a.workspaceSlots=2;
std::uint64_t bytes=0;if(!boundaryLayerPayloadBytes(a,4,2,bytes))return 1;
const auto expected=sizeof(a)+(2+2*3+3)*sizeof(int)+(2*7+2*6+3+2*2)*sizeof(double);
if(bytes!=expected||sizeof(SstConfigArgs)!=152||sizeof(CreateArgs)!=392)return 2;
a.wallCount=5;if(boundaryLayerPayloadBytes(a,4,2,bytes))return 3;a.wallCount=2;
a.nodes=128;if(!boundaryLayerPayloadBytes(a,4,2,bytes))return 4;a.nodes=129;if(boundaryLayerPayloadBytes(a,4,2,bytes))return 7;a.nodes=24;
a.speciesCount=3;if(boundaryLayerPayloadBytes(a,4,2,bytes))return 5;a.speciesCount=2;
a.workspaceSlots=0;if(!boundaryLayerPayloadBytes(a,4,2,bytes))return 6;
a.workspaceSlots=4097;if(boundaryLayerPayloadBytes(a,4,2,bytes))return 8;a.workspaceSlots=0;
a.budgetAudit=1;if(!boundaryLayerPayloadBytes(a,4,2,bytes))return 9;
a.budgetAudit=2;if(boundaryLayerPayloadBytes(a,4,2,bytes))return 10;
if(sizeof(a)!=80||boundaryLayerDiagnosticScalars!=25)return 11;}
''')
    result=subprocess.run(['g++','-std=c++17','-I'+str(ROOT),str(source),'-o',str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    result=subprocess.run([str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
