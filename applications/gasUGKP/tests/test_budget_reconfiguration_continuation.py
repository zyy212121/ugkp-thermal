"""Actual species reconfiguration is independent of removed audit output."""
import os
import subprocess
from test_shared_gas_adapter import ADAPTER_PROBE, ROOT


def test_reconfiguration_keeps_physical_inventory_without_audit_resources(tmp_path):
    header = ADAPTER_PROBE.split('int main(){', 1)[0]
    setup = ADAPTER_PROBE.split('int main(){', 1)[1].split(' state.turbulenceModel=', 1)[0]
    body = r'''
 assert(ugkwpGpuResidentStrictConfigureGasModelV1(&state,&args,text.data(),nullptr)==0);
 const double values[4]={.4,.5,.6,.5};
 for(int i=0;i<4;++i)state.gasSpecies.rho[i]=values[i];
 auto* saved=state.gasSpecies.rho;const int before=allocations;
 for(int repeat=0;repeat<3;++repeat){
  assert(ugkwpGpuResidentStrictConfigureGasModelV1(&state,&args,text.data(),nullptr)==0);
  assert(state.gasSpecies.rho==saved&&device.gasSpecies.rho==saved&&allocations==before);
  for(int i=0;i<4;++i)assert(state.gasSpecies.rho[i]==values[i]);
 }
 releaseSharedGasSpecies(state.gasSpecies);assert(allocations==0);
}
'''
    source=tmp_path/'reconfigure.cpp';source.write_text(header+'int main(){'+setup+body)
    binary=tmp_path/'reconfigure'
    build=subprocess.run(['g++','-std=c++17','-O1','-I'+str(ROOT),str(source),'-o',str(binary)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    path=tmp_path/'absent'/'ledger.jsonl'
    run=subprocess.run([str(binary)],env={**os.environ,'UGKP_GAS_BUDGET_OUTPUT':str(path)},capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
    assert not path.exists()
