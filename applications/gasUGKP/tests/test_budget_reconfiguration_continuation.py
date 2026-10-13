"""Actual adapter/ledger reconfiguration with bounded host allocation checks."""
import json
import os
from pathlib import Path
import subprocess
import pytest
from test_shared_gas_adapter import ADAPTER_PROBE, ROOT

@pytest.fixture(scope='module')
def reconfigure_probe(tmp_path_factory):
    out=tmp_path_factory.mktemp('budget-reconfigure')
    h=ADAPTER_PROBE.split('int main(){',1)[0]
    h=h.replace('struct DeviceState {','#include "common/gasTransport/GasBudgetAudit.H"\nstruct GasBudgetHostLedger;\nstruct DeviceState {')
    h=h.replace(' double* rho=nullptr;', ' double* rho=nullptr,*rhoE=nullptr,*V=nullptr;ugkwp::GasBudgetAuditState gasBudgetAudit;GasBudgetHostLedger*gasBudgetHost=nullptr;')
    h=h.replace('#include "applications/gasUGKP/private_backend/SharedGasAdapter.cuh"',r'''
#include <cstdio>
#include <new>
#include <memory>
bool failLedger=false,failMetadataBody=false,ledgerFailed=false,failClose=false;int metadataCountdown=-1;void* trackedLedger=nullptr;std::size_t ledgerSize=0;int closes=0;
void*operator new(std::size_t n){if(failMetadataBody){if(n==ledgerSize)metadataCountdown=1;else if(metadataCountdown==0){failMetadataBody=false;ledgerFailed=true;throw std::bad_alloc();}else if(metadataCountdown>0)--metadataCountdown;}if(failLedger&&n==ledgerSize){failLedger=false;ledgerFailed=true;throw std::bad_alloc();}if(void*p=std::malloc(n)){if(failMetadataBody&&n==ledgerSize)trackedLedger=p;return p;}throw std::bad_alloc();}
void operator delete(void*p)noexcept{if(p==trackedLedger)trackedLedger=nullptr;std::free(p);}void operator delete(void*p,std::size_t)noexcept{if(p==trackedLedger)trackedLedger=nullptr;std::free(p);}
namespace std{int countedClose(FILE*p){++closes;const int rc=::fclose(p);if(failClose){failClose=false;return EOF;}return rc;}}
#define fclose countedClose
#include "applications/gasUGKP/private_backend/SharedGasAdapter.cuh"
#undef fclose
''')
    body=r'''
const char* model2=R"(gasMode mixtureFrozen;species(A B);elements(X Y);speciesThermo{
 A{model linearCp;molarMass .028;minTemperature 100;maxTemperature 4000;coefficients(1040 0 0);atoms(1 0);}
 B{model linearCp;molarMass .032;minTemperature 100;maxTemperature 4000;coefficients(1040 0 0);atoms(0 2);}}
 diffusion{model none;})";
const char* model3=R"(gasMode mixtureFrozen;species(B A C);elements(X Y);speciesThermo{
 A{model linearCp;molarMass .028;minTemperature 100;maxTemperature 4000;coefficients(1040 0 0);atoms(1 0);}
 B{model linearCp;molarMass .032;minTemperature 100;maxTemperature 4000;coefficients(1040 0 0);atoms(0 2);}
 C{model linearCp;molarMass .044;minTemperature 100;maxTemperature 4000;coefficients(1100 0 0);atoms(9 9);}}
 diffusion{model none;})";
int configure(DeviceState*s,const std::string&text){auto m=ugkwp::parseGasModelProperties(text);ugkwpGpuIpc::GasModelConfigureArgsV1 a{1,1,3,0,m.speciesOrderHash,m.thermoHash,0,text.size(),0};return ugkwpGpuResidentStrictConfigureGasModelV1(s,&a,text.data(),nullptr);}
void publish(DeviceState*s,double time){double values[4]={.25,.5,.75,.5};assert(ugkwpGpuResidentStrictUploadSpeciesV1(s,nullptr,values)==0);assert(prepareGasBudgetRequest(s,.1,time)==0&&beginGasBudgetInterval(s)==0&&commitGasBudgetInterval(s)==0);}
int main(int argc,char**argv){assert(argc==4);const std::string mode=argv[1];DeviceState state,device;state.deviceState=&device;
 double rho[2]={1,1},energy[2]={5,6},volume[2]={1,2};state.rho=rho;state.rhoE=energy;state.V=volume;
 setenv("UGKP_GAS_BUDGET_OUTPUT",argv[2],1);assert(configure(&state,mode=="initial_extra"?model3:model2)==0);publish(&state,0);
 const auto oldHash=state.gasSpecies.thermo.speciesOrderHash;auto*oldSpecies=state.gasSpecies.rho;const int before=allocations;
 if(mode!="initial_extra"){
  if(mode=="open_failure"||mode=="close_failure")setenv("UGKP_GAS_BUDGET_OUTPUT",argv[3],1);
  if(mode=="metadata_failure"||mode=="metadata_body_failure"){ledgerSize=sizeof(GasBudgetHostLedger);failLedger=mode=="metadata_failure";failMetadataBody=mode=="metadata_body_failure";}
  if(mode=="close_failure")failClose=true;
  const int rc=configure(&state,model3);assert((rc!=0)==(mode!="extra"));
  assert(!state.gasModelPoisoned&&state.gasSpecies.rho==oldSpecies&&allocations==before);
  if(mode=="metadata_failure"||mode=="metadata_body_failure"){
   assert(ledgerFailed&&!trackedLedger&&state.gasSpecies.thermo.speciesOrderHash==oldHash&&device.gasSpecies.thermo.speciesOrderHash==oldHash);
   assert(state.gasBudgetHost->initialized&&state.gasBudgetHost->identity.find(std::to_string(oldHash))!=std::string::npos);
  }else{
   assert(state.gasSpecies.thermo.speciesOrderHash!=oldHash&&device.gasSpecies.thermo.speciesOrderHash==state.gasSpecies.thermo.speciesOrderHash);
   assert(!state.gasBudgetHost->initialized&&state.gasBudgetHost->identity.find(std::to_string(state.gasSpecies.thermo.speciesOrderHash))!=std::string::npos);
  }
  if(mode=="open_failure")assert(state.gasBudgetHost->path==argv[2]&&last.find("open failed")!=std::string::npos);
  if(mode=="close_failure")assert(state.gasBudgetHost->path==argv[3]&&closes==1&&last.find("close failed")!=std::string::npos);
  publish(&state,.1);
 }
 assert(state.gasBudgetHost->speciesCount==2&&state.gasBudgetAudit.speciesCount==2&&device.gasBudgetAudit.speciesCount==2);
 assert(state.gasBudgetHost->elementMassFactors.size()==4);
 releaseGasBudget(&state);releaseSharedGasSpecies(state.gasSpecies);assert(allocations==0);
 assert(closes==(mode=="close_failure"?2:1));
}
'''
    source=out/'probe.cpp';source.write_text(h+body);binary=out/'probe'
    build=subprocess.run(['g++','-std=c++17','-O0','-g','-fsanitize=address','-fno-omit-frame-pointer','-I'+str(ROOT),str(source),'-o',str(binary)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    return binary

@pytest.mark.parametrize('mode',['initial_extra','extra','open_failure','metadata_failure','metadata_body_failure','close_failure'])
def test_actual_selected_extent_and_output_error_reuse(tmp_path,reconfigure_probe,mode):
    first=tmp_path/'first.jsonl';second=tmp_path/'second.jsonl'
    if mode=='open_failure':second.write_text('preexisting destination\n')
    run=subprocess.run([str(reconfigure_probe),mode,str(first),str(second)],capture_output=True,text=True,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0'})
    assert run.returncode==0,run.stdout+run.stderr
    rows=[json.loads(line)for line in first.read_text().splitlines()]
    if mode=='close_failure':rows+=[json.loads(line)for line in second.read_text().splitlines()]
    if mode=='open_failure':assert second.read_text()=='preexisting destination\n'
    segments=[r for r in rows if r['record']=='NEW_SEGMENT'];accepted=[r for r in rows if r['record']=='ACCEPTED_INTERVAL']
    assert len(segments)==(1 if mode in ['initial_extra','metadata_failure','metadata_body_failure'] else 2)
    assert len(accepted)==(1 if mode=='initial_extra' else 2)
    last=segments[-1]
    assert last['species']==(['A','B']if mode in ['metadata_failure','metadata_body_failure'] else ['B','A'])
    assert last['output_path']==str(second if mode=='close_failure' else first)
    assert last['element_mass_factors_mol_per_kg']==pytest.approx([1/.028,0,0,2/.032]if mode in ['metadata_failure','metadata_body_failure']else[0,2/.032,1/.028,0])
    for row in accepted:
        assert row['inventory']==[3,17,1.25,1.75]
        assert len(row['cumulative_boundary_transport'])==4 and len(row['cumulative_chemistry_species'])==2
        assert row['completed_duration']==.1
