"""Native ownership/capture invalidation, using the real public adapter."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
from test_shared_gas_adapter import ADAPTER_PROBE, ROOT


def test_native_model_reconfigure_retains_species_and_invalidates_old_captures(tmp_path):
    source=ADAPTER_PROBE.replace('#include <cassert>', '#include <cassert>\n#include <cmath>')
    source=source.replace(' bool gasModelPoisoned=false;', ' double gasGraphDt=.25;struct Snapshot {bool valid=true;};struct Trial {Snapshot trial,interval;} gasTrial;\n bool gasModelPoisoned=false;')
    point=' assert(ugkwpGpuResidentStrictConfigureGasModelV1(&state,&args,text.data(),nullptr)==0);\n double density[]='
    replacement=r''' assert(ugkwpGpuResidentStrictConfigureGasModelV1(&state,&args,text.data(),nullptr)==0);
 state.gasSpecies.rho[0]=987;
 auto* inventory=state.gasSpecies.rho;state.gasGraphDt=.25;state.gasTrial.trial.valid=state.gasTrial.interval.valid=true;
 assert(ugkwpGpuResidentStrictConfigureGasModelV1(&state,&args,text.data(),nullptr)==0);
 assert(state.gasSpecies.rho==inventory&&device.gasSpecies.rho==inventory&&inventory[0]==987);
 assert(std::isnan(state.gasGraphDt)&&!state.gasTrial.trial.valid&&!state.gasTrial.interval.valid);
 inventory[0]=0; // Restore the zero initial fixture for the subsequent upload/error tests.
 double density[]='''
    assert point in source
    source=source.replace(point,replacement)
    (tmp_path/'probe.cpp').write_text(source)
    cmd=['g++','-std=c++17','-O1','-I'+str(ROOT),str(tmp_path/'probe.cpp'),'-o',str(tmp_path/'probe')]
    b=subprocess.run(cmd,capture_output=True,text=True);assert b.returncode==0,b.stderr
    r=subprocess.run([str(tmp_path/'probe')],capture_output=True,text=True)
    (tmp_path/'run.log').write_text(r.stdout+r.stderr)
    assert r.returncode==0,r.stdout+r.stderr


def test_wall_declared_species_count_does_not_replace_actual_supplied_storage(tmp_path):
    from test_boundary_layer_lifecycle import _publication_probe
    source=_publication_probe().replace('cfg.speciesCount=2;', 'cfg.speciesCount=1;')
    (tmp_path/'probe.cpp').write_text(source)
    cmd=['g++','-std=c++17','-O1','-I'+str(ROOT),'-I'+str(ROOT/'common'),str(tmp_path/'probe.cpp'),'-o',str(tmp_path/'probe')]
    b=subprocess.run(cmd,capture_output=True,text=True);assert b.returncode==0,b.stderr
    r=subprocess.run([str(tmp_path/'probe'),'0','-1','0'],capture_output=True,text=True)
    (tmp_path/'run.log').write_text(r.stdout+r.stderr)
    assert r.returncode==0,r.stdout+r.stderr


def test_actual_graph_protocol_rebuilds_after_configuration_and_retains_errors(tmp_path):
    import re
    import sys
    sys.path.insert(0,str((VALIDATION_ROOT / 'tests/gas_transport')))
    from test_gas_advance_view import PREAMBLE
    pre=PREAMBLE.replace('std::vector<std::string> trace;', 'std::vector<std::string> trace;int graphs=0,launches=0,destroys=0,failGraph=0;')
    pre=pre.replace('int cudaGraphExecDestroy(cudaGraphExec_t){return 0;}', 'int cudaGraphExecDestroy(cudaGraphExec_t){if(failGraph==1)return 7;++destroys;return 0;}')
    pre=pre.replace('int cudaGraphDestroy(cudaGraph_t){return 0;}', 'int cudaGraphDestroy(cudaGraph_t){if(failGraph==2)return 7;return 0;}')
    pre=pre.replace('int cudaStreamCreate(cudaStream_t*s){', 'int cudaStreamCreate(cudaStream_t*s){if(failGraph==3)return 7;')
    pre=pre.replace('int cudaStreamDestroy(cudaStream_t){return 0;}', 'int cudaStreamDestroy(cudaStream_t){return failGraph==6?7:0;}')
    pre=pre.replace('int cudaStreamBeginCapture(cudaStream_t,int){return 0;}', 'std::vector<cudaKernelNodeParams> graphNodes;int cudaStreamBeginCapture(cudaStream_t,int){graphNodes.clear();return failGraph==4?7:0;}')
    pre=pre.replace('std::vector<cudaKernelNodeParams> graphNodes;\nint cudaGraphGetNodes', 'int cudaGraphGetNodes')
    pre=pre.replace('int cudaStreamEndCapture(cudaStream_t,cudaGraph_t*g){', 'int cudaStreamEndCapture(cudaStream_t,cudaGraph_t*g){if(failGraph==5)return 7;')
    pre=pre.replace('int cudaGraphInstantiate(cudaGraphExec_t*e,cudaGraph_t,void*,void*,int){', 'int cudaGraphInstantiate(cudaGraphExec_t*e,cudaGraph_t,void*,void*,int){if(failGraph==7)return 7;++graphs;')
    pre=pre.replace('int cudaGraphLaunch(cudaGraphExec_t,void*){', 'int cudaGraphLaunch(cudaGraphExec_t,void*){if(failGraph==8)return 7;++launches;')
    storage=(ROOT/'applications/gasUGKP/private_backend/SharedGasStorage.cuh').read_text()
    helper=storage[storage.index('#include <limits>'):storage.index('template<class T>\nint releaseSharedGasPointer')]
    body=re.sub(r'<<<[\s\S]*?>>>','',(ROOT/'common/GpuGasAdvance.cuh').read_text())
    main=r'''
int main(){IndependentHost s;Payload device;s.deviceState=&device;expectedDevice=&device;expectedTime=2;
 assert(advancePureGasGraph(&s,.25,2)==0&&graphs==1&&launches==1&&destroys==0);
 assert(advancePureGasGraph(&s,.25,2)==0&&graphs==1&&launches==2&&destroys==0);
 invalidateSharedGasConfiguration(&s,false); // pointer-only publication is deviceState-indirect
 assert(advancePureGasGraph(&s,.25,2)==0&&graphs==1&&launches==3&&destroys==0);
 s.hostTurbulenceModel=3;s.hostGasTimeIntegrator=3;invalidateSharedGasConfiguration(&s);
 assert(advancePureGasGraph(&s,.25,2)==0&&graphs==2&&launches==4&&destroys==1);
 for(int fault=1;fault<=8;++fault){invalidateSharedGasConfiguration(&s);failGraph=fault;assert(advancePureGasGraph(&s,.25,2)!=0);failGraph=0;assert(advancePureGasGraph(&s,.25,2)==0);}
 std::printf("graph create=%d launch=%d destroy=%d eight actual error paths retained\n",graphs,launches,destroys);
}
'''
    (tmp_path/'probe.cpp').write_text('#include <cassert>\n#include "gasTransport/GasStateView.H"\n'+pre+helper+body+main)
    cmd=['g++','-std=c++17','-O1','-I'+str(ROOT/'common'),str(tmp_path/'probe.cpp'),'-o',str(tmp_path/'probe')]
    b=subprocess.run(cmd,capture_output=True,text=True);assert b.returncode==0,b.stderr
    r=subprocess.run([str(tmp_path/'probe')],capture_output=True,text=True);(tmp_path/'run.log').write_text(r.stdout+r.stderr)
    assert r.returncode==0,r.stdout+r.stderr


def test_native_wall_reconfigure_retains_physical_sst_inventory(tmp_path):
    from test_boundary_layer_lifecycle import _publication_probe
    original=_publication_probe()
    marker='  const int calls=publicationAttempts,live=allocations;\n  assert(configure()==0&&publicationAttempts==calls+5&&allocations==live);'
    assert marker in original
    source=original.replace('  const int calls=publicationAttempts,live=allocations;\n  assert(configure()==0&&publicationAttempts==calls+5&&allocations==live);', '''
  auto* inventory=state.k;inventory[0]=777;
  const int calls=publicationAttempts,live=allocations;
  assert(configure()==0&&publicationAttempts==calls+5&&allocations==live);
  assert(state.k==inventory&&inventory[0]==777&&device.k==inventory);
''')
    (tmp_path/'probe.cpp').write_text(source)
    cmd=['g++','-std=c++17','-O1','-I'+str(ROOT),'-I'+str(ROOT/'common'),str(tmp_path/'probe.cpp'),'-o',str(tmp_path/'probe')]
    b=subprocess.run(cmd,capture_output=True,text=True);assert b.returncode==0,b.stderr
    r=subprocess.run([str(tmp_path/'probe'),'1','-1','1'],capture_output=True,text=True)
    (tmp_path/'run.log').write_text(r.stdout+r.stderr)
    assert r.returncode==0,r.stdout+r.stderr


def test_native_all_sixteen_dynamic_fields_are_retained_at_table_only_cost(tmp_path):
    pre=ADAPTER_PROBE.split('int main(){',1)[0]
    pre=pre.replace('static int allocations=0,', 'static int mallocCalls=0,freeCalls=0,zeroCalls=0,viewCalls=0;static std::size_t h2dBytes=0;\nstatic int allocations=0,')
    pre=pre.replace('if(p){--allocations;', 'if(p){++freeCalls;--allocations;')
    pre=pre.replace('int cudaMemcpy(void*d,const void*s,std::size_t n,int){', 'int cudaMemcpy(void*d,const void*s,std::size_t n,int){++viewCalls;h2dBytes+=n;')
    pre=pre.replace('int cudaMemset(void*d,int v,std::size_t n){', 'int cudaMemset(void*d,int v,std::size_t n){if(n)++zeroCalls;')
    pre=pre.replace('++allocations;return 0;', '++allocations;++mallocCalls;return 0;')
    pre=pre.replace('int copyToDevice(T*d,const T*s,std::size_t n,const char*){', 'int copyToDevice(T*d,const T*s,std::size_t n,const char*){h2dBytes+=n*sizeof(T);')
    text=ADAPTER_PROBE[ADAPTER_PROBE.index(' const std::string text='):ADAPTER_PROBE.index(' state.turbulenceModel=')]
    source=pre+'\n#include <cstdio>\nint main(){DeviceState state,device;state.deviceState=&device;\n'+text+r'''
 assert(ugkwpGpuResidentStrictConfigureGasModelV1(&state,&args,text.data(),nullptr)==0);
 auto& s=state.gasSpecies;assert(!allocateSharedGasZero(s.chemistryStatus,2));
 struct Field {void* p;std::size_t n;};std::vector<Field> fields;
 #define FIELD(name,count) fields.push_back({s.name,(count)*sizeof(*s.name)})
 FIELD(rho,4);FIELD(initial,4);FIELD(flux,4);FIELD(gradX,4);FIELD(gradY,4);FIELD(gradZ,4);FIELD(limiter,2);FIELD(boundaryMassFraction,4);FIELD(compositionBoundaryFixed,2);FIELD(positivityScale,4);FIELD(soundSpeed,2);FIELD(heatCapacity,2);FIELD(gasConstant,2);FIELD(cellStatus,2);FIELD(faceStatus,2);FIELD(chemistryStatus,2);
 #undef FIELD
 std::vector<std::vector<unsigned char>> saved;for(std::size_t i=0;i<fields.size();++i){auto f=fields[i];std::memset(f.p,11+i,f.n);saved.emplace_back((unsigned char*)f.p,(unsigned char*)f.p+f.n);}
 mallocCalls=freeCalls=zeroCalls=viewCalls=0;h2dBytes=0;
 assert(ugkwpGpuResidentStrictConfigureGasModelV1(&state,&args,text.data(),nullptr)==0);
 for(std::size_t i=0;i<fields.size();++i)assert(!std::memcmp(fields[i].p,saved[i].data(),fields[i].n));
 assert(device.gasSpecies.chemistryStatus==s.chemistryStatus);
 const auto bytes=model.species.size()*sizeof(ugkwp::SpeciesThermoData<double>)+model.coefficients.size()*sizeof(double)+model.diffusionCoefficients.size()*sizeof(double)+sizeof(s);
 assert(mallocCalls==3&&freeCalls==3&&zeroCalls==0&&viewCalls==1&&h2dBytes==bytes);
 std::printf("native repeat allocations=%d frees=%d zero=%d view=%d H2D=%zu\n",mallocCalls,freeCalls,zeroCalls,viewCalls,h2dBytes);
 releaseSharedGasSpecies(s);assert(allocations==0);
}
'''
    (tmp_path/'probe.cpp').write_text(source)
    b=subprocess.run(['g++','-std=c++17','-O1','-I'+str(ROOT),str(tmp_path/'probe.cpp'),'-o',str(tmp_path/'probe')],capture_output=True,text=True);assert b.returncode==0,b.stderr
    r=subprocess.run([str(tmp_path/'probe')],capture_output=True,text=True);(tmp_path/'run.log').write_text(r.stdout+r.stderr)
    assert r.returncode==0,r.stdout+r.stderr


def test_native_actual_free_failure_retains_failed_owner_and_reports_committed_view(tmp_path):
    source=ADAPTER_PROBE.split('int main(){',1)[0]
    source=source.replace('static int allocations=0,', 'static bool failFree=false;\nstatic int allocations=0,')
    source=source.replace('int cudaFree(void*p){if(p){', 'int cudaFree(void*p){if(p&&failFree){failFree=false;return 7;}if(p){')
    model=ADAPTER_PROBE[ADAPTER_PROBE.index(' const std::string text='):ADAPTER_PROBE.index(' state.turbulenceModel=')]
    source+='\nint main(){DeviceState state,device;state.deviceState=&device;\n'+model+r'''
 assert(ugkwpGpuResidentStrictConfigureGasModelV1(&state,&args,text.data(),nullptr)==0);
 auto* rho=state.gasSpecies.rho;rho[0]=17;const int count=allocations;
 failFree=true;assert(ugkwpGpuResidentStrictConfigureGasModelV1(&state,&args,text.data(),nullptr)!=0);
 assert(state.gasModelPoisoned&&state.gasSpecies.rho==rho&&rho[0]==17&&device.gasSpecies.rho==rho&&allocations==count+1);
 assert(state.gasRejectedView.thermo.species||state.gasRejectedView.thermo.coefficients||state.gasRejectedView.diffusivity);
 releaseSharedGasSpecies(state.gasSpecies);releaseSharedGasSpecies(state.gasRejectedView);assert(allocations==0);
}
'''
    (tmp_path/'probe.cpp').write_text(source)
    b=subprocess.run(['g++','-std=c++17','-O1','-I'+str(ROOT),str(tmp_path/'probe.cpp'),'-o',str(tmp_path/'probe')],capture_output=True,text=True);assert b.returncode==0,b.stderr
    r=subprocess.run([str(tmp_path/'probe')],capture_output=True,text=True);(tmp_path/'run.log').write_text(r.stdout+r.stderr)
    assert r.returncode==0,r.stdout+r.stderr


def test_native_wall_free_error_reports_poison_and_keeps_failed_allocation(tmp_path):
    from test_boundary_layer_lifecycle import _publication_probe
    source=_publication_probe().replace('static int allocations=0,', 'static bool failFree=false;\nstatic int allocations=0,')
    source=source.replace('int cudaFree(void*p){if(p){', 'int cudaFree(void*p){if(p&&failFree){failFree=false;return 7;}if(p){')
    source=source.split('  const int calls=publicationAttempts,live=allocations;')[0]+r'''
  auto* inventory=state.k;inventory[0]=999;const int live=allocations;
  failFree=true;assert(configure()!=0&&state.gasModelPoisoned);
  assert(state.k==inventory&&inventory[0]==999&&allocations==live);
 }
 releaseBoundaryLayerStorage(state);assert(allocations==0);
}
'''
    (tmp_path/'probe.cpp').write_text(source)
    b=subprocess.run(['g++','-std=c++17','-O1','-I'+str(ROOT),'-I'+str(ROOT/'common'),str(tmp_path/'probe.cpp'),'-o',str(tmp_path/'probe')],capture_output=True,text=True);assert b.returncode==0,b.stderr
    r=subprocess.run([str(tmp_path/'probe'),'1','-1','1'],capture_output=True,text=True);(tmp_path/'run.log').write_text(r.stdout+r.stderr)
    assert r.returncode==0,r.stdout+r.stderr


import pytest

@pytest.mark.parametrize('failure',[1,2,3])
def test_native_wall_real_query_errors_leave_old_owner_usable(tmp_path,failure):
    from test_boundary_layer_lifecycle import _publication_probe
    source=_publication_probe().replace('const int cudaDevAttrMultiProcessorCount=1;', 'int queryFail=0;const int cudaDevAttrMultiProcessorCount=1;')
    source=source.replace('int cudaGetDevice(int*p){', 'int cudaGetDevice(int*p){if(queryFail==1)return 7;')
    source=source.replace('int cudaDeviceGetAttribute(int*p,int,int){', 'int cudaDeviceGetAttribute(int*p,int,int){if(queryFail==2)return 7;')
    source=source.replace('int cudaMemGetInfo(std::size_t*f,std::size_t*t){', 'int cudaMemGetInfo(std::size_t*f,std::size_t*t){if(queryFail==3)return 7;')
    source=source.split(' const int rc=configure();')[0]+r'''
 queryFail=FAILURE;assert(configure()!=0&&!state.gasModelPoisoned&&allocations==0);checkUnrelated(before,device);
 queryFail=0;assert(configure()==0&&state.gasBoundaryLayer.enabled);releaseBoundaryLayerStorage(state);assert(allocations==0);
}
'''.replace('FAILURE',str(failure))
    (tmp_path/'probe.cpp').write_text(source)
    b=subprocess.run(['g++','-std=c++17','-O1','-I'+str(ROOT),'-I'+str(ROOT/'common'),str(tmp_path/'probe.cpp'),'-o',str(tmp_path/'probe')],capture_output=True,text=True);assert b.returncode==0,b.stderr
    r=subprocess.run([str(tmp_path/'probe'),'1','-1','1'],capture_output=True,text=True);(tmp_path/'run.log').write_text(r.stdout+r.stderr)
    assert r.returncode==0,r.stdout+r.stderr


def _actual_sst_code(rev=None):
    import sys,re
    sys.path.insert(0,str((VALIDATION_ROOT / 'tests')))
    import test_sst_owner_lifecycle as owner
    if rev is None:


        current=owner.source('gasUGKP')
        alloc=owner.h.block(current,'int allocateFields(')
        slots=re.findall(r'allocate\(s->(\w+), (nc|nf), "(cudaMalloc SST [^"]+)"\)',alloc)
        assert len(slots)==22
        original_inventory=owner.inventory
        try:
            owner.inventory=lambda app: slots
            c=owner.code('gasUGKP')
        finally:
            owner.inventory=original_inventory
    else:
        c=owner.code('gasUGKP',rev)
    src=owner.source('gasUGKP',rev);scrub=owner.h.block(src,'void scrubHostCalculationScalars(')
    fields=set(re.findall(r's->(\w+)\s*=',scrub))
    existing=set(re.findall(r'\b(\w+)\s*(?:=|;|,)',c[c.index('struct DeviceState'):c.index('DeviceState*asState')]))
    extras='\n'.join('double '+f+'=0;' for f in sorted(fields-existing))
    c=c.replace('struct DeviceState{', '#include "common/gasTransport/GasCapabilities.H"\n#include "common/gasTransport/GasStateView.H"\nstruct DeviceState{\n'+extras+'\nugkwp::GasSpeciesState<double,2> gasSpecies;int hostGasFluxScheme=1,hostGasReconstruction=0,hostGasLimiter=0,hostGasTimeIntegrator=1;')
    adapter=(ROOT/'applications/gasUGKP/private_backend/SharedGasAdapter.cuh').read_text()
    helper=owner.h.block(adapter,'inline ugkwp::GasCapabilityRequest sharedGasCapabilityRequest')
    storage=(ROOT/'applications/gasUGKP/private_backend/SharedGasStorage.cuh').read_text()
    helper=storage[storage.index('#include <limits>'):storage.index('template<class T>\nint releaseSharedGasPointer')]+helper
    c=c.replace('int syncSstConfiguration(',scrub+'\n'+helper+'\nint syncSstConfiguration(',1)
    c=c.replace('int configure(DeviceState&s,Input&i){','double requestedKappa=.41,requestedE=9.8;int configure(DeviceState&s,Input&i){').replace('.25,0,.41,9.8,.09,i.k','.25,0,requestedKappa,requestedE,.09,i.k')
    c=c.replace('void launchInitialise(DeviceState*s,int grid,int block){','void launchInitialise(DeviceState*s,int grid,int block,bool initializeInventory=true){').replace('initialiseSstConservativeStateKernel(s);','initialiseSstConservativeStateKernel(s,initializeInventory);') if 'bool initializeInventory' in c else c
    c=c.replace('initialiseSstConservativeStateKernel<<<grid, block>>>(s->deviceState, !retainInventory);','launchInitialise(s->deviceState,grid,block,!retainInventory);')
    return c,owner.h


def test_native_sst_device_constants_survive_scrub_and_update_only_on_change(tmp_path):
    c,h=_actual_sst_code()
    result=h.run_cpp(tmp_path,c+r'''
int main(){DeviceState s,d;Input i;allocateOriginal(s);setup(s,d);s.gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;
 counters();assert(configure(s,i)==0);complete(s,d,i);assert(copies==8&&launches==1&&syncs==1);
 auto firstBytes=copiedBytes;auto p=d.sstJayatillekeP,y=d.sstThermalYPlus;
 double* fields[]={s.k,s.omega,s.rhoK,s.rhoOmega,s.rhoKInitial,s.rhoOmegaInitial,s.nut};
 for(int j=0;j<7;++j)for(int q=0;q<2;++q)fields[j][q]=700+j*2+q;
 scrubHostCalculationScalars(&s);counters();assert(configure(s,i)==0);
 assert(d.sstJayatillekeP==p&&d.sstThermalYPlus==y);assert(copies==6&&launches==0&&syncs==0&&allocations==0&&memsets==0&&copiedBytes==firstBytes-48);
 std::printf("SST repeat copies=%d H2D=%zu kernels=%d sync=%d\n",copies,copiedBytes,launches,syncs);
 requestedKappa=.37;requestedE=11;counters();assert(configure(s,i)==0);
 const double ratio=d.gasPrClamped/clampMin(d.turbulentPrandtl,OfSmall);
 assert(d.sstJayatillekeP==ugkpwall::jayatillekeSmoothP(ratio)&&d.sstThermalYPlus==ugkpwall::jayatillekeThermalYPlus(ratio,.37,11)&&d.sstThermalYPlus!=y);
 assert(copies==6&&launches==1&&syncs==1);for(int j=0;j<7;++j)for(int q=0;q<2;++q)assert(fields[j][q]==700+j*2+q);
 scrubHostCalculationScalars(&s);counters();assert(configure(s,i)==0&&launches==0&&syncs==0);
 cleanup(s);assert(live.empty());}
''');(tmp_path/'run.log').write_text(result)


def test_native_first_sst_original_bits_and_transfer_order(tmp_path):
    main=r'''
void word(double value){unsigned long long bits;std::memcpy(&bits,&value,8);std::cout<<" "<<bits;}
int main(){DeviceState s,d;Input i;allocateOriginal(s);setup(s,d);s.gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;counters();assert(configure(s,i)==0);complete(s,d,i);
 for(auto*p:{s.k,s.omega,s.rhoK,s.rhoOmega,s.rhoKInitial,s.rhoOmegaInitial,s.nut,s.sstWallDistance})for(int q=0;q<2;++q)word(p[q]);word(d.sstJayatillekeP);word(d.sstThermalYPlus);
 std::cout<<"\nCOST "<<allocations<<" "<<memsets<<" "<<copies<<" "<<copiedBytes<<" "<<launches<<" "<<syncs<<"\n";for(auto&e:events)std::cout<<e<<"\n";cleanup(s);assert(live.empty());}
'''

    c,h=_actual_sst_code()
    result=h.run_cpp(tmp_path,c+main)
    lines=result.splitlines()
    expected_bits=[
        4599075939470750515,4604480259023595110,4611686018427387904,4613937818241073152,
        4600427019358961664,4602903999154015436,4612811918334230528,4612248968380809216,
        4600427019358961664,4602903999154015436,4612811918334230528,4612248968380809216,
        0,0,4602678819172646912,4603579539098121011,13835124156674735100,4623202829888442977,
    ]
    assert list(map(int,lines[0].split()))==expected_bits
    assert lines[1:]==[
        "COST 0 0 8 296 1 1",
        "copy 1 16","copy 1 16","copy 1 16","copy 1 12","copy 1 12",
        "copy 1 24","copy 1 24","copy 1 176","launch","sync",
    ]
    (tmp_path/'run.log').write_text(result)


@pytest.mark.parametrize('failure',range(1,9))
def test_native_changed_sst_real_copy_launch_sync_failure_is_retryable(tmp_path,failure):
    c,h=_actual_sst_code()
    result=h.run_cpp(tmp_path,c+r'''
int main(){DeviceState s,d;Input i;allocateOriginal(s);setup(s,d);s.gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;assert(configure(s,i)==0);s.rhoK[0]=887;requestedKappa=.35;
 counters();if(FAILURE<=6)failCopy=FAILURE;else if(FAILURE==7)failLaunch=1;else failSync=1;assert(configure(s,i)!=0&&s.rhoK[0]==887);
 counters();assert(configure(s,i)==0&&launches==1&&syncs==1&&s.rhoK[0]==887);assert(d.sstThermalYPlus==ugkpwall::jayatillekeThermalYPlus(d.gasPrClamped/d.turbulentPrandtl,.35,9.8));cleanup(s);assert(live.empty());}
'''.replace('FAILURE',str(failure)));(tmp_path/'run.log').write_text(result)


def _free_failure_wall_source():
    from test_boundary_layer_lifecycle import _publication_probe
    source=_publication_probe().replace('static int allocations=0,', 'static int freeCalls=0,freeFailure=-1,freeFailure2=-1;static const void* freeTarget=nullptr;\nstatic int allocations=0,')
    source=source.replace('int cudaFree(void*p){if(p){', 'int cudaFree(void*p){if(p){const int call=freeCalls++;if(call==freeFailure||call==freeFailure2||p==freeTarget)return 7;')
    return source.split(' const int rc=configure();')[0]


@pytest.mark.parametrize('already_configured',[False,True])
@pytest.mark.parametrize('copy_failure',[0,8])
def test_failed_wall_candidate_free_retains_owner_until_resident_teardown(tmp_path,already_configured,copy_failure):
    source=_free_failure_wall_source()+r'''
 if(ALREADY)assert(configure()==0);const int old=allocations;auto* input=state.gasBoundaryLayerModel.input;
 freeCalls=0;freeFailure=0;copyFailAfter=COPY;assert(configure()!=0&&state.gasModelPoisoned);
 assert(allocations==old+1&&state.gasBoundaryLayerModel.input==input);
 freeFailure=-1;releaseBoundaryLayerStorage(state);assert(allocations==0);
}
'''.replace('ALREADY','true' if already_configured else 'false').replace('COPY',str(copy_failure))
    _compile_free_probe(tmp_path,source)


def _compile_free_probe(tmp_path,source):
    (tmp_path/'probe.cpp').write_text(source)
    command=['g++','-std=c++17','-O1','-I'+str(ROOT),'-I'+str(ROOT/'common'),str(tmp_path/'probe.cpp'),'-o',str(tmp_path/'probe')]
    b=subprocess.run(command,capture_output=True,text=True);(tmp_path/'build.log').write_text(b.stdout+b.stderr);assert b.returncode==0,b.stderr
    r=subprocess.run([str(tmp_path/'probe'),'1','-1','1'],capture_output=True,text=True);(tmp_path/'run.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stdout+r.stderr


@pytest.mark.parametrize('disable',[False,True])
def test_snapshot_free_failure_stops_wall_publication_and_keeps_retryable_owner(tmp_path,disable):
    source=_free_failure_wall_source()+r'''
 assert(configure()==0);const auto* input=state.gasBoundaryLayerModel.input;const int views=publicationAttempts,old=allocations;
 assert(!allocateSharedGasZero(state.gasTrial.trial.cells,2));state.gasTrial.trial.cellScalars=2;state.gasTrial.trial.valid=true;
 auto* retained=state.gasTrial.trial.cells;retained[0]=923;freeTarget=retained;if(DISABLE)cfg.wallCount=0;
 assert(configure()!=0&&publicationAttempts==views&&state.gasBoundaryLayerModel.input==input&&!state.gasModelPoisoned);
 assert(allocations==old+1&&state.gasTrial.trial.cells==retained&&retained[0]==923&&!state.gasTrial.trial.valid);
 assert(configure()!=0&&state.gasTrial.trial.cells==retained&&allocations==old+1);
 freeTarget=nullptr;assert(configure()==0);releaseSharedGasTrialStorage(state.gasTrial);releaseBoundaryLayerStorage(state);assert(allocations==0);
}
'''.replace('DISABLE','true' if disable else 'false')
    _compile_free_probe(tmp_path,source)


def test_old_wall_free_and_candidate_free_failures_keep_both_unique_owners(tmp_path):
    source=_free_failure_wall_source()+r'''
 assert(configure()==0);freeCalls=0;freeFailure=0;freeFailure2=16;
 assert(configure()!=0&&state.gasModelPoisoned&&allocations==2);
 auto* old=state.gasBoundaryLayer.faceSlot;auto* rejected=state.gasTrial.rejectedWall.gasBoundaryLayer.faceSlot;
 assert(old&&rejected&&old!=rejected);freeFailure=freeFailure2=-1;freeTarget=rejected;
 assert(releaseBoundaryLayerStorage(state)!=0&&allocations==1&&state.gasTrial.rejectedWall.gasBoundaryLayer.faceSlot==rejected);
 assert(releaseBoundaryLayerStorage(state)!=0&&allocations==1);
 freeTarget=nullptr;assert(releaseBoundaryLayerStorage(state)==0&&allocations==0);
 std::printf("old+candidate free faults retained distinct owners; repeated teardown error retained one; final live=%d\n",allocations);
}
'''
    _compile_free_probe(tmp_path,source)


def _actual_trial_code():
    import re
    text=((VALIDATION_ROOT / 'solvers/gasUGKP/tests/test_shared_gas_trial_policy.py')).read_text()
    c=re.search("code=r'''(.*?)'''",text,re.S).group(1).split('int main(){')[0]
    names='rho rhoUx rhoUy rhoUz rhoE Ux Uy Uz p Tgas gasFluxPositivityScale gasDiffusionNumber sstSourceNumber rhoK rhoOmega k omega nut gasBoundaryRho gasBoundaryUx gasBoundaryUy gasBoundaryUz gasBoundaryP gasBoundaryT V'.split()
    c=c.replace('__DECLARATIONS__','\n'.join('double* '+n+'=nullptr;' for n in names))
    marker='int allocations=0,'
    assert marker in c
    c=c.replace(marker, 'int copyCalls=0;void* freeTarget=nullptr;int mallocCalls=0;\n'+marker,1)
    c=c.replace('int cudaFree(void*p){if(p){', 'int cudaFree(void*p){if(p==freeTarget&&p)return 7;if(p){')
    c=c.replace('int cudaMemcpy(void*d,const void*s,std::size_t n,int kind){', 'int cudaMemcpy(void*d,const void*s,std::size_t n,int kind){++copyCalls;')
    c=c.replace('int allocate(T*&p,std::size_t n,const char*){', 'int allocate(T*&p,std::size_t n,const char*){++mallocCalls;')
    return c


def test_snapshot_release_ensure_and_wall_record_resize_preserve_failed_owners(tmp_path):
    source=_actual_trial_code()+r'''
int main(){DeviceState state;auto& saved=state.gasTrial.trial;
 assert(!sharedGasTrialDetail::ensureSnapshot(&state,saved));auto* retained=saved.cells;freeTarget=retained;const int before=mallocCalls;saved.cellScalars=1;saved.valid=true;
 assert(sharedGasTrialDetail::ensureSnapshot(&state,saved)!=0&&saved.cells==retained&&allocations==1&&!saved.valid&&mallocCalls==before);
 assert(sharedGasTrialDetail::ensureSnapshot(&state,saved)!=0&&saved.cells==retained&&mallocCalls==before);
 freeTarget=nullptr;assert(!sharedGasTrialDetail::ensureSnapshot(&state,saved));assert(!releaseSharedGasTrialStorage(state.gasTrial)&&allocations==0);
 state.gasBoundaryLayer.enabled=true;state.gasBoundaryLayer.count=1;
 assert(!allocateSharedGasZero(saved.wallRecords,1));auto* bytes=saved.wallRecords;saved.wallRecordBytes=1;saved.valid=true;freeTarget=bytes;const int copies=copyCalls,mallocs=mallocCalls;
 assert(sharedGasTrialDetail::snapshotWallRecords(&state,saved,false)!=0&&saved.wallRecords==bytes&&saved.wallRecordBytes==1&&!saved.valid&&copyCalls==copies&&mallocCalls==mallocs);
 assert(sharedGasTrialDetail::snapshotWallRecords(&state,saved,false)!=0&&saved.wallRecords==bytes&&allocations==1);
 freeTarget=nullptr;assert(!releaseSharedGasTrialStorage(state.gasTrial)&&allocations==0);std::puts("ensure/wall-record failed frees perform zero replacement allocation/copy and retain owner");}
'''
    _compile_free_probe(tmp_path,source)


def test_all_snapshot_release_channels_report_actual_error_and_retain_only_failed_pointer(tmp_path):
    source=_actual_trial_code()+r'''
int main(){for(int slot=0;slot<12;++slot){DeviceState state;std::vector<void*> pointers;
 for(auto* saved:{&state.gasTrial.trial,&state.gasTrial.interval}){
  assert(!allocateSharedGasZero(saved->cells,2));assert(!allocateSharedGasZero(saved->faces,2));
  assert(!allocateSharedGasZero(saved->cellStatus,2));assert(!allocateSharedGasZero(saved->faceStatus,2));
  assert(!allocateSharedGasZero(saved->chemistryStatus,2));assert(!allocateSharedGasZero(saved->wallRecords,2));
  pointers.insert(pointers.end(),{saved->cells,saved->faces,saved->cellStatus,saved->faceStatus,saved->chemistryStatus,saved->wallRecords});saved->valid=true;
 }
 freeTarget=pointers[slot];assert(releaseSharedGasTrialStorage(state.gasTrial)!=0&&allocations==1&&!state.gasTrial.trial.valid&&!state.gasTrial.interval.valid);
 assert(releaseSharedGasTrialStorage(state.gasTrial)!=0&&allocations==1);
 freeTarget=nullptr;assert(!releaseSharedGasTrialStorage(state.gasTrial)&&allocations==0);
 }std::puts("12 actual snapshot release channels: error, repeated failed release, successful retry");}
'''
    _compile_free_probe(tmp_path,source)




def test_snapshot_free_then_candidate_cleanup_free_preserves_both_until_teardown(tmp_path):
    source=_free_failure_wall_source()+r'''
 assert(configure()==0);const int old=allocations;assert(!allocateSharedGasZero(state.gasTrial.trial.cells,2));auto* saved=state.gasTrial.trial.cells;
 freeTarget=saved;freeCalls=0;freeFailure=1;assert(configure()!=0&&state.gasModelPoisoned&&allocations==old+2&&state.gasTrial.trial.cells==saved);
 freeTarget=nullptr;freeFailure=-1;assert(!releaseSharedGasTrialStorage(state.gasTrial));assert(!releaseBoundaryLayerStorage(state)&&allocations==0);
}
'''
    _compile_free_probe(tmp_path,source)


def _allocation_then_free_fault_source():
    c=_actual_trial_code().replace('int mallocCalls=0;', 'int mallocCalls=0,failAllocationAt=2;bool retainFirstAllocation=true;')
    c=c.replace('++mallocCalls;p=n?', '++mallocCalls;if(mallocCalls==failAllocationAt){p=nullptr;return 1;}p=n?')
    c=c.replace('if(p)++allocations;return 0;', 'if(p){++allocations;if(retainFirstAllocation&&mallocCalls==1)freeTarget=p;}return 0;')
    return c


def test_snapshot_allocation_and_cleanup_free_failure_cannot_overwrite_retained_storage(tmp_path):
    source=_allocation_then_free_fault_source()+r'''
int main(){DeviceState state;auto& saved=state.gasTrial.trial;assert(sharedGasTrialDetail::ensureSnapshot(&state,saved)!=0&&allocations==1&&saved.cells==freeTarget);
 auto* retained=saved.cells;const int calls=mallocCalls;assert(sharedGasTrialDetail::ensureSnapshot(&state,saved)!=0&&saved.cells==retained&&mallocCalls==calls);
 freeTarget=nullptr;retainFirstAllocation=false;failAllocationAt=-1;assert(!sharedGasTrialDetail::ensureSnapshot(&state,saved));assert(!releaseSharedGasTrialStorage(state.gasTrial)&&allocations==0);}
'''
    _compile_free_probe(tmp_path,source)
