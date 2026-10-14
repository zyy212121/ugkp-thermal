import difflib,json,shutil
from r2_runner import *
old=D/'inputs/fixtures/chemistry_trial.cu';text=old.read_text()
changed=text.replace('DeviceState* s=nullptr;testMem(s,1);fixtures.push_back(s);','DeviceState* s=new DeviceState;fixtures.push_back(s);testMem(s->deviceState,1);')
changed=changed.replace('s->deviceState=s;s->nCells=2;','s->nCells=2;')
changed=changed.replace('    return s;','    cudaCheck(cudaMemcpy(s->deviceState,s,sizeof(DeviceState),cudaMemcpyHostToDevice));\n    return s;',1)
changed=changed.replace('        advanceGasChemistryKernel<<<1,32>>>(s,interval,volume,true);sync();','        cudaCheck(cudaMemcpy(s->deviceState,s,sizeof(DeviceState),cudaMemcpyHostToDevice));\n        advanceGasChemistryKernel<<<1,32>>>(s->deviceState,interval,volume,true);sync();')
changed=changed.replace('            advanceGasChemistryKernel<<<1,32>>>(s,interval,s->V,false);sync();','            advanceGasChemistryKernel<<<1,32>>>(s->deviceState,interval,s->V,false);sync();')
changed=changed.replace('s->~DeviceState();','delete s;')
assert changed!=text
new=D/'inputs/fixtures/chemistry_trial_host_device_split.cu';new.write_text(changed)
patch=D/'patches/S202-fixture-host-device-split.patch';patch.write_text(''.join(difflib.unified_diff(text.splitlines(True),changed.splitlines(True),fromfile='a/reviewer/chemistry_trial.cu',tofile='b/reviewer/chemistry_trial_host_device_split.cu')))
(D/'patches/S202-fixture-host-device-split.json').write_text(json.dumps({'classification':'TEST_SUPPORT','production_modified':False,'original_run':'CHEM-OWNER','original_source_sha256':sha(old),'new_source_sha256':sha(new),'diagnosis':'Original fixture uses a managed DeviceState as both host canonical owner and device state. WSL limited UVM cannot host-read managed state after async kernel before synchronize. CUDA-GDB shows host prepareGasTrialTransport SIGSEGV reading that managed owner at0x204c00000. Production uses separate host canonical/device published state; new fixture follows it. Arrays and real native kernels/owner/arithmetic/assertions unchanged.','verification':'Host/device split recheck required; diagnosis only until green.'},indent=2)+'\n')
with (D/'TEST_PLAN.md').open('a') as f:f.write('\nS202支持修正：原chem fixture把canonical DeviceState也放managed内存，WSL首异步kernel后host读触发SIGSEGV（CUDA-GDB已定位）。仅按真实生产host/device分离owner及同步发布，原kernel/owner/assert保持；300s复验，失败原源/二进制/日志保留。\n')
args=json.loads((D/'inputs/fixtures/chemistry-command.json').read_text());binary=ROOT/'work/fixtures/chemistry/chemistry_trial_split';args[-3]=str(new);args[-1]=str(binary)
t=run('BUILD-CHEM-OWNER-S202',cmd(*args),[10,14,26],900,'Original actual owner with fixture-only host/device ownership repair.',inputs=[new,patch],kind='BUILD')
if t['exit_code']==0:
 t=run('CHEM-OWNER-S202',cmd(binary),[10,14,26],expected='Same16 failure controls, full requested owner retry/terminal/public success and unchanged acceptance/assertions.',inputs=[new,patch],binary=binary)
 raw=(D/t['log']).read_text();assertion(t,'PASS real CUDA chemistry Trial routing' in raw and 'FAIL ' not in raw,raw,test_support='S202')
