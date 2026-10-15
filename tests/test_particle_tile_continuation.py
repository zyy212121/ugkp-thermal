"""Actual current production bodies at bounded/symbolic INT_MAX ranges, no GPU."""
from pathlib import Path
import re
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[1]
FIX=ROOT/'tests/fixtures/particle_continuation'
def adapt(text):
    return text.replace('#pragma once','').replace('asm("trap;");','throw std::runtime_error("production trap");')
def run(code,tmp_path,defs=()):
    cpp=tmp_path/'probe.cpp';cpp.write_text(code);exe=tmp_path/'probe'
    cmd=['g++','-std=c++17','-O1','-g','-fsanitize=undefined','-fno-sanitize-recover=all','-I'+str(ROOT/'common'),*defs,str(cpp),'-o',str(exe)]
    q=subprocess.run(cmd,capture_output=True,text=True);assert q.returncode==0,q.stderr
    q=subprocess.run([str(exe)],capture_output=True,text=True,timeout=60);assert q.returncode==0,q.stdout+q.stderr

def test_effective_tile_matches_wide_reference(tmp_path):
    run((FIX/'tile_prefix.hpp').read_text()+(ROOT/'common/GpuHardwareReductionTile.cuh').read_text()+(FIX/'tile_helper_main.cpp').read_text(),tmp_path)

@pytest.mark.parametrize('base_only',[0,1])
def test_actual_count_materialize_clipped_descriptors(tmp_path,base_only):
    code=(FIX/'tile_prefix.hpp').read_text()+(FIX/'tile_state.hpp').read_text()
    for p in ['GpuReductionTaskCount.cuh','GpuReductionTaskWrite.cuh','GpuReductionTaskMaterialize.cuh']:code+=adapt((ROOT/'common'/p).read_text())
    run(code+(FIX/'tile_materialize_main.cpp').read_text(),tmp_path,['-DHAS_BASE='+str(base_only)])

def test_all_eight_actual_particle_worker_routes_at_intmax(tmp_path):
    code=(FIX/'tile_workers_prefix.hpp').read_text()
    for p in ['GpuCollisionPoolRange.cuh','operators/accumulateCsrSplitLogicalPoolTask.cuh','GpuCollisionPoolCell.cuh','GpuCollisionPoolSplitCell.cuh']:
        body=adapt((ROOT/'common'/p).read_text())
        body=re.sub(r'#include "(?:GpuPoolMomentOperations|GpuCollisionProbability).cuh"','',body)
        body=body.replace('__shared__ GPU_OPERATOR_REAL cellCollisionProbability;','GPU_OPERATOR_REAL cellCollisionProbability=1;').replace('extern __shared__ GPU_OPERATOR_REAL scratch[];','GPU_OPERATOR_REAL scratch[8]{};')
        code+=body
    code+='void runMoment(DeviceState& s,int begin,int end){const int c=0;\n#define GPU_MOMENT_BEGIN begin\n#define GPU_MOMENT_COUNT_TYPE int\n#define GPU_MOMENT_COUNT survivorCount\n'+(ROOT/'common/GpuParticleMomentRange.inl').read_text()+'\n}\n'
    run(code+(FIX/'tile_workers_main.cpp').read_text(),tmp_path)
