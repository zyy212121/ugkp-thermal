"""Shared namespace owners must first be included outside application wrappers."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess

ROOT = SOURCE_ROOT


def test_global_dependencies_cover_boundary_layer_host_namespace(tmp_path):
    source=tmp_path/'namespace.cpp'
    source.write_text(r'''
#include "GpuGasOperatorDependencies.cuh"
namespace application_precision_wrapper {
#include "gasTransport/GasBoundaryLayerHost.H"
static_assert(ugkwp::GasStateTraits<int>::speciesCount==0,"nested namespace masks global gas traits");
static_assert(std::is_same<ugkwp::GasStateTraits<int>,::ugkwp::GasStateTraits<int>>::value,
              "state traits namespace owner must be global");
}
int main(){}
''')
    result=subprocess.run(['g++','-std=c++17','-I'+str(ROOT/'common'),str(source),'-o',str(tmp_path/'probe')],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
