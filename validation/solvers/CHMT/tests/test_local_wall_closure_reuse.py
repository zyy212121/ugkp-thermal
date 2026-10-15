"""Reuse only identical closure inputs within one surface temperature evaluation."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import pytest
APP = (SOURCE_ROOT / 'applications/CHMT')
ROOT = APP.parents[1]
SOURCE = r'''
#include "tests/TestSupport.H"
#include "ablation/InterfaceMath.H"
#include <cstdio>
static void* tracked=nullptr;
static unsigned calls=0;
extern "C" __attribute__((no_instrument_function)) void __cyg_profile_func_enter(void* fn,void*){if(fn==tracked)++calls;}
extern "C" __attribute__((no_instrument_function)) void __cyg_profile_func_exit(void*,void*){}
int main(){
 using namespace chmt;
 auto p=chmt_test::physics();p.gasMode=ugkwp::GasMode::MixtureFrozen;
 p.gasViscosity=.02;p.gasConductivity=.8;
 for(int s=0;s<Ns;++s){p.species[s].cp1=0;p.gasDiffusivity[s]=.01;}
 p.wallModel.family=ugkwp::gaswall::WallFamily::BoundaryLayer;
 p.wallModel.model=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport;
 const auto bulk=chmt_test::gas(p);
 GasWallClosureContext context;context.ownerDistance=.2;context.matchingDistance=.8;
 context.matchingPressure=context.mechanicalPressure=bulk.pressure;
 context.matching.temperature=bulk.temperature;context.matching.velocity[1]=10;
 for(int s=0;s<Ns;++s)context.matching.massFraction[s]=bulk.Y[s];
 context.matching.massFraction[0]=.8;context.matching.massFraction[1]=.2;
 SurfacePhysicsInput in;in.bulk=bulk;in.wallContext=context;in.area=in.gasArea=in.solidArea=2;
 in.normal=in.gasNormal=in.solidNormal={1,0,0};in.gasDistance=.2;in.solidDistance=.1;
 in.hasSolid=true;in.solidVolume=1;in.dt=.01;in.baseVelocity={0,2,0};
 in.useAcceptedFlux=true;in.useSweptGeometry=true;in.acceptedGasSpecies[0]=.000024;in.acceptedGasSpecies[1]=.000006;in.acceptedCondensed[0]=.00003;
 MaterialPrimitive material;material.temperature=500;material.conductivity=2;
 tracked=reinterpret_cast<void*>(&evaluateInterfaceGasClosure<false>);
 for(int variant=0;variant<6;++variant){
  auto input=in;auto config=p;Real temperature=600;
  if(variant==1)temperature=620;
  if(variant==2)input.wallContext.matchingDistance=.9;
  if(variant==3)config.gasConductivity=1.1;
  if(variant==4){input.acceptedGasSpecies[0]*=2;input.acceptedGasSpecies[1]*=2;input.acceptedCondensed[0]*=2;}
  if(variant==5)input.baseVelocity.y=3;
  SurfacePhysicsResult result;calls=0;
  bool ok=evaluateSurfaceTemperature(input,config,material,500,temperature,false,result);
  std::printf("%d %u %.17g %.17g %.17g %.17g\n",ok,calls,result.gasState.Y[0],result.gasEnergy,result.gasState.temperature,result.gasState.velocity.y);
  if(!ok)return 1;
 }
 return 0;
}

'''


def compile_probe(tmp_path, header=None, source_text=SOURCE):
    source = tmp_path / ("reference.cpp" if header else "cached.cpp")
    code = source_text if header is None else source_text.replace('"ablation/InterfaceMath.H"', '"'+str(header)+'"')
    source.write_text(code)
    binary = source.with_suffix("")
    subprocess.run(["g++", "-std=c++17", "-O0", "-finstrument-functions", "-fno-inline", "-I"+str(APP), '-I'+str(VALIDATION_ROOT / 'solvers/CHMT'), "-I"+str(ROOT / "common"), str(source), "-o", str(binary)], check=True)
    return subprocess.check_output([str(binary)], text=True).splitlines()


def uncached_header(tmp_path):
    header = tmp_path / "uncached.H"
    code = (APP / "ablation/InterfaceMath.H").read_text()
    code = code.replace('"../../../common/gasWall/WallModel.H"', '"'+str(ROOT / 'common/gasWall/WallModel.H')+'"')
    assert code.count("bool sameClosure=wallClosureReady") == 1
    code = code.replace("bool sameClosure=wallClosureReady", "bool sameClosure=false")
    header.write_text(code)
    return header


def bvp_probe(source_text):
    # Exercise the nonlinear BVP and its real shared workspace with SST and
    # chemistry disabled, isolating workspace reuse from extra model physics.
    source_text = source_text.replace("BoundaryLayerModel::ConstantTransport", "BoundaryLayerModel::ReactingSst")
    source_text = source_text.replace("GasWallClosureContext context;", """
 ugkwp::gaswall::WallWorkspace<Real,Ns> workspace;
 GasWallClosureContext context;context.workspace=&workspace;
""")
    return source_text.replace("  if(!ok)return 1;", r'''
  if(!ok)return 1;
  Real residual=0;
  for(int n=0;n<config.wallModel.nodes;++n)for(int j=0;j<Ns+4;++j)
   residual=maxValue(residual,absValue(workspace.residual[n][j]));
  if(!(residual<=config.wallModel.absoluteTolerance+config.wallModel.relativeTolerance))return 2;
  std::printf("workspace %.17g",residual);
  for(int n=0;n<config.wallModel.nodes;++n){
   std::printf(" %.17g",workspace.y[n]);
   for(int j=0;j<Ns+4;++j)std::printf(" %.17g %.17g",workspace.state[n][j],workspace.residual[n][j]);
  }
  std::printf("\n");
''')


def test_identical_local_closure_is_evaluated_once(tmp_path):
    cached = compile_probe(tmp_path)
    assert len(cached) == 6
    for row in cached:
        assert row.split()[:2] == ["1", "1"]


def test_reuse_preserves_uncached_output_and_does_not_cross_calls(tmp_path):
    header = uncached_header(tmp_path)
    cached = compile_probe(tmp_path)
    reference = compile_probe(tmp_path, header)
    assert len(cached) == len(reference) == 6
    for got, want in zip(cached, reference):
        assert int(want.split()[1]) > 1
        assert got.split()[0] == want.split()[0] == "1"
        assert got.split()[2:] == want.split()[2:]
    assert len(set(row.split()[3] for row in cached)) > 3


def test_bvp_reuse_preserves_profile_and_converged_residual(tmp_path):
    source_text = bvp_probe(SOURCE)
    cached = compile_probe(tmp_path, source_text=source_text)
    reference = compile_probe(tmp_path, uncached_header(tmp_path), source_text)
    assert len(cached) == len(reference) == 12
    for index in range(0, len(cached), 2):
        got, want = cached[index].split(), reference[index].split()
        assert got[:2] == ["1", "1"]
        assert want[0] == "1" and int(want[1]) > 1
        assert got[2:] == want[2:]
        # A repeated solve must neither refine the result using an old profile
        # nor produce different active workspace state or residual components.
        assert cached[index+1].startswith("workspace ")
        assert cached[index+1] == reference[index+1]


@pytest.mark.parametrize("reacting", [False, True])
def test_changing_composition_dependent_flux_recomputes_closure(tmp_path, reacting):
    source_text = SOURCE.replace("in.useAcceptedFlux=true;", "in.useAcceptedFlux=false;")
    source_text = source_text.replace("variant<6", "variant<1")
    source_text = source_text.replace(" MaterialPrimitive material;", """
 // The actual surface-reaction law changes flux with the relaxed composition.
 // Keep swept wall speed and dry-wall velocity fixed to isolate the flux key.
 in.dt=0;
 p.material.enableSurfaceReactions=true;p.material.nSurfaceReactions=1;
 auto& reaction=p.material.surfaceReactions[0];reaction.A=.00003;
 reaction.gasNu[0]=.8;reaction.gasNu[1]=.2;reaction.condensedNu[0]=-1;
 p.material.surfaceGasOrder[0][0]=1;
 MaterialPrimitive material;
""")
    if reacting:
        source_text = bvp_probe(source_text)
    cached = compile_probe(tmp_path, source_text=source_text)
    reference = compile_probe(tmp_path, uncached_header(tmp_path), source_text)
    assert len(cached) == len(reference) == (2 if reacting else 1)
    assert cached[0].split()[0] == "1"
    assert int(cached[0].split()[1]) > 1
    assert cached == reference
