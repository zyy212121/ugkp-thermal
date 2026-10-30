"""Strict-Advance heat ordering evidence, with explicitly separate HOST/CUDA scopes.

HOST tests execute production caloric/source/recovery/particle kernels, not a
stubbed full solver. The opt-in CUDA fixture calls the actual resident Advance.
"""
from decimal import Decimal, localcontext
import json
import math
import os
from pathlib import Path
import re
import subprocess

import pytest

from test_actual_particle_energy_budget import particle_fixture
from test_mixture_state import HERE, ROOT


OWNER = ROOT / "common/thermal/GpuResidentStrict.cu"


def braced_function(text, signature):
    start = text.index("{", text.index(signature))
    depth = 1
    end = start + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start + 1:end - 1]


def test_actual_advance_source_order_and_time_arguments():
    owner=OWNER.read_text()
    body = braced_function(owner, 'extern "C" int ugkwpGpuResidentStrictAdvance')
    particle = braced_function(owner, 'int advanceResidentThermalParticles(')
    call_start=body.index('    if(advanceResidentThermalParticles(')
    call_end=body.index('    UGKP_DEV_PROBE_ENTER(ProbeBoundary)',call_start)
    body=body[:call_start]+particle+body[call_end:]
    stages = ["advanceGasFluxStage", "applyGasVolumeFractionSourceKernel",
              "snapshotParticleGasCouplingStateKernel", "applyEulerianParticleMaterialHeatKernel",
              "injectBoundaryParticlesKernel", "prepareParticleDirectoryAndSchedule",
              "applyCollisionalPressureKick<true>", "clearPoissonThermalPoolKernel",
              "correctPoissonThermalizedMobileParticlesKernel", "correctPoissonThermalizedStuckParticlesKernel",
              "acceptPairedParticleHeatKernel",
              "relaxColdWall1DParticlesToResidentGasKernel", "relaxMobileParticlesToResidentGasKernelStatic",
              "settlePairedParticleHeatKernel", "trackParticlesLocalFaceWalkKernel", "launchPostTransportMomentPipeline",
              "applyCollisionalPressureKick<false>", "finaliseGasBoundaryStage"]
    positions = [body.index(name) for name in stages]
    assert positions == sorted(positions)
    stripped = re.sub(r"<<<.*?>>>", "", body, flags=re.S)
    assert body.count("applyEulerianParticleMaterialHeatKernel<<<") == 1
    assert "clearPairedParticleHeatKernel" not in body
    assert "reducePairedParticleHeatKernel" not in body
    for correction in ("Mobile", "Stuck"):
        assert re.search(r"correctPoissonThermalized"+correction+r"ParticlesKernel<true>\s*\(\s*s->deviceState,\s*0,\s*dt\s*\)", stripped)
    assert re.search(r"relaxMobileParticlesToResidentGasKernelStatic\s*\(\s*s->deviceState,\s*dt,", stripped)
    for half in ("true", "false"):
        assert f"applyCollisionalPressureKick<{half}>(s, GPU_R(0.5)*dt" in body
    assert re.search(r"advanceOneStep\s*\(\s*runTime.deltaTValue\(\),\s*runTime.value\(\)",
                     (ROOT / "applications/CHT/diluteUgkwpFoam.C").read_text())
    wrapper = braced_function((ROOT/"applications/CHT/gpu/GpuResidentStrict.H").read_text(), "void advanceOneStep(")
    assert "ugkwpGpuResidentStrictAdvance(handle_, dt, simulationTime)" in wrapper
    assert 'extern "C" int ugkwpGpuResidentStrictAdvance' in OWNER.read_text()


def cache_path_fixture():
    # Reuse storage/HOST-CUDA syntax adapters only. Replace all three linear
    # caloric adapters before including production heat and recovery bodies.
    source = particle_fixture().split("void printState", 1)[0]
    replacements = {
        "Real particleTemperatureFromSpecificEnthalpyDevice(Real hp){return hp/Real(1000);}":
            '#include "operators/particleTemperatureFromSpecificEnthalpyDevice.cuh"',
        "Real particleSpecificHeatDevice(Real){return Real(1000);}":
            '#include "operators/particleSpecificHeatDevice.cuh"',
        "Real particleSpecificEnthalpyDevice(Real t){return Real(1000)*t;}": "",
        "template<int Model>Real cellDragInverseTime(DeviceState&,int,Real,Real,Real,Real,int){return Real(1);}":
            "template<int Model>Real cellDragInverseTime(DeviceState&,int,Real,Real,Real,Real,int){std::abort();}",
        "template<class Model>Real dragInverseTimeDevice(const DeviceState&,int,Real,Real,Real,Real,Model){return Real(1);}":
            "template<class Model>Real dragInverseTimeDevice(const DeviceState&,int,Real,Real,Real,Real,Model){std::abort();}",
        "struct DeviceState:State{": """struct DeviceState:State{
 Real epsS[2]={},rhoUsx[2]={},rhoUsy[2]={},rhoUsz[2]={},rhoEs[2]={},rhoDs[2]={},rhoHp[2]={};
 Real Usx[2]={},Usy[2]={},Usz[2]={},theta[2]={},Tp[2]={},dMeanCell[2]={};
""",
    }
    for old, new in replacements.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    return source + r'''
#define GPU_RECOVERY_INLINE
#include "operators/clearSolidCell.cuh"
#include "operators/recoverPrimitivesKernel.cuh"
#include "operators/solidRecoveryFromParticleMomentsCell.cuh"
#include "operators/snapshotParticleGasCouplingStateKernel.cuh"
void record(const char* stage,const DeviceState&s){
 const double values[]={double(s.rhoE[0]),double(s.pT[0]),double(s.Tgas[0]),double(s.couplingTgasOld[0]),
 double(Real(1)-solidEpsFromMomentDevice(s,0)),double(s.rho[0]),double(s.rhoUx[0]),double(s.rhoUy[0]),double(s.rhoUz[0]),
 double(s.pm[0]),double(s.pux[0]),double(s.puy[0]),double(s.puz[0]),double(s.pTheta[0]),
 double(s.momRhoHpP[0]),double(s.momRhoEP[0]),double(particleSpecificEnthalpyDevice(s.pT[0]))};
 std::printf("\"%s\":[",stage);for(int i=0;i<17;++i)std::printf("%s%.17g",i?",":"",values[i]);std::printf("]");
}
int main(int argc,char**argv){
 const int scenario=std::atoi(argv[1]);DeviceState s;setup(s);
 s.gasSpecies.mode=ugkwp::GasMode::SingleLegacy;s.V[0]=1;s.cellPlaneCount[0]=0;
 s.rho[0]=1;s.Ux[0]=s.Uy[0]=s.Uz[0]=0;s.pux[0]=s.puy[0]=s.puz[0]=0;
 s.pTheta[0]=0;s.pT[0]=2800;s.Tgas[0]=3200;s.dragModelId=0;s.gasMu=Real(1e-5);
 s.solveParticleTemperature=1;s.particleGasHeatTransferModelId=scenario==0?0:1;
 s.gasPrOneThird=std::pow(s.gasPrClamped,Real(1)/Real(3));
 const double dt=scenario==1?0:scenario==2?1e-8:1e-3;
 if(scenario==2)s.Ux[0]=s.pux[0]=40000;
 s.rhoUx[0]=s.rho[0]*s.Ux[0];s.rhoUy[0]=s.rhoUz[0]=0;
 s.rhoE[0]=double(s.rhoUx[0])*double(s.rhoUx[0])/(2*double(s.rho[0]))
   +double(s.rho[0])*double(s.Rgas)*double(s.Tgas[0])/(double(s.gammaGas)-1);
 publish(s);solidRecoveryFromParticleMomentsCell(s,0);recoverPrimitivesKernel(&s);
 s.epsGPrev[0]=Real(1)-solidEpsFromMomentDevice(s,0);
 std::printf("{");record("before",s);
 applyGasVolumeFractionSourceKernel(&s,dt);recoverPrimitivesKernel(&s);
 std::printf(",");record("volume",s);
 if(s.particleGasHeatTransferModelId)snapshotParticleGasCouplingStateKernel(&s);
 std::printf(",");record("snapshot",s);
 if(s.particleGasHeatTransferModelId)applyEulerianParticleMaterialHeatKernel(&s,dt);
 std::printf(",");record("source",s);
 recoverPrimitivesKernel(&s);std::printf(",");record("recovered",s);
 // Strict Advance has another recovery before the collision pool. No fake
 // collision/pressure/tracking functions are called in this HOST cache proof.
 recoverPrimitivesKernel(&s);std::printf(",");record("pre_relax",s);
 const Real oldT=s.pT[0];
 const Real oldGas=s.couplingTgasOld[0],newGas=s.Tgas[0];
 const Real full=particleTemperatureAfterGasRelaxation(s,0,oldT,oldGas,Real(0),s.pd[0],
                   s.rhoSolid*particleSpecificHeatDevice(oldT),dt);
 const Real half=particleTemperatureAfterGasRelaxation(s,0,oldT,oldGas,Real(0),s.pd[0],
                   s.rhoSolid*particleSpecificHeatDevice(oldT),dt/2);
 DeviceState fresh=s;fresh.couplingTgasOld[0]=newGas;
 relaxOneParticleToResidentGas(fresh,0,dt,0);
 relaxOneParticleToResidentGas(s,0,dt,0);
 std::printf(",");record("relaxed",s);
 publish(s);solidRecoveryFromParticleMomentsCell(s,0);
 std::printf(",");record("moments",s);
 recoverPrimitivesKernel(&s);std::printf(",");record("final_recovered",s);
 std::printf(",\"dt\":%.17g,\"full_cache_T\":%.17g,\"half_cache_T\":%.17g,\"fresh_cache_T\":%.17g,",
 dt,double(full),double(half),double(fresh.pT[0]));
 std::printf("\"caloric\":[%.17g,%.17g,%.17g,%.17g,%.17g]}\n",
 double(Foam::gpuThermal::aluminaPropertyTemperatureMinK),double(Foam::gpuThermal::aluminaPropertyTemperatureMaxK),
 double(Foam::gpuThermal::aluminaSpecificHeatJkgK(Foam::gpuThermal::aluminaPropertyTemperatureMinK)),
 double(Foam::gpuThermal::aluminaSpecificHeatJkgK(Foam::gpuThermal::aluminaPropertyTemperatureMaxK)),
 double(Foam::gpuThermal::aluminaCpMolarSlopeJMolK2/Foam::gpuThermal::aluminaMolarMassKgMol));
}
'''


FIELDS = ("gas_energy", "particle_T", "gas_T", "coupling_T", "eps", "rho", "gx", "gy", "gz",
          "mass", "px", "py", "pz", "theta", "moment_heat", "moment_mechanical", "rounded_h")
STAGES = ("before", "volume", "snapshot", "source", "recovered", "pre_relax", "relaxed", "moments", "final_recovered")


@pytest.fixture()
def cache_probe(tmp_path, request):
    return compile_cache_probe(tmp_path), 64


def compile_cache_probe(tmp_path, source_root=ROOT):
    legacy = (HERE / "gas_state_probe.cpp").read_text()
    cpp = tmp_path / "cache_path.cpp"
    source = cache_path_fixture()
    if source_root != ROOT:
        source = source.replace("GasStateView<Real,Time,ugkwp::SstCoefficients,double>",
                                "GasStateView<Real,Time,ugkwp::SstCoefficients>")
        source = source.replace("s.rhoE=new double[8]{};", "s.rhoE=new Real[8]{};")
        source = source.replace("s.rhoENext=new double[8]{};", "s.rhoENext=new Real[8]{};")
    cpp.write_text(legacy[:legacy.index("void hashBytes")] + source)
    exe = tmp_path / "cache_path"
    command = ["g++", "-std=c++17", "-O0", "-I"+str(HERE), "-I"+str(source_root/"common"), "-I"+str(source_root/"common/gasNumerics"),
               str(cpp), "-o", str(exe)]
    run = subprocess.run(command, text=True, capture_output=True)
    assert run.returncode == 0, run.stdout + run.stderr
    return exe


def observe_cache(probe, scenario):
    exe, _ = probe
    run = subprocess.run([str(exe), str(scenario)], text=True, capture_output=True)
    assert run.returncode == 0, run.stdout + run.stderr
    result = json.loads(run.stdout)
    for stage in STAGES:
        result[stage] = dict(zip(FIELDS, result[stage]))
    return result


def caloric_inventory(state, coefficients):
    """Independent Decimal evaluation of the production piecewise caloric law.

    Uses stored temperature and the actual precision-specific material constants;
    does not substitute an intended source for a published particle increment.
    """
    with localcontext() as context:
        context.prec = 70
        d = lambda x: Decimal.from_float(float(x))
        low, high, cp0, cp1, slope = map(d, coefficients)
        temperature = d(state["particle_T"])
        if temperature <= low:
            h = cp0*temperature
        elif temperature <= high:
            offset = temperature-low
            h = cp0*low+cp0*offset+slope*offset*offset/2
        else:
            span = high-low
            h = cp0*low+cp0*span+slope*span*span/2+cp1*(temperature-high)
        mechanical = d(state["mass"])*(sum(d(state["p"+axis])**2 for axis in "xyz")/2
                                      + Decimal("1.5")*d(state["theta"]))
        return d(state["eps"])*d(state["gas_energy"]) + mechanical + d(state["mass"])*h


@pytest.mark.parametrize("scenario", [0, 1], ids=["disabled_heat", "zero_time"])
def test_closed_cache_path_zero_change_preserves_actual_energy(cache_probe, scenario):
    o = observe_cache(cache_probe, scenario)
    for stage in STAGES:
        assert o[stage]["gas_energy"] == o["before"]["gas_energy"]
        assert o[stage]["particle_T"] == o["before"]["particle_T"]
        assert caloric_inventory(o[stage], o["caloric"]) == caloric_inventory(o["before"], o["caloric"])


@pytest.mark.parametrize("scenario", [2, 3], ids=["micro_heat", "finite_heat"])
def test_recovery_does_not_replace_the_actual_particle_coupling_cache(cache_probe, scenario):
    o = observe_cache(cache_probe, scenario)
    before, source, recovered, after = (o[k] for k in ("before", "source", "recovered", "final_recovered"))
    assert o["volume"]["gas_energy"] == before["gas_energy"]
    assert source["particle_T"] == before["particle_T"]
    for stage in STAGES:
        for field in ("eps", "rho", "gx", "gy", "gz", "mass", "px", "py", "pz", "theta"):
            assert o[stage][field] == before[field]
    assert source["gas_energy"] < before["gas_energy"]
    for stage in ("recovered", "pre_relax", "relaxed", "moments", "final_recovered"):
        assert o[stage]["gas_energy"] == source["gas_energy"]
        assert o[stage]["coupling_T"] == o["snapshot"]["gas_T"]
    assert after["particle_T"] == o["full_cache_T"]
    if scenario == 3:
        assert recovered["gas_T"] < o["snapshot"]["gas_T"]
        assert after["particle_T"] != o["half_cache_T"]
        assert after["particle_T"] != o["fresh_cache_T"]
    residual = caloric_inventory(after, o["caloric"]) - caloric_inventory(before, o["caloric"])
    # Characterization, not conservation acceptance. Output is deliberately
    # retained even when a later approved pairing repair changes the residual.
    print(json.dumps({"scope": "HOST production cache path, not full Advance", "bits": cache_probe[1],
                      "scenario": scenario, "dt": o["dt"], "actual_total_delta_J": str(residual),
                      "gas_delta_J": (source["gas_energy"]-before["gas_energy"])*before["eps"],
                      "particle_delta_T": after["particle_T"]-before["particle_T"],
                      "old_coupling_T": after["coupling_T"], "recovered_gas_T": recovered["gas_T"],
                      "counterfactual_fresh_cache_particle_T": o["fresh_cache_T"]}, sort_keys=True))


def test_exact_baseline_cache_path_identifies_migration_side(tmp_path):
    baseline = os.environ.get("UGKWP_PARTICLE_BUDGET_BASELINE_ROOT")
    if not baseline:
        pytest.skip("set UGKWP_PARTICLE_BUDGET_BASELINE_ROOT to an exact old source snapshot")
    old_dir, new_dir = tmp_path/"old", tmp_path/"new"
    old_dir.mkdir(); new_dir.mkdir()
    old = compile_cache_probe(old_dir, Path(baseline))
    new = compile_cache_probe(new_dir)
    for scenario in (0, 2, 3):
        a, b = observe_cache((old, 64), scenario), observe_cache((new, 64), scenario)
        # Coupling cached initial gas T can differ in the last native ULP after
        # recovering the exact baseline's rounded gas energy. Report it too.
        for observation in (a, b):
            expected = observation["snapshot"]["gas_T"] if scenario else observation["before"]["coupling_T"]
            assert observation["final_recovered"]["coupling_T"] == expected
        da = caloric_inventory(a["final_recovered"], a["caloric"]) - caloric_inventory(a["before"], a["caloric"])
        db = caloric_inventory(b["final_recovered"], b["caloric"]) - caloric_inventory(b["before"], b["caloric"])
        if scenario == 3:
            assert da > 1 and db > 1
        print(json.dumps({"scope": "exact-baseline HOST cache path", "bits": 64, "scenario": scenario,
                          "baseline_total_delta_J": str(da), "current_total_delta_J": str(db),
                          "baseline_coupling_T": a["snapshot"]["coupling_T"],
                          "current_coupling_T": b["snapshot"]["coupling_T"]}, sort_keys=True))


def native_cuda_fixture():
    """Generate a real resident-Advance driver; never substitute numeric stages.

    Field clearing is derived from allocateFields so scratch/state allocations
    retain their production sizes. This is a one-cell cube with stationary,
    adiabatic reflecting faces and one motionless mobile parcel at its centre.
    """
    allocator = braced_function(OWNER.read_text(), "int allocateFields(DeviceState* s)")
    allocations = re.findall(r"allocate\s*\(\s*s->(\w+),\s*([^,]+),\s*\"", allocator)
    zeros = []
    for field, count in allocations:
        if field == "deviceState" or field.startswith("flatPressure"):
            continue
        zeros.append(f"if(s->{field}) CUDA_OK(cudaMemset(s->{field},0,({count})*sizeof(*s->{field}))); ")
    assert len(zeros) > 150
    namespace = ""
    return r'''
#include "GpuResidentStrict.cu"
#include <cstdio>
#include <cstdlib>
NAMESPACE
#define CHECK(x) do{if(!(x)){std::fprintf(stderr,"FAIL line %d: %s; %s\n",__LINE__,#x,lastError);std::exit(2);}}while(0)
#define CUDA_OK(x) do{const cudaError_t e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"CUDA line %d: %s\n",__LINE__,cudaGetErrorString(e));std::exit(3);}}while(0)
void zeroAllocated(DeviceState*s){
 const size_t nc=s->nCells,nf=s->nFaces,np=s->particleCapacity,nPlanes=s->nCellPlanes;
 const size_t segmentedTaskCapacity=(np+s->reductionBlockThreads-1)/s->reductionBlockThreads+2*nc+1;
ZEROS
}
__global__ void initialiseClosedCube(DeviceState*sp){
 DeviceState&s=*sp;
 if(threadIdx.x||blockIdx.x)return;
 s.V[0]=1;s.cellLength[0]=1;s.cellPlaneCount[0]=6;s.cellPlaneStart[0]=0;
 s.rho[0]=1;s.Tgas[0]=3200;s.p[0]=s.rho[0]*s.Rgas*s.Tgas[0];
 s.rhoE[0]=double(s.rho[0])*double(s.Rgas)*double(s.Tgas[0])/(double(s.gammaGas)-1);
 for(int f=0;f<6;++f){
  const int axis=f/2;const GpuReal sign=f%2?GPU_R(1):GPU_R(-1);
  s.faceOwner[f]=0;s.faceNeighbour[f]=-1;s.facePeriodicPair[f]=-1;
  s.cellFaceId[f]=f;s.cellFaceNeighbor[f]=-1;s.cellFaceKind[f]=2;
  s.cellFaceRestitution[f]=1;s.cellFaceTangential[f]=1;
  s.Sfx[f]=s.planeNx[f]=axis==0?sign:GPU_R(0);
  s.Sfy[f]=s.planeNy[f]=axis==1?sign:GPU_R(0);
  s.Sfz[f]=s.planeNz[f]=axis==2?sign:GPU_R(0);
  s.faceCx[f]=s.Sfx[f]*GPU_R(.5);s.faceCy[f]=s.Sfy[f]*GPU_R(.5);s.faceCz[f]=s.Sfz[f]*GPU_R(.5);
  s.planeD[f]=GPU_R(.5);s.magSf[f]=1;s.deltaCoeffs[f]=2;s.faceWeight[f]=1;
  s.gasBoundaryKind[f]=2;s.gasBoundaryRho[f]=1;s.gasBoundaryP[f]=s.p[0];s.gasBoundaryT[f]=3200;
 }
 *s.particleCountDevice=1;s.pStatus[0]=1;s.pCellId[0]=0;s.pStuck[0]=0;s.pStuckFaceId[0]=-1;
 s.pm[0]=100;s.pd[0]=GPU_R(.001);s.pT[0]=2800;s.pRng[0]=1234567;s.pOrigId[0]=42;
}
template<class T>double readOne(const T*p){T value=0;CHECK(copyToHost(&value,p,1,"fixture read")==0);return value;}
void printState(const char*tag,DeviceState*s){
 const double eps=readOne(s->epsGPrev);
 const double v[]={readOne(s->rhoE),readOne(s->pT),readOne(s->Tgas),readOne(s->couplingTgasOld),eps,
 readOne(s->rho),readOne(s->rhoUx),readOne(s->rhoUy),readOne(s->rhoUz),readOne(s->pm),readOne(s->pux),readOne(s->puy),readOne(s->puz),readOne(s->pTheta),
 readOne(s->momRhoHpP),readOne(s->momRhoEP),double(Foam::gpuThermal::aluminaSpecificEnthalpyJkg(GpuReal(readOne(s->pT))))};
 std::printf("\"%s\":[",tag);for(int i=0;i<17;++i)std::printf("%s%.17g",i?",":"",v[i]);std::printf("]");
}
int main(int argc,char**argv){
 const int scenario=argc>1?std::atoi(argv[1]):0;
 const int csr=argc>2?std::atoi(argv[2]):1, integrator=argc>3?std::atoi(argv[3]):1;
 const double dt=scenario==1?1e-8:1e-4;
 CUDA_OK(cudaDeviceSetLimit(cudaLimitStackSize,65536));
 DeviceState*s=new DeviceState;
 s->diagnosticOwner=new ugkp::diagnostics::GpuOwner;
 CUDA_OK(s->diagnosticOwner->initialise());s->diagnostics=s->diagnosticOwner->device();
 s->nCells=1;s->nFaces=6;s->nInternalFaces=0;s->nCellPlanes=6;s->particleCapacity=1;
 s->particlesMayBePresent=true;s->dragModelId=0;s->gravityEnabled=0;s->collisionalPressureEnabled=0;
 s->jammingPressureEnabled=0;s->nBoundarySources=0;s->solveParticleTemperature=1;
 s->particleGasHeatTransferModelId=scenario==0?0:1;s->particleWallHeatTransferEnabled=0;
 s->coldWallSolidificationEnabled=0;s->coldWall2DEnabled=0;
 s->rhoSolid=1000;s->invRhoSolid=GPU_R(.001);s->gasMu=GPU_R(1e-5);
 s->gasCp=s->gammaGas*s->Rgas/(s->gammaGas-GPU_R(1));s->gasPrClamped=GPU_R(.7);
 s->gasPrOneThird=std::pow(s->gasPrClamped,GPU_R(1)/GPU_R(3));
 s->particleDiameterFallback=s->particleDiameterMin=s->particleDiameterMax=GPU_R(.001);
 s->TpMin=1;s->TpMax=10000;s->TgasMin=1;s->thetaMin=0;
 s->gasFluxScheme=s->hostGasFluxScheme=1;s->gasTimeIntegrator=s->hostGasTimeIntegrator=integrator;
 s->gasReconstruction=0;s->gasLimiter=0;s->gasGraphMode=0;s->fixedWorkBlockTuned=1;
 s->csrCellLocalPathEnabled=csr;s->csrHeavyReductionEnabled=0;s->csrHeavyReductionMode=0;
 CHECK(allocateFields(s)==0);zeroAllocated(s);s->trackingWorkGrid=s->coldWallWorkGrid=1;
 CHECK(syncDeviceState(s,"fixture initial state")==0);
 initialiseClosedCube<<<1,1>>>(s->deviceState);CUDA_OK(cudaDeviceSynchronize());
 CHECK(rebuildResidentParticleMomentsFromParticles(s,1)==0);CUDA_OK(cudaDeviceSynchronize());
 CHECK(syncDeviceState(s,"fixture rebuilt state")==0);
 double maxCo=0;CHECK(ugkwpGpuResidentStrictComputeGasCourant(s,dt,0.,&maxCo)==0);
 CHECK(maxCo>=0&&maxCo<.5);CUDA_OK(cudaDeviceSynchronize());
 std::printf("{");printState("before",s);
 std::printf(",\"carry_before\":%.17g",readOne(s->pThermalEnthalpyResidual));
 // This is the actual entire exported resident host function. No stage shim.
 CHECK(ugkwpGpuResidentStrictAdvance(s,dt,dt)==0);CUDA_OK(cudaDeviceSynchronize());
 CHECK(readOne(s->particleCountDevice)==1);CHECK(readOne(s->pStatus)==1);CHECK(readOne(s->pCellId)==0);
 CHECK(readOne(s->pOrigId)==42);CHECK(readOne(s->pStuck)==0);
 CHECK(readOne(s->pux)==0&&readOne(s->puy)==0&&readOne(s->puz)==0);
 CHECK(readOne(s->px)==0&&readOne(s->py)==0&&readOne(s->pz)==0);
 for(int f=0;f<6;++f){CHECK(readOne(s->gasPhiRho+f)==0);CHECK(readOne(s->gasPhiRhoE+f)==0);}
 CHECK(readOne(s->poissonPoolMass)==0);CHECK(readOne(s->theta)==0);
 std::printf(",");printState("after",s);
 std::printf(",\"carry_after\":%.17g",readOne(s->pThermalEnthalpyResidual));
 std::printf(",\"parameters\":[%.17g,%.17g,%.17g,%.17g,%.17g]",double(s->Rgas),double(s->gammaGas),double(s->rhoSolid),double(s->gasMu*s->gasCp/s->gasPrClamped),double(s->particleDiameterFallback));
 std::printf(",\"dt\":%.17g,\"caloric\":[%.17g,%.17g,%.17g,%.17g,%.17g]}\n",dt,
 2327.0,4000.0,
 ugkwp::ParticleCaloric::liquidCp(2327.0),
 ugkwp::ParticleCaloric::liquidCp(4000.0),
 (0.0031/0.101961276));
 releaseState(s);
}
'''.replace("NAMESPACE", namespace).replace("ZEROS", "\n".join(zeros))


def test_native_cuda_fixture_calls_unmodified_full_advance():
    source = native_cuda_fixture()
    assert '#include "GpuResidentStrict.cu"' in source
    assert "ugkwpGpuResidentStrictAdvance(s,dt,dt)" in source
    assert "allocateFields(s)" in source
    assert "#define __global__" not in source
    assert "applyEulerianParticleMaterialHeatKernel<<<" not in source
    assert "relaxOneParticleToResidentGas(" not in source


def test_opt_in_native_cuda_full_heat_advance(tmp_path):
    mode = os.environ.get("UGKWP_FULL_HEAT_CUDA", "")
    if mode not in ("compile", "run"):
        pytest.skip("opt-in only: UGKWP_FULL_HEAT_CUDA=compile or run")
    path = tmp_path / "full_heat_advance.cu"
    path.write_text(native_cuda_fixture())
    executable = tmp_path / "full_heat_advance"
    app = OWNER.parent
    nvcc = str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc")
    command = [nvcc, "-std=c++17", "-O1", "--fmad=false", "--expt-relaxed-constexpr", "--Ofast-compile=min",
               "-arch="+os.environ.get("UGKWP_CUDA_ARCH", "sm_89"), "-I"+str(app), "-I"+str(ROOT/"common"), str(path), str(app/"GpuWallEnergy64.cu"),
               "-o", str(executable)]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    if mode == "compile":
        return
    import itertools
    for scenario, csr, integrator in itertools.product((0,1,2),(0,1),(1,2,3)):
        result = subprocess.run([str(executable), str(scenario), str(csr), str(integrator)], text=True, capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr
        o = json.loads(result.stdout)
        for stage in ("before", "after"):
            o[stage] = dict(zip(FIELDS, o[stage]))
        delta = caloric_inventory(o["after"], o["caloric"]) - caloric_inventory(o["before"], o["caloric"])
        delta += Decimal.from_float(o["after"]["mass"])*Decimal.from_float(o["carry_after"])-Decimal.from_float(o["before"]["mass"])*Decimal.from_float(o["carry_before"])
        if scenario == 0:
            assert delta == 0
        else:
            assert o["after"]["gas_energy"] < o["before"]["gas_energy"]
            before,after=o["before"],o["after"]
            R,gamma,rhos,k,diameter=o["parameters"]
            low,high,cp0,cp1,slope=o["caloric"]
            cp=cp0+slope*(min(high,max(low,before["particle_T"]))-low)
            rate=12*k/(rhos*cp*diameter**2)
            cg=before["eps"]*before["rho"]*R/(gamma-1)
            cs=before["mass"]*cp
            heat=cg*cs/(cg+cs)*(before["gas_T"]-before["particle_T"])*(-math.expm1(-rate*(1+cs/cg)*o["dt"]))
            expected_gas=before["gas_energy"]-heat/before["eps"]
            expected_T=before["particle_T"]+(-math.expm1(-rate*o["dt"]))*(before["gas_T"]-before["particle_T"])
            assert math.isclose(after["gas_energy"],expected_gas,rel_tol=2e-14,abs_tol=1e-9)
            assert math.isclose(after["particle_T"],expected_T,rel_tol=2e-15,abs_tol=2e-12)
            expected=dict(after,particle_T=expected_T,gas_energy=expected_gas)
            expected_delta=caloric_inventory(expected,o["caloric"])-caloric_inventory(before,o["caloric"])
            assert abs(delta-expected_delta)<Decimal("0.000002")
            assert o["after"]["particle_T"] > o["before"]["particle_T"] or o["carry_after"] > o["carry_before"]
        print(json.dumps({"scope": "actual CUDA full resident Advance", "bits": 64,
                          "scenario": scenario, "dt": o["dt"], "actual_total_delta_J": str(delta)}, sort_keys=True))


def test_host_fixture_uses_production_caloric_and_recovery_bodies():
    source = cache_path_fixture()
    for header in ("operators/particleSpecificHeatDevice.cuh",
                   "operators/particleTemperatureFromSpecificEnthalpyDevice.cuh",
                   "operators/snapshotParticleGasCouplingStateKernel.cuh",
                   "operators/recoverPrimitivesKernel.cuh",
                   "operators/solidRecoveryFromParticleMomentsCell.cuh"):
        assert f'#include "{header}"' in source
    assert "return Real(1000)" not in source
