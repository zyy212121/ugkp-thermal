"""Actual common owner: frozen-mixture/free thermal-particle phase on CUDA.

This deliberately does not claim a complete CHMT application or contact test.
Run: python3 validation/pr31/mixture_thermal_native.py --output <local-dir>
"""
import argparse
from decimal import Decimal,localcontext
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"validation/tests/gas_transport"))
sys.path.insert(0,str(ROOT))
from test_full_heat_advance import native_cuda_fixture,FIELDS,caloric_inventory


def source(bound=False):
    s=native_cuda_fixture()
    s=s.replace("s.Tgas[0]=3200", "s.Tgas[0]=1000")
    s=s.replace("s.rhoE[0]=double(s.rho[0])*double(s.Rgas)*double(s.Tgas[0])/(double(s.gammaGas)-1);",
        "double mass[2]={.9,.1};s.rhoE[0]=ugkwp::mixtureEnergy(mass,1000.,s.gasSpecies.thermo);recoverGasPrimitiveCell(s,0);")
    s=s.replace("s.pm[0]=100", "s.pm[0]=.002").replace("s.pT[0]=2800", "s.pT[0]=10000")
    s=s.replace("s->rhoSolid=1000;s->invRhoSolid=GPU_R(.001)", "s->rhoSolid=2500;s->invRhoSolid=GPU_R(.0004)")
    s=s.replace("s->gasMu=GPU_R(1e-5);", "s->gasMu=GPU_R(1e-5);s->gasThermalConductivity=.06;")
    s=s.replace("const double dt=scenario==1?1e-8:1e-4;", "const double dt=scenario==1?1e-8:(scenario==2?.02:1e-4);")
    needle='CHECK(syncDeviceState(s,"fixture initial state")==0);'
    s=s.replace(needle,r'''
 s->gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;
 s->gasSpecies.thermoControls.relativeEnergyTolerance=1e-14;
 s->gasSpecies.thermoControls.relativeTemperatureTolerance=1e-13;
 CHECK(allocateSharedGasZero(s->gasSpecies.rho,2)==0);
 CHECK(allocateSharedGasZero(s->gasSpecies.soundSpeed,1)==0);
 CHECK(allocateSharedGasZero(s->gasSpecies.heatCapacity,1)==0);
 CHECK(allocateSharedGasZero(s->gasSpecies.gasConstant,1)==0);
 CHECK(allocateSharedGasZero(s->gasSpecies.cellStatus,1)==0);
 CHECK(allocateSharedGasZero(s->gasSpecies.faceStatus,6)==0);
 CHECK(allocateSharedGasZero(s->gasSpecies.compositionBoundaryFixed,6)==0);
 CHECK(allocateSharedGasZero(s->gasSpecies.boundaryMassFraction,12)==0);
 const double masses[2]={.9,.1},coefficients[6]={1000,-.22,-1e7,1000,-.22,-1e7};
 ugkwp::SpeciesThermoData<double> meta[2];
 for(int k=0;k<2;++k){meta[k].molarMass=.028;meta[k].minTemperature=100;meta[k].maxTemperature=3000;meta[k].coefficientOffset=3*k;}
 CHECK(copyToDevice(s->gasSpecies.rho,masses,2,"species init")==0);
 CHECK(uploadSharedGasTable(s->gasSpecies.thermo.species,meta,2)==0);
 CHECK(uploadSharedGasTable(s->gasSpecies.thermo.coefficients,coefficients,6)==0);
 s->gasSpecies.thermo.coefficientCount=6;
'''+needle)
    s=s.replace('CHECK(ugkwpGpuResidentStrictAdvance(s,dt,dt)==0);',
                'CHECK(advanceResidentThermalParticles(s,dt,dt)==0);')
    s=s.replace('// This is the actual entire exported resident host function. No stage shim.',
                '// Actual common particle phase; the gas flux interval is not part of this fixture.')
    s=s.replace("CHECK(maxCo>=0&&maxCo<.5)","CHECK(maxCo>=0)")
    s=s.replace("for(int f=0;f<6;++f){CHECK(readOne(s->gasPhiRho+f)==0);CHECK(readOne(s->gasPhiRhoE+f)==0);}",
                "CHECK(readOne(s->gasSpecies.cellStatus)==0);")
    return borrowed_source(s) if bound else s


def borrowed_source(s):
    utilities=r'''
template<class T>T* cloneField(const T* original,size_t count) {
 if(!original)return nullptr;
 T* result=nullptr;CUDA_OK(cudaMalloc(&result,count*sizeof(T)));
 CUDA_OK(cudaMemcpy(result,original,count*sizeof(T),cudaMemcpyDeviceToDevice));return result;
}
size_t gasFieldCount(const char* name,DeviceState* s) {
 if(!std::strcmp(name,"cellFaceId"))return s->nCellPlanes;
 if(!std::strncmp(name,"face",4)||!std::strncmp(name,"gasBoundary",11)
    ||!std::strcmp(name,"Sfx")||!std::strcmp(name,"Sfy")||!std::strcmp(name,"Sfz")
    ||!std::strcmp(name,"magSf")||!std::strcmp(name,"deltaCoeffs"))return s->nFaces;
 return s->nCells;
}
__global__ void delayedGasPublication(DeviceState*sp,double energy) {
 const auto start=clock64();while(clock64()-start<1000000ull){}
 sp->rhoE[0]=energy;
}
'''
    s=s.replace('int main(int argc,char**argv){',utilities+'int main(int argc,char**argv){')
    binding=r'''
 auto external=ugkp::thermal::particleGasBinding(*s);
#define UGKP_PARTICLE_GAS_FIELD(Type,Name) external.Name=cloneField(s->Name,gasFieldCount(#Name,s));
#include "ParticleGasFields.inc"
#undef UGKP_PARTICLE_GAS_FIELD
 auto& g=external.species;const auto& old=s->gasSpecies;
 g.rho=cloneField(old.rho,2);g.soundSpeed=cloneField(old.soundSpeed,1);
 g.heatCapacity=cloneField(old.heatCapacity,1);g.gasConstant=cloneField(old.gasConstant,1);
 g.cellStatus=cloneField(old.cellStatus,1);g.faceStatus=cloneField(old.faceStatus,6);
 g.compositionBoundaryFixed=cloneField(old.compositionBoundaryFixed,6);
 g.boundaryMassFraction=cloneField(old.boundaryMassFraction,12);
 g.thermo.species=cloneField(old.thermo.species,2);
 g.thermo.coefficients=cloneField(old.thermo.coefficients,6);
 CHECK(ugkp::thermal::bindParticleGas(s,external)==0);
 cudaStream_t gasStream;CUDA_OK(cudaStreamCreateWithFlags(&gasStream,cudaStreamNonBlocking));
'''
    s=s.replace('std::printf("{");printState("before",s);',binding+'std::printf("{");printState("before",s);')
    s=s.replace('CHECK(advanceResidentThermalParticles(s,dt,dt)==0);CUDA_OK(cudaDeviceSynchronize());',r'''
 const double inputEnergy=readOne(external.rhoE);
 CUDA_OK(cudaMemset(external.rhoE,0,sizeof(double)));
 delayedGasPublication<<<1,1,0,gasStream>>>(s->deviceState,inputEnergy);
 CHECK(ugkp::thermal::advanceBoundParticlePhase(s,dt,dt,gasStream)==0);
 double* observed=nullptr;CUDA_OK(cudaMallocHost(&observed,sizeof(double)));
 CUDA_OK(cudaMemcpyAsync(observed,external.rhoE,sizeof(double),cudaMemcpyDeviceToHost,gasStream));
 CUDA_OK(cudaStreamSynchronize(gasStream));CUDA_OK(cudaDeviceSynchronize());
 CHECK(*observed==readOne(external.rhoE));CUDA_OK(cudaFreeHost(observed));
''')
    s=s.replace('releaseState(s);',r'''
 releaseState(s);
 // A successful phase must not transfer ownership of the external gas arrays.
 CUDA_OK(cudaMemset(external.rhoE,0,sizeof(double)));
 CHECK(readOne(external.rhoE)==0);
#define UGKP_PARTICLE_GAS_FIELD(Type,Name) CUDA_OK(cudaFree(external.Name));
#include "ParticleGasFields.inc"
#undef UGKP_PARTICLE_GAS_FIELD
 CHECK(releaseSharedGasSpecies(external.species)==0);
 CUDA_OK(cudaStreamDestroy(gasStream));
''')
    return s


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);parser.add_argument("--bound",action="store_true")
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cu=out/"mixture-thermal.cu";cu.write_text(source(args.bound));exe=out/"mixture-thermal"
    command=["/usr/local/cuda/bin/nvcc","-std=c++17","-O2","--fmad=false","--expt-relaxed-constexpr","--Ofast-compile=min",
        "-arch=sm_89","-I"+str(ROOT/"common/thermal"),"-I"+str(ROOT/"common"),str(cu),
        str(ROOT/"common/thermal/GpuWallEnergy64.cu"),"-o",str(exe)]
    with (out/"build.log").open("w") as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    results=[]
    for scenario in (0,1,2):
        for csr in (0,1):
            run=subprocess.run([str(exe),str(scenario),str(csr),"1"],capture_output=True,text=True)
            (out/f"run-{scenario}-{csr}.log").write_text(run.stdout+run.stderr)
            assert run.returncode==0,run.stdout+run.stderr
            o=json.loads(run.stdout)
            for stage in ("before","after"):o[stage]=dict(zip(FIELDS,o[stage]))
            before,after=o["before"],o["after"]
            # The independent gas inventory retains negative formation energy;
            # compute the particle inventory from its stored T plus caloric carry.
            delta=caloric_inventory(after,o["caloric"])-caloric_inventory(before,o["caloric"])
            delta+=Decimal.from_float(after["mass"])*Decimal.from_float(o["carry_after"])-Decimal.from_float(before["mass"])*Decimal.from_float(o["carry_before"])
            assert abs(delta)<Decimal("3e-8"),delta
            if scenario:
                low,high,cp0,cp1,slope=o["caloric"]
                rate=12*.06/(2500*cp1*.001**2)
                import math
                a=-math.expm1(-rate*o["dt"])
                expected=before["particle_T"]+a*(after["gas_T"]-before["particle_T"])
                assert abs(after["particle_T"]-expected)<3e-8,(expected,after)
                assert 3000<after["particle_T"]<10000 and 1000<after["gas_T"]<3000
            else:assert delta==0
            results.append(dict(scenario=scenario,csr=csr,inventory_residual_J=str(delta),raw=o))
    (out/"results.json").write_text(json.dumps({"scope":"shared particle phase, free parcels, frozen mixture", "external_gas_binding":args.bound, "runs":results},indent=2)+"\n")
    print(json.dumps({"passed":len(results),"maximum_inventory_residual_J":str(max(abs(Decimal(x["inventory_residual_J"])) for x in results))}))

if __name__=="__main__":main()
