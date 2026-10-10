"""Host C++14 behavior checks for the application-independent thermodynamic owner.

NASA7 formulas and O2/H2O coefficients are from the pinned Cantera 3.1.0
h2o2.yaml ideal-gas phase, rather than any application-specific table:
https://raw.githubusercontent.com/Cantera/cantera/v3.1.0/data/h2o2.yaml
https://cantera.org/3.1/reference/thermo/species-thermo.html#the-nasa-7-coefficient-polynomial-parameterization
"""
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]

SOURCE = r'''
#include "common/gasTransport/MixtureThermo.H"
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <initializer_list>
#include <type_traits>
using namespace ugkwp;
using R = TEST_REAL;
static bool near(R a,R b,R relative=R(3e-6),R absolute=R(2e-5)) {
    return std::abs(a-b)<=absolute+relative*std::abs(b);
}
template<int N> struct LinearModel {
    SpeciesThermoData<R> species[N]; R coefficients[N*3];
    SpeciesThermoView<R,N> view;
    LinearModel() {
        for(int i=0;i<N;++i) {
            species[i] = SpeciesThermoData<R>();
            species[i].model=SpeciesThermoModel::LinearCp;
            species[i].coefficientOffset=3*i;
            species[i].molarMass=R(0.028);
            species[i].minTemperature=R(200);
            species[i].maxTemperature=R(4000);
            species[i].referenceTemperature=R(300);
            species[i].referenceEntropy=R(1500);
            species[i].hasEntropyReference=true;
            species[i].referencePressure=R(101325);
            coefficients[3*i]=R(1000+100*i);
            coefficients[3*i+1]=R(0.2);
            coefficients[3*i+2]=R(-2e6+1e5*i);
        }
        view.species=species; view.coefficients=coefficients;
        view.coefficientCount=3*N;
    }
};
void check_linear() {
    LinearModel<2> m;
    assert(validateSpeciesThermoView(m.view)==ThermoStatus::Success);
    const R T=R(700), gasR=universalGasConstant<R>()/R(.028);
    const R h=R(-2e6)+R(1000)*T+R(.1)*T*T;
    assert(near(speciesCp(0,T,m.view),R(1140)));
    assert(near(speciesCv(0,T,m.view),R(1140)-gasR));
    assert(near(speciesH(0,T,m.view),h));
    assert(near(speciesE(0,T,m.view),h-gasR*T));
    const R s=R(1500)+R(1000)*std::log(T/R(300))+R(.2)*(T-R(300));
    assert(near(speciesEntropyStandard(0,T,m.view),s));
    assert(near(speciesGibbsStandard(0,T,m.view),h-T*s));
    assert(!near(speciesH(0,T,m.view),speciesCp(0,T,m.view)*T));
    R masses[2]={R(.2),R(.8)};
    const R U=R(.2)*speciesE(0,T,m.view)+R(.8)*speciesE(1,T,m.view);
    assert(near(mixtureEnergy(masses,T,m.view),U));
    assert(near(mixtureHeatCapacity(masses,T,m.view),R(.2)*speciesCv(0,T,m.view)+R(.8)*speciesCv(1,T,m.view)));
    ThermoInversionControls<R> controls;
    if(sizeof(R)==4) { controls.relativeEnergyTolerance=R(2e-6); controls.absoluteTemperatureTolerance=R(.003); controls.relativeTemperatureTolerance=R(2e-6); }
    R recovered=R(1700);
    assert(invertMixtureEnergy(masses,U,m.view,controls,recovered)==ThermoStatus::Success);
    assert(near(recovered,T));
    masses[0]=R(0); masses[1]=R(1e-20);
    recovered=R(200);
    assert(invertMixtureEnergy(masses,mixtureEnergy(masses,T,m.view),m.view,controls,recovered)==ThermoStatus::Success);
    assert(near(recovered,T));
}
void check_failure() {
    LinearModel<2> m; R mass[2]={R(.1),R(.9)};
    const R U=mixtureEnergy(mass,R(700),m.view);
    ThermoInversionControls<R> controls;
    R output=R(1234.5), snapshot=output;
    mass[0]=R(-1);
    assert(invertMixtureEnergy(mass,U,m.view,controls,output)==ThermoStatus::InvalidComposition);
    assert(std::memcmp(&output,&snapshot,sizeof(R))==0);
    mass[0]=R(.1);
    assert(invertMixtureEnergy(mass,R(1e20),m.view,controls,output)==ThermoStatus::EnergyOutOfRange);
    assert(std::memcmp(&output,&snapshot,sizeof(R))==0);
    controls.maximumIterations=0;
    assert(invertMixtureEnergy(mass,U,m.view,controls,output)==ThermoStatus::InvalidControls);
    assert(output==snapshot);
    controls.maximumIterations=1; controls.absoluteEnergyTolerance=R(0); controls.relativeEnergyTolerance=R(0);
    controls.absoluteTemperatureTolerance=R(0); controls.relativeTemperatureTolerance=R(0);
    assert(invertMixtureEnergy(mass,U,m.view,controls,output)!=ThermoStatus::Success);
    assert(output==snapshot);
    assert(std::isnan(speciesCp(0,R(199),m.view)));
    assert(std::isnan(speciesE(2,R(700),m.view)));
    m.species[0].hasEntropyReference=false;
    assert(std::isfinite(speciesCp(0,R(700),m.view)));
    assert(std::isnan(speciesEntropyStandard(0,R(700),m.view)));
    assert(std::isnan(speciesGibbsStandard(0,R(700),m.view)));
    m.species[0].hasEntropyReference=true;
    m.species[0].coefficientOffset=std::numeric_limits<int>::max();
    assert(validateSpeciesThermoView(m.view)==ThermoStatus::InvalidModel);
    assert(std::isnan(speciesCp(0,R(700),m.view)));
    m.species[0].coefficientOffset=0;
    m.coefficients[0]=R(1); m.coefficients[1]=R(0);
    assert(validateSpeciesThermoView(m.view)==ThermoStatus::NonPositiveHeatCapacity);
    assert(std::isnan(speciesCv(0,R(700),m.view)));
    m.species[0].referencePressure=R(0);
    assert(validateSpeciesThermoView(m.view)==ThermoStatus::InvalidModel);
}
void check_nonfinite_validation() {
    LinearModel<1> m;
    R mass[1]={R(1)}; ThermoInversionControls<R> controls;
    R output=R(456);
    m.coefficients[1]=(std::numeric_limits<R>::max)()/R(100);
    assert(validateSpeciesThermoView(m.view)!=ThermoStatus::Success);
    assert(invertMixtureEnergy(mass,R(100),m.view,controls,output)!=ThermoStatus::Success);
    assert(output==R(456));
    m=LinearModel<1>(); m.view.species=m.species; m.view.coefficients=m.coefficients;
    m.species[0].referenceEntropy=(std::numeric_limits<R>::max)()/R(2);
    assert(std::isnan(speciesGibbsStandard(0,R(700),m.view)));
    R energy=R(11), capacity=R(22); mass[0]=R(-1);
    assert(evaluateMixtureEnergy(mass,R(700),m.view,energy,capacity)==ThermoStatus::InvalidComposition);
    assert(energy==R(11) && capacity==R(22));
}
void check_invalid_interior_inversion() {
    SpeciesThermoData<R> data;
    data.model=SpeciesThermoModel::NASA7; data.molarMass=R(.028);
    data.minTemperature=R(200); data.midTemperature=R(1000); data.maxTemperature=R(3500); data.referencePressure=R(101325);
    R coefficients[14]={R(25.9),R(-.1),R(.0001),R(0),R(0),R(0),R(0),R(3.5),R(0),R(0),R(0),R(0),R(0),R(0)};
    SpeciesThermoView<R,1> view; view.species=&data; view.coefficients=coefficients; view.coefficientCount=14;
    R mass[1]={R(1)},out=R(1200),saved=out; ThermoInversionControls<R> control;
    const R target=mixtureEnergy(mass,R(1200),view);
    assert(invertMixtureEnergy(mass,target,view,control,out)==ThermoStatus::NonPositiveHeatCapacity);
    assert(out==saved);
}
void check_nasa() {
    SpeciesThermoData<R> data[2];
    R coefficients[]={
        R(3.78245636),R(-2.99673416e-3),R(9.84730201e-6),R(-9.68129509e-9),R(3.24372837e-12),R(-1063.94356),R(3.65767573),
        R(3.28253784),R(1.48308754e-3),R(-7.57966669e-7),R(2.09470555e-10),R(-2.16717794e-14),R(-1088.45772),R(5.45323129),
        R(4.19864056),R(-2.0364341e-3),R(6.52040211e-6),R(-5.48797062e-9),R(1.77197817e-12),R(-3.02937267e4),R(-.849032208),
        R(3.03399249),R(2.17691804e-3),R(-1.64072518e-7),R(-9.7041987e-11),R(1.68200992e-14),R(-3.00042971e4),R(4.9667701)};
    for(int s=0;s<2;++s) { data[s].model=SpeciesThermoModel::NASA7; data[s].coefficientOffset=14*s;
        data[s].molarMass=s==0?R(.031998):R(.018015); data[s].minTemperature=R(200);
        data[s].midTemperature=R(1000); data[s].maxTemperature=R(3500); data[s].referencePressure=R(101325); }
    SpeciesThermoView<R,2> view; view.species=data; view.coefficients=coefficients; view.coefficientCount=28;
    assert(validateSpeciesThermoView(view)==ThermoStatus::Success);
    // Independent high-precision NASA7 evaluation, in J/mol (not J/kmol).
    assert(near(speciesCp(0,R(300),view)*data[0].molarMass,R(29.38807113248397)));
    assert(near(speciesH(0,R(300),view)*data[0].molarMass,R(54.35877860915948),R(2e-5),R(.002)));
    assert(near(speciesEntropyStandard(0,R(300),view)*data[0].molarMass,R(205.3300549002818)));
    assert(near(speciesCp(1,R(2000),view)*data[1].molarMass,R(51.75191079437037)));
    assert(near(speciesH(1,R(2000),view)*data[1].molarMass,R(-168787.93186160715)));
    assert(near(speciesEntropyStandard(1,R(2000),view)*data[1].molarMass,R(264.91577284700549)));
    for(int s=0;s<2;++s) for(R T: {R(200),R(999.9),R(1000),R(1000.1),R(2500),R(3500)}) {
        assert(near(speciesE(s,T,view),speciesH(s,T,view)-universalGasConstant<R>()/data[s].molarMass*T));
        assert(near(speciesGibbsStandard(s,T,view),speciesH(s,T,view)-T*speciesEntropyStandard(s,T,view)));
    }
    R masses[2]={R(.7),R(.3)};
    ThermoInversionControls<R> controls;
    if(sizeof(R)==4) { controls.relativeEnergyTolerance=R(2e-6); controls.absoluteTemperatureTolerance=R(.003); controls.relativeTemperatureTolerance=R(2e-6); }
    for(R T: {R(200),R(400),R(999.9),R(1000),R(1000.1),R(1700),R(3500)}) {
        R recovered=R(100); assert(invertMixtureEnergy(masses,mixtureEnergy(masses,T,view),view,controls,recovered)==ThermoStatus::Success);
        assert(near(recovered,T,R(5e-6),R(.005)));
    }
    // cp/R = 1 + ((T-500)/100)^2 - .1: positive endpoints, negative cv inside.
    coefficients[0]=R(25.9); coefficients[1]=R(-.1); coefficients[2]=R(.0001); coefficients[3]=coefficients[4]=R(0);
    assert(validateSpeciesThermoView(view)==ThermoStatus::NonPositiveHeatCapacity);
}
template<int N> void check_contract() {
    static_assert(std::is_trivially_copyable<SpeciesThermoView<R,N>>::value,"view must be device-copyable");
    static_assert(std::is_trivially_copyable<GasMechanismView<R,N>>::value,"mechanism must be device-copyable");
    static_assert(sizeof(SpeciesThermoView<R,N>)<256,"do not embed species tables in views");
    static_assert(sizeof(GasMechanismView<R,N>)<256,"do not embed reactions in views");
    LinearModel<N> model; R masses[N]; for(int i=0;i<N;++i) masses[i]=R(1)/R(N);
    assert(std::isfinite(mixtureEnergy(masses,R(800),model.view)));
    ClosedReactorInput<R,N> input; input.speciesMass[0]=R(1); input.totalMass=R(1); input.internalEnergy=R(-100); input.volume=R(.2);
    GasReactionData<R> reaction; reaction.type=GasReactionType::Troe; reaction.reversible=true; reaction.duplicate=true;
    reaction.highRate.activationEnergy=R(-1000); reaction.lowRate.activationEnergy=R(-2000);
    reaction.troe.hasT2=true; reaction.troe.T2=R(10000);
    GasMechanismView<R,N> mechanism; mechanism.reactions=&reaction; mechanism.reactionCount=1;
    assert(mechanism.reactions[0].highRate.activationEnergy<R(0));
    assert(GasMode::SingleLegacy!=GasMode::MixtureFrozen && GasMode::MixtureChemistry!=GasMode::MixtureFrozen);
}
int main(int argc,char**argv) {
    assert(argc==2);
    if(std::strcmp(argv[1],"linear")==0) check_linear();
    else if(std::strcmp(argv[1],"failure")==0) check_failure();
    else if(std::strcmp(argv[1],"nasa")==0) check_nasa();
    else if(std::strcmp(argv[1],"nonfinite")==0) check_nonfinite_validation();
    else if(std::strcmp(argv[1],"invalid_interior")==0) check_invalid_interior_inversion();
    else { check_contract<1>(); check_contract<2>(); check_contract<10>(); }
}
'''

@pytest.fixture(params=["float", "double"])
def thermo_probe(tmp_path, request):
    source = tmp_path / "probe.cpp"
    source.write_text(SOURCE)
    binary = tmp_path / "probe"
    compile_result = subprocess.run(
        ["g++", "-std=c++14", "-O2", "-Wall", "-Wextra", "-pedantic", "-DTEST_REAL="+request.param,
         "-I"+str(ROOT), str(source), "-o", str(binary)], capture_output=True, text=True)
    assert compile_result.returncode == 0, compile_result.stderr
    return binary

@pytest.mark.parametrize("scenario", ["linear", "failure", "nasa", "contract", "nonfinite", "invalid_interior"])
def test_pure_thermo_behavior(thermo_probe, scenario):
    result = subprocess.run([str(thermo_probe), scenario], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cuda_thermo_compilation_when_available(tmp_path):
    """Compile the real device functions; a host macro shim is not CUDA evidence."""
    import shutil
    nvcc = shutil.which("nvcc")
    if not nvcc:
        pytest.skip("nvcc is not installed; host tests do not prove CUDA parity")
    source = tmp_path / "thermo.cu"
    source.write_text(r'''
#include "common/gasTransport/MixtureThermo.H"
template<class Real, int N>
__device__ void evaluate(ugkwp::SpeciesThermoView<Real,N> model, Real* out) {
    Real mass[N] = {}; mass[0]=Real(1);
    Real temperature=Real(500);
    ugkwp::ThermoInversionControls<Real> controls;
    out[0]=ugkwp::speciesCp(0,temperature,model);
    out[1]=ugkwp::speciesGibbsStandard(0,temperature,model);
    out[2]=ugkwp::mixtureEnergy(mass,temperature,model);
    out[3]=Real(ugkwp::invertMixtureEnergy(mass,out[2],model,controls,temperature));
    out[4]=temperature;
}
__global__ void probe1(ugkwp::SpeciesThermoView<float,1> v,float* out) {evaluate(v,out);}
__global__ void probe2(ugkwp::SpeciesThermoView<double,2> v,double* out) {evaluate(v,out);}
__global__ void probe10(ugkwp::SpeciesThermoView<double,10> v,double* out) {evaluate(v,out);}
''')
    result = subprocess.run([nvcc, "-std=c++14", "-I"+str(ROOT), "-c", str(source),
                             "-o", str(tmp_path / "thermo.o")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
