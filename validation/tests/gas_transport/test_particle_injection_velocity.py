"""Execute the real injection kernel: override, fallback, schedule and mass budget."""
import ast
import subprocess
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT


def injection_fixture():
    tree=ast.parse((VALIDATION_ROOT/"tests/gas_transport/test_particle_thermal_carry_lifecycle.py").read_text())
    function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="test_actual_injection_initializes_reused_slot_without_other_solver_storage")
    assignment=next(x for x in function.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="code" for t in x.targets))
    code=ast.literal_eval(assignment.value)
    code=code.replace("bool scheduledInletFaceDevice(const DeviceState&,int){return false;}","bool scheduled=false; bool scheduledInletFaceDevice(const DeviceState&,int){return scheduled;}")
    code=code.replace("return 0;}\nunsigned long long mixSeed", "return .1;}\nunsigned long long mixSeed")
    return code.split("int main(){",1)[0]


def test_actual_injection_velocity_and_mass(tmp_path):
    code=injection_fixture()+r'''
int main(){
 blockDim.x=1;
 for(bool schedule : {false,true})for(bool specified : {false,true}) {
  scheduled=schedule;DeviceState s;
  s.particleInjectionVelocitySpecified=specified;
  s.particleInjectionVelocity[0]=.03;s.particleInjectionVelocity[1]=.1;s.particleInjectionVelocity[2]=-.02;
  s.rhoSolid=2800;s.Sfy[0]=-1;s.sourceUy[0]=.2;s.gasBoundaryUy[0]=.3;
  const double vy=specified?.1:(schedule?.3:.2);
  const double vx=specified?.03:0.,vz=specified?-.02:0.;
  const double rate=.1*2800*vy;s.sourceMassRate[0]=rate;
  const double dt=1.5/rate;
  injectBoundaryParticlesKernel(&s,dt,0.);
  assert(s.n==1&&s.pStatus[0]==1);
  assert(std::abs(s.pux[0]-vx)<1e-15&&std::abs(s.puy[0]-vy)<1e-15&&std::abs(s.puz[0]-vz)<1e-15);
  assert(std::abs(s.pm[0]+s.sourceResidualMass[0]-rate*dt)<1e-14);
  assert(std::abs(s.pm[0]*s.puy[0]-vy)<1e-15);
 }
 // Explicit zero wins over a nonzero scheduled gas velocity.
 scheduled=true;DeviceState zero;zero.particleInjectionVelocitySpecified=true;
 zero.Sfy[0]=-1;zero.gasBoundaryUy[0]=1;injectBoundaryParticlesKernel(&zero,1.,0.);assert(zero.n==0);
 // Outward override must not inject.
 DeviceState out;out.particleInjectionVelocitySpecified=true;out.particleInjectionVelocity[1]=-1;
 out.Sfy[0]=-1;injectBoundaryParticlesKernel(&out,1.,0.);assert(out.n==0);
}
'''
    cpp=tmp_path/"injection.cpp";exe=tmp_path/"injection";cpp.write_text(code)
    subprocess.run(["g++","-std=c++17","-O2","-I"+str(SOURCE_ROOT/"common"),str(cpp),"-o",str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
