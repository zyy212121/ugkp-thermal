"""General-EOS SLAU2 must preserve contact, energy-reference and ALE contracts.
These checks exercise the real mixture face dispatcher. Removing scheme 8
support, using constant-gamma energy, or acoustically diffusing shear fails.
"""
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture
from test_mixture_contact_flux import CONTACT, _multispecies_fixture

@pytest.mark.parametrize('shear_sign',[-1,1])
@pytest.mark.parametrize('native',[False,True])
@pytest.mark.parametrize('mode',['stationary','forward','reverse','ale'])
def test_slau2_contact_energy_species_and_ale(tmp_path,native,mode,shear_sign):
    velocity={'stationary':0,'forward':25,'reverse':-25,'ale':25}[mode]
    geometry='''
s.gasGeometry.enabled=true;s.gasGeometry.interval=.001;
Real ov[2]={1,1},nv[2]={1,1},sweep[3]={.025,-.025,.025};
s.gasGeometry.oldVolume=ov;s.gasGeometry.newVolume=nv;s.gasGeometry.faceSweptVolume=sweep;
''' if mode=='ale' else ''
    body=r'''
int main(){State s;initialise(s);s.gasFluxScheme=8;contact(s,VELOCITY,SHEAR);GEOMETRY
Real f[5];ck(computeRiemannGasFaceFluxDevice<false>(s,0,f[0],f[1],f[2],f[3],f[4],NATIVE),"SLAU2 mixture dispatch");
const int c=VELOCITY>=0?0:1;const Real un=ALE?0:VELOCITY;
Real want[5]={s.rho[c]*un,s.rhoUx[c]*un+100000,s.rhoUy[c]*un,s.rhoUz[c]*un,(s.rhoE[c]+100000)*un+(ALE?2500000:0)};
for(int j=0;j<5;++j)ck(std::abs(f[j]-want[j])<1e-8*(1+std::abs(want[j])),"SLAU2 exact contact/ALE flux");
ck(s.gasSpecies.flux[0]+s.gasSpecies.flux[3]==f[0],"SLAU2 mass closure");
ck(std::abs(s.gasSpecies.flux[0]-s.gasSpecies.rho[c]*un)<1e-8,"SLAU2 contact species");}
'''.replace('VELOCITY',str(velocity)).replace('SHEAR',str(shear_sign)).replace('GEOMETRY',geometry).replace('NATIVE',str(native).lower()).replace('ALE','true' if mode=='ale' else 'false')
    compile_probe(tmp_path,fixture()+CONTACT+body)

@pytest.mark.parametrize('native',[False,True])
@pytest.mark.parametrize('pressure_ratio',[1,5])
def test_slau2_formation_reference_covariance(tmp_path,native,pressure_ratio):
    compile_probe(tmp_path,fixture()+CONTACT+r'''
int main(){State s;initialise(s);s.gasFluxScheme=8;contact(s,100);
for(int j=0;j<2;++j)s.gasSpecies.rho[2*j+1]*=RATIO;
s.rho[1]*=RATIO;s.rhoUx[1]*=RATIO;s.rhoUy[1]*=RATIO;s.rhoUz[1]*=RATIO;s.rhoE[1]*=RATIO;
threadIdx.x=1;recoverGasPrimitivesKernel(&s);Real a[5],b[5];
ck(computeRiemannGasFaceFluxDevice<false>(s,0,a[0],a[1],a[2],a[3],a[4],NATIVE),"base SLAU2");
Real fs[2]={s.gasSpecies.flux[0],s.gasSpecies.flux[3]};auto* coeff=const_cast<Real*>(s.gasSpecies.thermo.coefficients);coeff[2]+=3e6;coeff[5]-=5e6;
for(int c=0;c<2;++c){s.rhoE[c]+=3e6*s.gasSpecies.rho[c]-5e6*s.gasSpecies.rho[2+c];threadIdx.x=c;recoverGasPrimitivesKernel(&s);}
ck(computeRiemannGasFaceFluxDevice<false>(s,0,b[0],b[1],b[2],b[3],b[4],NATIVE),"shifted SLAU2");
for(int j=0;j<4;++j)ck(std::abs(a[j]-b[j])<1e-9*(1+std::abs(a[j])),"SLAU2 physical flux reference dependence");
Real shift=3e6*fs[0]-5e6*fs[1];ck(std::abs(b[4]-a[4]-shift)<1e-9*(1+std::abs(shift)),"SLAU2 full energy inconsistent with species");}
'''.replace('RATIO',str(pressure_ratio)).replace('NATIVE',str(native).lower()))

@pytest.mark.parametrize('count',[3,10])
def test_slau2_multispecies_contact(tmp_path,count):
    compile_probe(tmp_path,_multispecies_fixture(count)+r'''
int main(){State s;initialise(s);s.gasFluxScheme=8;
for(int c=0;c<2;++c){Real part[NS];s.rho[c]=1;s.rhoUx[c]=30;s.rhoUy[c]=c?40:-50;
for(int k=0;k<NS;++k){part[k]=k==(c?NS-1:0)?.7:.3/(NS-1);s.gasSpecies.rho[2*k+c]=part[k];}
s.rhoE[c]=ugkwp::mixtureEnergy(part,Real(700),s.gasSpecies.thermo)+.5*(900+s.rhoUy[c]*s.rhoUy[c]);threadIdx.x=c;recoverGasPrimitivesKernel(&s);}
Real f[5];ck(computeRiemannGasFaceFluxDevice<false>(s,0,f[0],f[1],f[2],f[3],f[4]),"SLAU2 multicomponent dispatch");
Real sum=0;for(int k=0;k<NS;++k){Real v=s.gasSpecies.flux[3*k];sum+=v;ck(std::abs(v-30*s.gasSpecies.rho[2*k])<1e-8,"SLAU2 multispecies upwinding");}ck(std::abs(sum-f[0])<1e-12,"SLAU2 multispecies sum");}
'''.replace('NS',str(count)))


def test_general_eos_slau2_matches_existing_gamma_law_mass_pressure(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){for(int i=0;i<240;++i){
 ugkpriemann::Primitive l{Real(.8+.03*(i%11)),Real(-60+i%117),Real(31-i%63),Real(17),Real(100000+1000*(i%37))};
 ugkpriemann::Primitive r{Real(.6+.04*(i%13)),Real(50-i%103),Real(i%73-42),Real(-11),Real(110000-1000*(i%31))};
 auto a=ugkpriemann::slau2FluxUnitNormal(l,r,Real(.6),Real(.8),Real(0),Real(1.4),Real(1e-12),Real(1e-9));
 auto b=ugkpriemann::slau2FluxUnitNormal(l,ugkpriemann::conservative(l,1.4),std::sqrt(1.4*l.p/l.rho),r,ugkpriemann::conservative(r,1.4),std::sqrt(1.4*r.p/r.rho),Real(.6),Real(.8),Real(0));
 ck(a.valid&&b.valid&&!a.usedFallback&&!b.usedFallback,"SLAU2 reference range used alternate flux");
 for(int k=0;k<5;++k)ck(std::abs(a.flux[k]-b.flux[k])<1e-11*(1+std::abs(a.flux[k])),"SLAU2 generic split differs from existing formula");
}}
''')


def test_slau2_configuration_create_code_reaches_mixture_dispatch(tmp_path):
    compile_probe(tmp_path,fixture()+CONTACT+r'''
int main(){const int code=ugkpriemann::createCodeFromScheme(ugkpriemann::Scheme::SLAU2);
ck(code==8,"SLAU2 configuration contract unexpectedly changed");
ck(ugkwp::mixtureGasFluxImplemented(code),"SLAU2 config selects an unavailable mixture producer");
ck(!ugkwp::mixtureGasFluxImplemented(7),"HLLC-ADC must not silently select SLAU2");
State s;initialise(s);s.gasFluxScheme=code;contact(s,0);Real f[5];
ck(computeRiemannGasFaceFluxDevice<false>(s,0,f[0],f[1],f[2],f[3],f[4]),"actual SLAU2 configuration code not dispatched");
ck(std::abs(f[2])<1e-6,"configured SLAU2 contact artificially transports shear");}
''')
