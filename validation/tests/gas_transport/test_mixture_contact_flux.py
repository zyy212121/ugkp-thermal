'''Production mixture HLLC: contact/shear, energy reference and ALE contracts.
Removing the mixture HLLC dispatch or applying acoustic diffusion to shear must
fail these physical flux checks. The expected contact flux is the exact Euler
flux, not another numerical solver's result.
'''
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture

CONTACT=r'''
void contact(State&s,Real normalVelocity,Real shearSign=1){
 auto* m=const_cast<ugkwp::SpeciesThermoData<Real>*>(s.gasSpecies.thermo.species);
 auto* a=const_cast<Real*>(s.gasSpecies.thermo.coefficients);
 m[0].molarMass=.028;m[1].molarMass=.044;
 a[0]=1000;a[1]=.1;a[2]=-1e6;a[3]=1200;a[4]=.2;a[5]=2e6;
 s.gasSpecies.thermoControls.relativeEnergyTolerance=1e-12;
 s.gasSpecies.thermoControls.relativeTemperatureTolerance=1e-12;
 for(int c=0;c<2;++c){
  Real Y[2]={c?Real(.2):Real(.8),c?Real(.8):Real(.2)};
  Real T=c?Real(1100):Real(500);Real rho=Real(100000)/(ugkwp::mixtureGasConstant(Y,s.gasSpecies.thermo)*T);
  s.rho[c]=rho;s.gasSpecies.rho[c]=rho*Y[0];s.gasSpecies.rho[2+c]=rho*Y[1];
  s.rhoUx[c]=rho*normalVelocity;s.rhoUy[c]=rho*shearSign*(c?Real(-70):Real(120));s.rhoUz[c]=rho*Real(23);
  Real partial[2]={s.gasSpecies.rho[c],s.gasSpecies.rho[2+c]};
  s.rhoE[c]=ugkwp::mixtureEnergy(partial,T,s.gasSpecies.thermo)
    +(s.rhoUx[c]*s.rhoUx[c]+s.rhoUy[c]*s.rhoUy[c]+s.rhoUz[c]*s.rhoUz[c])/(2*rho);
  threadIdx.x=c;recoverGasPrimitivesKernel(&s);ck(s.gasSpecies.cellStatus[c]==0,"contact input recovery");
  s.gasGradientLimiterRho[c]=s.gasGradientLimiterP[c]=s.gasGradientLimiterUx[c]=s.gasGradientLimiterUy[c]=s.gasGradientLimiterUz[c]=s.gasSpecies.limiter[c]=1;
 }
}
'''

@pytest.mark.parametrize('native',[False,True])
@pytest.mark.parametrize('reconstruction',[0,1])
@pytest.mark.parametrize('shear_sign',[-1,1])
def test_stationary_mixture_shear_has_no_acoustic_transport(tmp_path,native,reconstruction,shear_sign):
    compile_probe(tmp_path,fixture()+CONTACT+r'''
int main(){State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;
s.gasFluxScheme=4;s.gasReconstruction=RECON;contact(s,0,SHEAR);
Real f[5];ck(computeRiemannGasFaceFluxDevice<false>(s,0,f[0],f[1],f[2],f[3],f[4],NATIVE),"mixture HLLC selected but not evaluated");
ck(!ugkp::diagnostics::hasFatal(&d),"valid contact latched fatal");
ck(std::abs(f[0])<1e-9&&std::abs(f[2])<1e-6&&std::abs(f[3])<1e-6,"stationary contact/shear acoustic diffusion");
ck(std::abs(f[1]-100000)<1e-6,"contact pressure force");ck(std::abs(f[4])<1e-3,"stationary formation/total energy diffusion");
ck(std::abs(s.gasSpecies.flux[0])<1e-9&&s.gasSpecies.flux[0]+s.gasSpecies.flux[3]==f[0],"stationary species closure");
}
'''.replace('RECON',str(reconstruction)).replace('SHEAR',str(shear_sign)).replace('NATIVE','true'if native else'false'))

@pytest.mark.parametrize('velocity',[-300,-1,1,300])
def test_moving_mixture_contact_uses_upwind_total_energy_and_species(tmp_path,velocity):
    compile_probe(tmp_path,fixture()+CONTACT+r'''
int main(){State s;initialise(s);s.gasFluxScheme=4;contact(s,VELOCITY);Real f[5];
ck(computeRiemannGasFaceFluxDevice<false>(s,0,f[0],f[1],f[2],f[3],f[4]),"moving mixture HLLC dispatch");
const int c=VELOCITY>0?0:1;const Real u=VELOCITY;const Real want[5]={s.rho[c]*u,s.rhoUx[c]*u+100000,s.rhoUy[c]*u,s.rhoUz[c]*u,(s.rhoE[c]+100000)*u};
for(int k=0;k<5;++k)ck(std::abs(f[k]-want[k])<1e-10*(1+std::abs(want[k])),"moving contact exact Euler flux");
ck(std::abs(s.gasSpecies.flux[0]-s.gasSpecies.rho[c]*u)<1e-9,"composition not selected with contact");
ck(s.gasSpecies.flux[0]+s.gasSpecies.flux[3]==f[0],"moving contact mass closure");
}
'''.replace('VELOCITY',str(velocity)))

@pytest.mark.parametrize('velocity',[-100,100])
def test_species_energy_reference_shift_changes_only_carried_energy(tmp_path,velocity):
    compile_probe(tmp_path,fixture()+CONTACT+r'''
int main(){State s;initialise(s);s.gasFluxScheme=4;contact(s,VELOCITY);Real a[5],b[5];
ck(computeRiemannGasFaceFluxDevice<false>(s,0,a[0],a[1],a[2],a[3],a[4]),"base contact");
Real species[2]={s.gasSpecies.flux[0],s.gasSpecies.flux[3]};
auto* coeff=const_cast<Real*>(s.gasSpecies.thermo.coefficients);coeff[2]+=3e6;coeff[5]-=5e6;
for(int c=0;c<2;++c){s.rhoE[c]+=3e6*s.gasSpecies.rho[c]-5e6*s.gasSpecies.rho[2+c];threadIdx.x=c;recoverGasPrimitivesKernel(&s);ck(s.gasSpecies.cellStatus[c]==0,"reference-shift recovery");}
ck(computeRiemannGasFaceFluxDevice<false>(s,0,b[0],b[1],b[2],b[3],b[4]),"shifted contact");
for(int k=0;k<4;++k)ck(std::abs(a[k]-b[k])<1e-9*(1+std::abs(a[k])),"energy reference changed physical flux");
Real shift=3e6*species[0]-5e6*species[1];ck(std::abs((b[4]-a[4])-shift)<1e-9*(1+std::abs(shift)),"formation offsets not carried by species flux");
}
'''.replace('VELOCITY',str(velocity)))

def test_reversed_face_orientation_is_conservative(tmp_path):
    compile_probe(tmp_path,fixture()+CONTACT+r'''
int main(){State s;initialise(s);s.gasFluxScheme=4;contact(s,50);Real a[5],b[5];
ck(computeRiemannGasFaceFluxDevice<false>(s,0,a[0],a[1],a[2],a[3],a[4]),"forward face");Real fa=s.gasSpecies.flux[0];
s.faceOwner[0]=1;s.faceNeighbour[0]=0;s.Sfx[0]=-1;
ck(computeRiemannGasFaceFluxDevice<false>(s,0,b[0],b[1],b[2],b[3],b[4]),"reverse face");
for(int k=0;k<5;++k)ck(std::abs(a[k]+b[k])<1e-10*(1+std::abs(a[k])),"orientation changed conserved interface flux");
ck(std::abs(fa+s.gasSpecies.flux[0])<1e-9,"species orientation antisymmetry");}
''')

def test_contact_comoving_ale_has_pressure_work_and_no_relative_transport(tmp_path):
    compile_probe(tmp_path,fixture()+CONTACT+r'''
int main(){State s;initialise(s);s.gasFluxScheme=4;contact(s,25);
Real oldV[2]={1,1},newV[2]={1,1},sweep[3]={.025,-.025,.025};
s.gasGeometry.enabled=true;s.gasGeometry.interval=.001;s.gasGeometry.oldVolume=oldV;s.gasGeometry.newVolume=newV;s.gasGeometry.faceSweptVolume=sweep;
Real f[5];ck(computeRiemannGasFaceFluxDevice<false>(s,0,f[0],f[1],f[2],f[3],f[4]),"ALE contact face");
ck(std::abs(f[0])<1e-9&&std::abs(f[2])<1e-6&&std::abs(f[3])<1e-6,"ALE relative contact transported");
ck(std::abs(f[1]-100000)<1e-6&&std::abs(f[4]-2500000)<1e-3,"ALE pressure work not restored");
ck(s.gasSpecies.flux[0]+s.gasSpecies.flux[3]==f[0],"ALE species closure");}
''')

def _multispecies_fixture(count):
    source=fixture().replace('GasSpeciesState<Real,2>','GasSpeciesState<Real,NCOUNT>')
    for field in ['rho','initial','gradX','gradY','gradZ','positivityScale']:
        source=source.replace(f'g.{field}=new Real[4]{{}}',f'g.{field}=new Real[2*NCOUNT]{{}}')
    for field in ['flux','boundaryMassFraction']:
        source=source.replace(f'g.{field}=new Real[6]{{}}',f'g.{field}=new Real[3*NCOUNT]{{}}')
    source=source.replace('SpeciesThermoData<Real>[2]','SpeciesThermoData<Real>[NCOUNT]')
    source=source.replace('new Real[6]{1000,0,-1e6,1000,0,2e6}','new Real[3*NCOUNT]{}').replace('g.thermo.coefficientCount=6','g.thermo.coefficientCount=3*NCOUNT')
    source=source.replace('for(int k=0;k<2;++k){meta[k].molarMass=.028;', 'for(int k=0;k<NCOUNT;++k){coeff[3*k]=1500;coeff[3*k+1]=.1;coeff[3*k+2]=(k-1)*1e6;meta[k].molarMass=.028;')
    # Initialise()'s first recovery needs a partial array of the full dimension.
    source=source.replace('Real rhos[2]={g.rho[c],g.rho[2+c]}','Real rhos[NCOUNT]={};for(int k=0;k<NCOUNT;++k)rhos[k]=g.rho[2*k+c]')
    return source.replace('NCOUNT',str(count))

@pytest.mark.parametrize('count',[3,10])
@pytest.mark.parametrize('reconstruction',[0,1])
def test_all_species_use_the_same_contact_mass_flux(tmp_path,count,reconstruction):
    source=_multispecies_fixture(count)
    body=r'''
int main(){State s;initialise(s);s.gasFluxScheme=4;s.gasReconstruction=NRECON;
for(int c=0;c<2;++c){Real part[NCOUNT];s.rho[c]=1;s.rhoUx[c]=30;s.rhoUy[c]=c?40:-50;
 for(int k=0;k<NCOUNT;++k){part[k]=k==(c?NCOUNT-1:0)?Real(.7):Real(.3)/(NCOUNT-1);s.gasSpecies.rho[2*k+c]=part[k];}
 s.rhoE[c]=ugkwp::mixtureEnergy(part,Real(700),s.gasSpecies.thermo)+Real(.5)*(30*30+s.rhoUy[c]*s.rhoUy[c]);threadIdx.x=c;recoverGasPrimitivesKernel(&s);ck(s.gasSpecies.cellStatus[c]==0,"multi input");}
Real f[5];ck(computeRiemannGasFaceFluxDevice<false>(s,0,f[0],f[1],f[2],f[3],f[4]),"multi contact dispatch");
Real sum=0;for(int k=0;k<NCOUNT;++k){Real fk=s.gasSpecies.flux[3*k];sum+=fk;ck(std::abs(fk-30*s.gasSpecies.rho[2*k])<1e-8,"species not contact upwind");}
ck(std::abs(sum-f[0])<1e-12,"all species mass closure");}
'''
    compile_probe(tmp_path,(source+body).replace('NCOUNT',str(count)).replace('NRECON',str(reconstruction)))


@pytest.mark.parametrize('blowing',[-.3,.3])
@pytest.mark.parametrize('native',[False,True])
def test_physical_wall_blowing_exchange_is_independent_of_interior_solver(tmp_path,blowing,native):
    compile_probe(tmp_path,fixture()+r"""
int main(){State s;initialise(s);auto&w=s.gasBoundaryLayer;
w.enabled=true;w.count=1;int faces[3]={-1,0,-1},owners[2]={0,-1},status[1]={};
ugkwp::GasBoundaryLayerExchange<Real> exchange[1];ugkwp::GasBoundaryLayerSstClosure<Real> closure[1];Real species[2]={BLOW/3,2*BLOW/3};
w.faceSlot=faces;w.ownerSlot=owners;w.status=status;w.exchange=exchange;w.sst=closure;w.speciesFlux=species;
auto&e=exchange[0];e.mass=BLOW;e.momentumX=23;e.momentumY=-7;e.momentumZ=3;e.energy=-820;e.matchingPressure=s.p[0];e.ready=e.massReady=true;
s.riemannBoundaryKind[1]=2;Real reference[5];s.gasFluxScheme=2;
ck(computeRiemannGasFaceFluxDevice<true>(s,1,reference[0],reference[1],reference[2],reference[3],reference[4],NATIVE),"baseline physical wall");
for(int scheme:{1,4}){s.gasFluxScheme=scheme;Real f[5];
ck(computeRiemannGasFaceFluxDevice<true>(s,1,f[0],f[1],f[2],f[3],f[4],NATIVE),"selected physical wall");
for(int k=0;k<5;++k)ck(f[k]==reference[k],"interior solver changed wall exchange");
ck(f[0]==BLOW&&f[4]==-820,"blowing mass/energy lost");
ck(s.gasSpecies.flux[1]==species[0]&&s.gasSpecies.flux[4]==species[1],"blowing species lost");}}
""".replace('BLOW',str(blowing)).replace('NATIVE','true'if native else'false'))


@pytest.mark.parametrize('reconstruction',[0,1])
@pytest.mark.parametrize('native',[False,True])
def test_formation_reference_covariance_across_pressure_wave(tmp_path,reconstruction,native):
    compile_probe(tmp_path,fixture()+CONTACT+r"""
int main(){State s;initialise(s);s.gasFluxScheme=4;s.gasReconstruction=RECON;contact(s,100);
// Same T and composition on the right, five times the density and pressure.
s.rho[1]*=5;s.rhoUx[1]*=5;s.rhoUy[1]*=5;s.rhoUz[1]*=5;s.rhoE[1]*=5;
for(int k=0;k<2;++k)s.gasSpecies.rho[2*k+1]*=5;
threadIdx.x=1;recoverGasPrimitivesKernel(&s);ck(s.gasSpecies.cellStatus[1]==0,"pressure-wave input");
Real a[5],b[5];ck(computeRiemannGasFaceFluxDevice<false>(s,0,a[0],a[1],a[2],a[3],a[4],NATIVE),"base pressure-wave flux");
Real species[2]={s.gasSpecies.flux[0],s.gasSpecies.flux[3]};auto* coeff=const_cast<Real*>(s.gasSpecies.thermo.coefficients);coeff[2]+=3e6;coeff[5]-=5e6;
for(int c=0;c<2;++c){s.rhoE[c]+=3e6*s.gasSpecies.rho[c]-5e6*s.gasSpecies.rho[2+c];threadIdx.x=c;recoverGasPrimitivesKernel(&s);ck(s.gasSpecies.cellStatus[c]==0,"reference-shift state");}
ck(computeRiemannGasFaceFluxDevice<false>(s,0,b[0],b[1],b[2],b[3],b[4],NATIVE),"shifted pressure-wave flux");
for(int k=0;k<4;++k)ck(std::abs(a[k]-b[k])<1e-9*(1+std::abs(a[k])),"reference changed physical pressure-wave flux");
Real shift=3e6*species[0]-5e6*species[1];ck(std::abs(b[4]-a[4]-shift)<1e-9*(1+std::abs(shift)),"pressure-wave formation flux is not species covariant");}
""".replace('RECON',str(reconstruction)).replace('NATIVE','true'if native else'false'))


@pytest.mark.parametrize('scheme',[2,4])
def test_absent_species_roundoff_cannot_erase_pressure_flux(tmp_path,scheme):
    compile_probe(tmp_path,fixture()+r"""
int main(){State s;initialise(s);s.gasFluxScheme=SCHEME;s.gasReconstruction=1;
for(int c=0;c<2;++c){s.rho[c]=1;s.rhoUx[c]=10;s.rhoUy[c]=s.rhoUz[c]=0;
 s.gasSpecies.rho[c]=c?Real(1):std::nextafter(Real(1),Real(0));s.gasSpecies.rho[2+c]=0;
 Real partial[2]={s.gasSpecies.rho[c],0};s.rhoE[c]=ugkwp::mixtureEnergy(partial,Real(700),s.gasSpecies.thermo)+50;
 threadIdx.x=c;recoverGasPrimitivesKernel(&s);ck(s.gasSpecies.cellStatus[c]==0,"legal roundoff state");
 s.gasGradientLimiterRho[c]=s.gasGradientLimiterP[c]=s.gasGradientLimiterUx[c]=s.gasGradientLimiterUy[c]=s.gasGradientLimiterUz[c]=s.gasSpecies.limiter[c]=1;}
for(int f=0;f<3;++f){threadIdx.x=f;computeGasInternalFaceFluxKernel<false>(&s,Real(.001));}
ck(s.gasSpecies.flux[3]==0,"reconstruction invented outgoing absent species");
Real mass=s.gasPhiRho[0],normalMomentum=s.gasPhiRhoUx[0];
for(int c=0;c<2;++c){threadIdx.x=c;computeGasFluxPositivityScaleKernel(&s,Real(.001));}
threadIdx.x=0;applyGasFluxPositivityScaleKernel(&s);
ck(s.gasPhiRho[0]==mass&&s.gasPhiRhoUx[0]==normalMomentum,"roundoff absent species erased pressure force");}
""".replace('SCHEME',str(scheme)))


@pytest.mark.parametrize('count',[3,10])
def test_multispecies_reconstruction_uses_one_feasible_simplex_slope(tmp_path,count):
    compile_probe(tmp_path,_multispecies_fixture(count)+r"""
int main(){State s;initialise(s);s.gasReconstruction=1;
for(int c=0;c<2;++c){Real part[NCOUNT]{};part[0]=.4;part[1]=.4;part[2]=.2;
 for(int k=0;k<NCOUNT;++k)s.gasSpecies.rho[2*k+c]=part[k];
 s.rhoE[c]=ugkwp::mixtureEnergy(part,Real(700),s.gasSpecies.thermo);
 threadIdx.x=c;recoverGasPrimitivesKernel(&s);ck(s.gasSpecies.cellStatus[c]==0,"simplex input");s.gasSpecies.limiter[c]=1;}
s.gasSpecies.gradX[0]=.4;s.gasSpecies.gradX[2]=.4;s.gasSpecies.gradX[4]=-.8;
Real y[NCOUNT];GasPrimDevice face{};bool done=false;
ck(reconstructGasMixtureFace(s,0,0,face,y,false,&done)&&done,"simplex reconstruction");
// At dx=1/2, the raw state (.6,.6,-.2) meets the simplex at alpha=1/2.
ck(std::abs(y[0]-.5)<1e-14&&std::abs(y[1]-.5)<1e-14,"feasible slope changed direction");
Real sum=0;for(int k=0;k<NCOUNT;++k){ck(y[k]>=0&&y[k]<=1,"nonphysical face fraction");sum+=y[k];if(k>=2)ck(std::abs(y[k])<1e-14,"absent species invented");}
ck(std::abs(sum-1)<1e-15,"face fractions not one simplex");}
""".replace('NCOUNT',str(count)))


@pytest.mark.parametrize('count',[3,10])
@pytest.mark.parametrize('scheme',[4,8])
@pytest.mark.parametrize('reconstruction',[0,1])
def test_upwind_absent_species_cannot_erase_momentum_or_energy(tmp_path,count,scheme,reconstruction):
    """Last-species closure must not invent outflow from an empty donor.

    Both directions, execution paths and ALE frames exercise the real face
    producer and inventory limiter. The Euler contact flux is the reference.
    """
    compile_probe(tmp_path,_multispecies_fixture(count)+r'''
int main(){
for(int direction:{1,-1})for(bool moving:{false,true})for(bool native:{false,true}){
 State s;initialise(s);s.gasFluxScheme=SCHEME;s.gasReconstruction=RECON;
 const Real relativeVelocity=direction*Real(SCHEME==4?3:5)/7;
 // Opposite absolute/relative directions detect use of the wrong ALE donor.
 const Real meshVelocity=moving?-direction*Real(1.25):Real(0);
 const Real velocity=relativeVelocity+meshVelocity;
 const int donor=direction>0?0:1;
 for(int c=0;c<2;++c){Real part[NS]{};
  part[0]=c==donor?Real(.3):Real(.2);part[1]=c==donor?Real(.7):Real(.3);
  part[NS-1]=c==donor?Real(0):Real(.5);
  s.rho[c]=1;s.rhoUx[c]=velocity;s.rhoUy[c]=s.rhoUz[c]=0;
  for(int k=0;k<NS;++k)s.gasSpecies.rho[2*k+c]=part[k];
  s.rhoE[c]=ugkwp::mixtureEnergy(part,Real(700),s.gasSpecies.thermo)+Real(.5)*velocity*velocity;
  threadIdx.x=c;recoverGasPrimitivesKernel(&s);ck(s.gasSpecies.cellStatus[c]==0,"absent-donor input recovery");
  s.gasGradientLimiterRho[c]=s.gasGradientLimiterP[c]=s.gasGradientLimiterUx[c]=s.gasGradientLimiterUy[c]=s.gasGradientLimiterUz[c]=s.gasSpecies.limiter[c]=1;
 }
 Real oldV[2]={1,1},newV[2]={1,1},sweep[3]={meshVelocity*.001,-meshVelocity*.001,meshVelocity*.001};
 if(moving){s.gasGeometry.enabled=true;s.gasGeometry.interval=.001;s.gasGeometry.oldVolume=oldV;s.gasGeometry.newVolume=newV;s.gasGeometry.faceSweptVolume=sweep;}
 threadIdx.x=0;computeGasInternalFaceFluxKernel<false>(&s,Real(.001),native);
 Real absentFlux=s.gasSpecies.flux[3*(NS-1)],mass=s.gasPhiRho[0],momentum=s.gasPhiRhoUx[0],energy=s.gasPhiRhoE[0];
 for(int c=0;c<2;++c){threadIdx.x=c;computeGasFluxPositivityScaleKernel(&s,Real(.001));}
 threadIdx.x=0;applyGasFluxPositivityScaleKernel(&s,native);
 if(absentFlux!=0||s.gasPhiRhoUx[0]!=momentum)
  std::fprintf(stderr,"scheme=%d recon=%d ns=%d direction=%d moving=%d native=%d absent=%.17g factor=%.17g momentum=%.17g limited=%.17g\n",SCHEME,RECON,NS,direction,moving,native,absentFlux,s.gasSpecies.positivityScale[2*(NS-1)+donor],momentum,s.gasPhiRhoUx[0]);
 ck(absentFlux==0,"upwind absent species acquired a closure outflow");
 ck(s.gasPhiRho[0]==mass&&s.gasPhiRhoUx[0]==momentum&&s.gasPhiRhoE[0]==energy,"absent donor erased physical mass/momentum/energy flux");
 ck(std::abs(mass-relativeVelocity)<1e-9,"wrong contact mass flux");
 ck(std::abs(momentum-(s.p[donor]+velocity*relativeVelocity))<1e-6,"wrong contact momentum flux");
 ck(std::abs(energy-(s.rhoE[donor]*relativeVelocity+s.p[donor]*velocity))<1e-4,"wrong contact total-energy flux");
 Real sum=0;for(int k=0;k<NS;++k){Real value=s.gasSpecies.flux[3*k];sum+=value;
  ck(std::abs(value-relativeVelocity*s.gasSpecies.rho[2*k+donor])<1e-9,"species did not follow the relative mass donor");
  if(s.gasSpecies.rho[2*k+donor]==0)ck(value==0,"absent species flux is not exactly zero");}
 ck(std::abs(sum-mass)<1e-15,"species mass closure failed");
}}
'''.replace('NS',str(count)).replace('SCHEME',str(scheme)).replace('RECON',str(reconstruction)))