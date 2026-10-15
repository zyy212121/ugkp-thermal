"""Roundoff closure uses populated species without suppressing physical inflow.

These compile production CUDA headers through the existing serial host shim;
they do not claim CUDA runtime coverage.
"""
from pathlib import Path
import re

import pytest

from test_mixture_state import compile_probe


def closure_fixture():
    fields = re.findall(
        r"    ([\w:]+\*) (\w+)\{\};",
        (Path(__file__).parent / "legacy_gas_fixture.hpp").read_text(),
    )
    allocations = "\n".join(
        f"s.{name}=new {kind[:-1]}[8]{{}};" for kind, name in fields
    )
    return r'''
struct State:ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>{ugkwp::GasSpeciesState<Real,3> gasSpecies;};
void ck(bool good,const char*message){if(!good){std::fprintf(stderr,"%s\n",message);std::abort();}}
bool sameBits(Real a,Real b){return std::memcmp(&a,&b,sizeof(Real))==0;}
void recover(State&s,int c){
    Real partial[3];for(int k=0;k<3;++k)partial[k]=s.gasSpecies.rho[k*2+c];
    s.rho[c]=partial[0]+partial[1]+partial[2];
    s.rhoUx[c]=s.rho[c]*s.Ux[c];
    s.rhoE[c]=ugkwp::mixtureEnergy(partial,Real(700),s.gasSpecies.thermo)+Real(.5)*s.rho[c]*s.Ux[c]*s.Ux[c];
    threadIdx.x=c;recoverGasPrimitivesKernel(&s);ck(s.gasSpecies.cellStatus[c]==0,"fixture recovery failed");
}
void initialise(State&s){
ALLOCATE
    s.nCells=2;s.nFaces=3;s.nInternalFaces=1;s.gasFluxScheme=1;s.gasReconstruction=0;
    s.gasLimiter=0;s.gasMu=0;s.gasPrClamped=.7;s.gammaGas=1.4;s.Rgas=287;s.gasCp=1004.5;s.rhoMin=1e-12;s.TgasMin=1e-9;
    auto&g=s.gasSpecies;g.mode=ugkwp::GasMode::MixtureFrozen;
    g.rho=new Real[6]{};g.initial=new Real[6]{};g.flux=new Real[9]{};
    g.gradX=new Real[6]{};g.gradY=new Real[6]{};g.gradZ=new Real[6]{};
    g.limiter=new Real[2]{};g.boundaryMassFraction=new Real[9]{};g.compositionBoundaryFixed=new int[3]{};
    g.positivityScale=new Real[6]{};g.soundSpeed=new Real[2]{};g.heatCapacity=new Real[2]{};g.gasConstant=new Real[2]{};
    g.cellStatus=new int[2]{};g.faceStatus=new int[3]{};
    auto*meta=new ugkwp::SpeciesThermoData<Real>[3];auto*coeff=new Real[9]{1000,0,-1e6,1000,0,2e6,1000,0,7e6};
    g.thermo.species=meta;g.thermo.coefficients=coeff;g.thermo.coefficientCount=9;
    g.thermoControls.relativeEnergyTolerance=Real(2e-6);g.thermoControls.relativeTemperatureTolerance=Real(2e-6);
    for(int k=0;k<3;++k){meta[k].molarMass=.028;meta[k].minTemperature=100;meta[k].maxTemperature=3000;meta[k].referencePressure=101325;meta[k].coefficientOffset=3*k;}
    for(int c=0;c<2;++c){s.V[c]=1;s.Cx[c]=c;s.Tgas[c]=700;s.cellPlaneStart[c]=2*c;s.cellPlaneCount[c]=2;
        g.rho[c]=c?Real(.2):Real(.1);g.rho[2+c]=1-g.rho[c];}
    s.cellFaceId[0]=0;s.cellFaceId[1]=1;s.cellFaceId[2]=0;s.cellFaceId[3]=2;
    for(int f=0;f<3;++f){s.faceOwner[f]=f==2?1:0;s.faceNeighbour[f]=f==0?1:-1;s.facePeriodicPair[f]=-1;
        s.magSf[f]=1;s.Sfx[f]=f==1?-1:1;s.faceCx[f]=f==0?.5:f==1?-.5:1.5;s.faceWeight[f]=Real(.37);
        s.deltaCoeffs[f]=f==0?1:2;s.gasBoundaryKind[f]=s.riemannBoundaryKind[f]=f==0?0:1;}
    blockDim.x=1;blockIdx.x=0;for(int c=0;c<2;++c)recover(s,c);
}
void face(State&s){threadIdx.x=0;computeGasInternalFaceFluxKernel<false>(&s,1e-6);ck(s.gasSpecies.faceStatus[0]==0,"valid face rejected");}
Real scalarFlux(const State&s,int k){
    const Real l=s.gasSpecies.rho[k*2],r=s.gasSpecies.rho[k*2+1];
    const Real ul=s.Ux[0],ur=s.Ux[1],al=s.gasSpecies.soundSpeed[0],ar=s.gasSpecies.soundSpeed[1];
    return s.gasFluxScheme==2
        ?ugkpriemann::hllConservativeComponentFlux(ul*l,ur*r,l,r,ul,ur,al,ar)
        :ugkpriemann::rusanovConservativeComponentFlux(ul*l,ur*r,l,r,std::max(std::abs(ul)+al,std::abs(ur)+ar));
}
'''.replace("ALLOCATE", allocations)


@pytest.mark.parametrize("scheme", [1, 2])
def test_absent_species_does_not_close_or_zero_pressure_flux(tmp_path, scheme):
    compile_probe(tmp_path, closure_fixture() + r'''
int main(){State s;initialise(s);s.gasFluxScheme=SCHEME;face(s);
    ck(s.gasSpecies.flux[6]==0,"absent species carries convective closure residual");
    const Real sum=s.gasSpecies.flux[0]+s.gasSpecies.flux[3]+s.gasSpecies.flux[6];
    ck(sum==s.gasPhiRho[0],"convective closure changed total mass");
    const Real pressure=s.gasPhiRhoUx[0];ck(pressure>0,"fixture lacks pressure flux");
    for(int c=0;c<2;++c){threadIdx.x=c;computeGasFluxPositivityScaleKernel(&s,1e-6);}
    threadIdx.x=0;applyGasFluxPositivityScaleKernel(&s);
    ck(s.gasPhiRhoUx[0]==pressure,"zero-inventory roundoff suppressed pressure flux");
}
'''.replace("SCHEME", str(scheme)))


@pytest.mark.parametrize("scheme", [1, 2])
def test_populated_last_convective_arithmetic_and_one_sided_inflow(tmp_path, scheme):
    compile_probe(tmp_path, closure_fixture() + r'''
int main(){
    for(int emptySide=-1;emptySide<2;++emptySide){State s;initialise(s);s.gasFluxScheme=SCHEME;
        for(int c=0;c<2;++c){s.gasSpecies.rho[4+c]=c==emptySide?Real(0):Real(.15+.1*c);
            s.gasSpecies.rho[2+c]=1-s.gasSpecies.rho[c]-s.gasSpecies.rho[4+c];recover(s,c);}
        face(s);const Real first=scalarFlux(s,0),second=scalarFlux(s,1);
        ck(sameBits(s.gasSpecies.flux[0],first)&&sameBits(s.gasSpecies.flux[3],second),"populated-last face arithmetic changed");
        ck(sameBits(s.gasSpecies.flux[6],s.gasPhiRho[0]-(first+second)),"populated-last closure arithmetic changed");
        if(emptySide>=0){
            ck(emptySide==0?s.gasSpecies.flux[6]<0:s.gasSpecies.flux[6]>0,"one-sided physical inflow suppressed");
            for(int c=0;c<2;++c){threadIdx.x=c;computeGasFluxPositivityScaleKernel(&s,1e-6);}
            threadIdx.x=0;applyGasFluxPositivityScaleKernel(&s);
            for(int c=0;c<2;++c){threadIdx.x=c;applyGasFluxDivergenceByCellKernel(&s,1e-6);}
            ck(s.gasSpecies.rho[4+emptySide]>0,"absent cell did not receive physical inflow");
        }
    }
}
'''.replace("SCHEME", str(scheme)))


def test_absent_species_gradient_stays_exactly_zero(tmp_path):
    compile_probe(tmp_path, closure_fixture() + r'''
int main(){State s;initialise(s);
    for(int c=0;c<2;++c){threadIdx.x=c;computeGasPrimitiveGradientsKernel(&s);
        ck(s.gasSpecies.gradX[4+c]==0&&s.gasSpecies.gradY[4+c]==0&&s.gasSpecies.gradZ[4+c]==0,"absent species carries gradient closure residual");
        ck(s.gasSpecies.gradX[c]+s.gasSpecies.gradX[2+c]==0,"populated gradient sum not closed");}
}
''')


def test_gradient_preserves_populated_last_and_one_sided_species(tmp_path):
    compile_probe(tmp_path, closure_fixture() + r'''
int main(){State s;initialise(s);s.gasSpecies.rho[5]=Real(.3);s.gasSpecies.rho[3]=1-s.gasSpecies.rho[1]-s.gasSpecies.rho[5];recover(s,1);
    for(int c=0;c<2;++c){threadIdx.x=c;computeGasPrimitiveGradientsKernel(&s);}
    const Real w=s.faceWeight[0];
    ck(s.gasSpecies.gradX[4]==(1-w)*s.gasSpecies.rho[5],"physical gradient into absent owner changed");
    Real raw[2];for(int k=0;k<2;++k)raw[k]=-(w*s.gasSpecies.rho[k*2]+(1-w)*s.gasSpecies.rho[k*2+1])+s.gasSpecies.rho[k*2+1];
    ck(sameBits(s.gasSpecies.gradX[1],raw[0])&&sameBits(s.gasSpecies.gradX[3],raw[1]),"populated-last gradient arithmetic changed");
    ck(sameBits(s.gasSpecies.gradX[5],-(raw[0]+raw[1])),"populated-last gradient closure changed");
}
''')


def test_absent_diffusion_species_zero_and_final_enthalpy(tmp_path):
    compile_probe(tmp_path, closure_fixture() + r'''
int main(){State s;initialise(s);
    for(int sample=0;sample<2;++sample){
        Real y[3]={Real(.1),1-Real(.1),0},g=sample?Real(.17):Real(.11),grad[3]={g,-g,0};
        Real D[3]={sample?Real(.43):Real(.07),Real(.01),Real(.07)},j[3],energy;
        ck(ugkwp::correctedSpeciesDiffusion(y,grad,Real(2),D,Real(0),Real(.8),Real(700),s.gasSpecies.thermo,j,energy),"valid diffusion rejected");
        ck(j[2]==0,"absent species carries diffusion closure residual");ck(j[0]+j[1]+j[2]==0,"diffusion not closed");
        Real expected=0;for(int k=0;k<3;++k)expected+=ugkwp::speciesH(k,Real(700),s.gasSpecies.thermo)*j[k];
        ck(energy==expected,"enthalpy was not computed from final corrected diffusion");
    }
}
''')


def test_diffusion_keeps_populated_last_arithmetic_and_zero_face_inflow(tmp_path):
    compile_probe(tmp_path, closure_fixture() + r'''
int main(){State s;initialise(s);Real D[3]={Real(.03),Real(.01),Real(.07)};
    for(int absent=0;absent<2;++absent){Real y[3]={Real(.1),absent?Real(.9):Real(.7),absent?Real(0):Real(.2)},grad[3]={Real(.17),Real(-.11),Real(-.06)},j[3],energy;
        ck(ugkwp::correctedSpeciesDiffusion(y,grad,Real(2),D,Real(0),Real(.8),Real(700),s.gasSpecies.thermo,j,energy),"one-sided diffusion rejected");
        Real raw[3],sum=0;for(int k=0;k<3;++k){raw[k]=-Real(2)*D[k]*grad[k];sum+=raw[k];}
        if(!absent){ck(sameBits(j[0],raw[0]-y[0]*sum)&&sameBits(j[1],raw[1]-y[1]*sum),"populated-last diffusion arithmetic changed");ck(sameBits(j[2],-(j[0]+j[1])),"populated-last diffusion closure changed");}
        else ck(j[2]==raw[2]&&j[2]>0,"zero face mass fraction erased physical diffusion");
        Real expected=0;for(int k=0;k<3;++k)expected+=ugkwp::speciesH(k,Real(700),s.gasSpecies.thermo)*j[k];
        ck(energy==expected,"final diffusion enthalpy mismatch");
    }
}
''')
