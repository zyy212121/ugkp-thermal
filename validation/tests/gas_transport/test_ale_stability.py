"""Shared stability bounds use actual boundary calorics and turbulent diffusion."""
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture

def test_courant_uses_hot_fixed_boundary_mixture_sound_speed(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.riemannBoundaryKind[2]=0;s.riemannBoundaryPFix[2]=1;s.riemannBoundaryTFix[2]=1;s.riemannBoundaryP[2]=s.p[1];s.riemannBoundaryT[2]=1400;
s.gasSpecies.compositionBoundaryFixed[2]=1;s.gasSpecies.boundaryMassFraction[2]=.4;s.gasSpecies.boundaryMassFraction[5]=.6;
const Real y[2]={Real(.4),Real(.6)};const Real R=ugkwp::mixtureGasConstant(y,s.gasSpecies.thermo),cv=ugkwp::mixtureHeatCapacity(y,Real(1400),s.gasSpecies.thermo);const Real expected=sqrt((cv+R)/cv*R*Real(1400));
threadIdx.x=2;computeGasCourantFieldKernel(&s,1e-4);ck(s.gasSpecies.faceStatus[2]==0,"valid hot boundary rejected");ck(std::abs(s.gasPhiRhoE[2]-expected)<Real(1e-5)*expected,"boundary Courant reused cold owner sound speed");}
''')
