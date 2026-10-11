"""Independent host review of relative scaling in zero-sum exchange audits."""
from pathlib import Path
import math
import subprocess
import pytest

APP = Path(__file__).resolve().parents[1]
SOURCE = r'''
#include "coupling/IntervalAudit.H"
#include <iomanip>
#include <iostream>
#include <stdexcept>
using namespace chmt;
static void check(bool value,const char* label){if(!value)throw std::runtime_error(label);}
int main(){
    PhysicsConfig p;HostState base,candidate;std::string error;WindowConservationAudit audit;
    base.gas.resize(1);base.solid.resize(1);
    base.gas[0].mass=base.gas[0].species[0]=1;base.gas[0].energy=10;
    base.solid[0].condensed[0]=1;base.solid[0].energy=20;
    CouplingInterval interval;interval.end=1;
    auto history=[&](double impulse){
        IntervalHistory result;check(result.begin(interval,error),"begin");GasIntervalRecord record;record.end=1;record.gasGeometry=2;
        ExchangePacket packet;packet.kind=ExchangeKind::GasSolid;packet.stage=1;packet.geometry=2;
        packet.face=1;packet.gasCell=packet.solidCell=0;packet.momentum.x=impulse;packet.consumerMask=ConsumeGas;
        record.packets={packet};check(result.appendAccepted(record,error),"append");return result;
    };
    auto accepted=history(90);
    base.budget.exchangeMomentum[GasParticipant].x=90;base.budget.exchangeMomentum[SolidParticipant].x=-90;
    candidate=base;candidate.time=1;candidate.budget.exchangeMomentum[GasParticipant].x=180;
    candidate.budget.exchangeMomentum[SolidParticipant].x=std::nextafter(-180.,0.);
    const double gas=180.-90.,solid=std::nextafter(-180.,0.)-(-90.);
    const double residual=gas+solid,scale=std::abs(gas)+std::abs(solid);
    const double oldTolerance=p.tolerances.absoluteMass+p.tolerances.relativeMass*std::abs(residual);
    const double newTolerance=p.tolerances.absoluteMass+p.tolerances.relativeMass*scale;
    check(auditCoupledCandidate(base,candidate,accepted,p,audit,error),"actual-transfer normalized roundoff rejected");
    std::cout<<std::setprecision(17)<<"absolute="<<p.tolerances.absoluteMass<<"\nrelative="<<p.tolerances.relativeMass
        <<"\nresidual="<<residual<<"\nscale="<<scale<<"\nold_tolerance="<<oldTolerance<<"\nnew_tolerance="<<newTolerance
        <<"\nold_normalized="<<std::abs(residual)/oldTolerance<<"\nnew_normalized="<<std::abs(residual)/newTolerance
        <<"\naudit_maximum="<<audit.maximumExchangeResidual<<'\n';
    auto strict=p;strict.tolerances.relativeMass=0;
    check(!auditCoupledCandidate(base,candidate,accepted,strict,audit,error),"absolute tolerance was silently enlarged");
    candidate.budget.exchangeMomentum[SolidParticipant].x=-180.+4e-8;
    const double largeResidual=90.+candidate.budget.exchangeMomentum[SolidParticipant].x+90.;
    check(!auditCoupledCandidate(base,candidate,accepted,p,audit,error),"beyond-relative tolerance accepted");
    std::cout<<"perturbed_residual="<<largeResidual<<"\nperturbed_audit_maximum="<<audit.maximumExchangeResidual<<'\n';
    candidate=base;candidate.time=1;candidate.budget.exchangeMomentum[GasParticipant].x=180;
    candidate.budget.exchangeMomentum[FilmParticipant].x=-90;
    check(!auditCoupledCandidate(base,candidate,accepted,p,audit,error),"wrong recipient hidden by global cancellation");
    base.budget={};candidate=base;candidate.time=1;auto zero=history(0);
    candidate.budget.exchangeMomentum[GasParticipant].x=.75e-14;
    check(auditCoupledCandidate(base,candidate,zero,strict,audit,error),"below exact absolute tolerance rejected");
    candidate.budget.exchangeMomentum[GasParticipant].x=1.25e-14;
    check(!auditCoupledCandidate(base,candidate,zero,strict,audit,error),"above exact absolute tolerance accepted");
    base.budget.exchangeMomentum[GasParticipant].x=1e12;base.budget.exchangeMomentum[SolidParticipant].x=-1e12;
    candidate=base;candidate.time=1;candidate.budget.exchangeMomentum[GasParticipant].x=1e12+90;
    candidate.budget.exchangeMomentum[SolidParticipant].x=std::nextafter(-1e12-90.,0.);
    check(!auditCoupledCandidate(base,candidate,accepted,p,audit,error),"historical cumulative budget inflated current-window tolerance");
    std::cout<<"checks=6\n";
}
'''


def test_relative_zero_sum_scaling_preserves_absolute_and_owner_checks(tmp_path):
    source=tmp_path/'review.cpp';source.write_text(SOURCE)
    binary=tmp_path/'review'
    build=subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-pedantic','-I'+str(APP),str(source),'-o',str(binary)],capture_output=True,text=True)
    assert build.returncode == 0, build.stderr
    run=subprocess.run([str(binary)],capture_output=True,text=True)
    assert run.returncode == 0, run.stdout+run.stderr
    values={key:float(value) for key,value in (line.split('=') for line in run.stdout.splitlines())}
    assert values['absolute'] == 1e-14 and values['relative'] == 1e-10
    assert values['residual'] == math.ulp(180.)
    assert values['old_normalized'] > 1 and values['new_normalized'] < 1e-5
    assert values['new_tolerance'] == pytest.approx(1e-14+1e-10*values['scale'],rel=1e-15)
    assert values['perturbed_residual'] > 2*values['new_tolerance']
    assert values['perturbed_audit_maximum'] > 1
    assert values['checks'] == 6
    print(run.stdout)
