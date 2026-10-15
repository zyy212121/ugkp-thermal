"""Completed publication and rounded clock metadata, without zero-progress loops."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import runpy
import subprocess
APP=(SOURCE_ROOT / 'applications/CHMT')
RUNTIME=runpy.run_path(str((VALIDATION_ROOT / 'solvers/CHMT/tests/test_coupled_runtime_transactions.py')))

def test_computed_endpoint_upload_preserves_values_and_copy_errors(tmp_path):
 RUNTIME['run_probe'](tmp_path,r'''
 Fixture f;std::unique_ptr<Backend,void(*)(Backend*)> b(createBackend(f.model,f.gas,{},GasExecutionOptions{},f.h,f.error),destroyBackend);assert(b);
 assert(beginGasWindow(*b,f.program,f.error));auto candidate=b->trial;candidate.time=123;candidate.gas[0].mass=-2;candidate.gas[0].energy=-3;candidate.gas[0].species[0]=-5;candidate.gasMesh.geometryVersion=999;
 assert(commitGasWindow(*b,candidate,f.error));assert(b->accepted.time==123&&b->accepted.gas[0].mass==-2&&b->accepted.gas[0].energy==-3);
 const auto&v=b->storage.hostView();assert(v.rho[0]==-2&&v.rhoE[0]==-3&&v.gasSpecies.rho[0]==-5);
 candidate.gas[0].energy=44;shimFailCopy=1;assert(!commitGasWindow(*b,candidate,f.error));assert(b->accepted.gas[0].energy==-3);
 ''')

def test_supplied_wall_program_skips_discarded_certificate(tmp_path):
 RUNTIME['run_probe'](tmp_path,r'''
 Fixture f;std::unique_ptr<Backend,void(*)(Backend*)> b(createBackend(f.model,f.gas,{},GasExecutionOptions{},f.h,f.error),destroyBackend);assert(b);
 f.program.interval.identity.sequence=77;f.program.surface.persistentId[0]=999;f.program.knots[1].faces[0].primaryKind=ExchangeKind::GasFilm;
 f.program.knots[0].faces[0].temperature=-9;assert(beginGasWindow(*b,f.program,f.error));
 WallKnot sample;assert(sampleGasWallProgramLocal(b->program,WindowTime{},sample,f.error));assert(sample.faces[0].temperature==-9);
 auto missing=b->program;missing.knots.clear();assert(!sampleGasWallProgramLocal(missing,WindowTime{},sample,f.error));
 ''')

def test_rounded_endpoint_and_supplied_history_are_not_refused(tmp_path):
 code=r'''
#include "coupling/IntervalAudit.H"
#include <cassert>
using namespace chmt;
int main(){std::string e;double origin=1099511627776.;CouplingControls controls;controls.interval=1e-6;controls.minimumInterval=1e-12;
 CouplingWindowController c;assert(c.begin(origin,origin+1,1,controls,e));assert(c.interval().end==origin);
 controls.interval=1;assert(c.begin(origin,origin+1,1,controls,e));assert(c.shorten(1e-6,e));assert(c.interval().end==origin);
 // Only admission and the original addition are exercised: no falsely progressing evolution loop.
 IntervalHistory h;CouplingInterval w;w.begin=w.end=origin;assert(h.begin(w,e));GasIntervalRecord r;r.begin=r.end=origin;r.gasTrace.resize(1);r.gasGradient.resize(2);r.radiationEnergy={2};
 ExchangePacket p;p.kind=ExchangeKind::SolidFilm;p.mass=9;p.consumerMask=0;r.packets={p};assert(h.appendAccepted(r,e));
 HostState state;state.time=-1;state.ledger={p};assert(finalizeIntervalLedger(h,state,e));assert(state.ledger.size()==2&&state.ledger[0].mass==9&&state.ledger[0].consumerMask==0);
 std::vector<SolidQ> solid(1);MaterialDonorReserve reserve;assert(reserve.reset(solid,{},0,e));MaterialDonorPlan plan;plan.interval.begin=7;plan.interval.end=8;
 MaterialReserveKnot a,b;a.time=8;b.time=7;a.cumulativeSolid.resize(1);b.cumulativeSolid.resize(1);plan.knots={a,b};assert(reserve.setPlan(plan,e));
 // Do not execute intentionally unordered interpolation or missing receiver indexing.
}
'''
 src=tmp_path/'interval.cpp';src.write_text(code);exe=tmp_path/'interval'
 subprocess.run(['g++','-std=c++14','-O2','-I'+str(APP), '-I'+str(VALIDATION_ROOT / 'solvers/CHMT'),str(src),'-o',str(exe)],check=True);subprocess.run([str(exe)],check=True)
