"""Actual sparse reserve arithmetic admits supplied balances without proof scans."""
from pathlib import Path
import subprocess
APP=Path(__file__).resolve().parents[1]
SOURCE=r'''
#include "coupling/IntervalContracts.H"
#include <cassert>
#include <cmath>
using namespace chmt;
GasIntervalRecord record(double begin,double end,double amount){GasIntervalRecord r;r.begin=begin;r.end=end;r.microSequence=9;ExchangePacket p;p.kind=ExchangeKind::GasSolid;p.solidCell=0;p.mass=p.species[0]=p.condensed[0]=amount;p.stage=0;p.step=999;r.packets={p};return r;}
int main(){std::string error;std::vector<SolidQ> solid(1);solid[0].condensed[0]=1;std::vector<FilmQ> film;
 MaterialDonorReserve reserve;assert(reserve.reset(solid,film,0,error));auto r=record(0,1,2);
 assert(reserve.apply(r,error));assert(reserve.solid()[0].condensed[0]==-1);
 auto credit=record(1,2,-3);assert(reserve.apply(credit,error));assert(reserve.solid()[0].condensed[0]==2);
 auto unavailable=record(2,3,1);unavailable.packets[0].solidCell=4;assert(!reserve.apply(unavailable,error));assert(reserve.solid()[0].condensed[0]==2);
 solid[0].condensed[0]=-1;assert(reserve.reset(solid,film,0,error));assert(reserve.apply(record(0,1,-2),error));assert(reserve.solid()[0].condensed[0]==1);
 // Intermediate plan deficit has no physical debit; the actual endpoint plan is applied once.
 solid[0].condensed[0]=1;assert(reserve.reset(solid,film,0,error));MaterialDonorPlan plan;plan.interval.end=3;
 for(int i=0;i<4;++i){MaterialReserveKnot k;k.time=i;k.cumulativeSolid.resize(1);k.cumulativeSolid[0].condensed[0]=i==1?-10:i==3?2:0;plan.knots.push_back(k);}
 assert(reserve.setPlan(plan,error));auto empty=record(0,2,0);empty.packets.clear();assert(reserve.apply(empty,error));
 assert(reserve.apply(record(2,3,.5),error));assert(reserve.solid()[0].condensed[0]==2.5);
 assert(reserve.work().lastPlanSamples==reserve.work().lastComponents);
 // Supplied nonfinite component is still fed through the original compensated sum.
 auto nonfinite=record(3,4,std::numeric_limits<double>::quiet_NaN());assert(reserve.apply(nonfinite,error));assert(std::isnan(reserve.solid()[0].condensed[0]));
}
'''
def test_sparse_reserve_uses_actual_endpoint_without_admission_scans(tmp_path):
    src=tmp_path/'reserve.cpp';src.write_text(SOURCE);exe=tmp_path/'reserve'
    subprocess.run(['g++','-std=c++14','-O2','-Wall','-Wextra','-Werror','-pedantic','-I'+str(APP),str(src),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
