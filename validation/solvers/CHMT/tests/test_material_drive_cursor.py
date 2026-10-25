"""Reuse material-history locations without changing half-open sample semantics."""
from validation.tools.paths import SOURCE_ROOT
import subprocess

def test_material_cursor_matches_first_record_and_slab_bounds(tmp_path):
    source=tmp_path/'cursor.cpp';binary=tmp_path/'cursor'
    source.write_text(r"""
#include "materials/MaterialDrive.H"
#include <cassert>
#include <limits>
#include <vector>
using namespace chmt;
const GasIntervalRecord* linear(const IntervalHistory& h,WindowTime time){
 for(const auto& r:h.records())if(time<r.localEnd)return &r;
 return nullptr;
}
int main(){
 const double origin=0x1p40,dt=0x1p-20;std::string error;
 CouplingInterval window;window.begin=origin;window.end=origin+256*dt;
 IntervalHistory history;assert(history.begin(window,error));
 for(int i=0;i<256;++i){GasIntervalRecord r;r.actualDt=dt;r.localBegin=WindowTime(i*dt);r.localEnd=r.localBegin+dt;
  r.begin=windowAbsolute(origin,r.localBegin);r.end=windowAbsolute(origin,r.localEnd);
  r.gasTrace.resize(1);auto& q=r.gasTrace[0];q.temperature=300+i;q.pressure=100000;q.rho=1;q.Y[0]=1;
  r.radiationReceiver={i<200?ConsumeSolid:ConsumeGas};assert(history.appendAccepted(r,error));}
 assert(history.complete());MaterialDriveCursor cached(history);
 // Exact boundaries select the next record. Fractional queries reuse the
 // preceding location; reverse/random queries must recover the first match.
 for(int i=0;i<=1024;++i){const WindowTime time(i*dt/4);assert(cached.at(time)==linear(history,time));}
 for(int i=1024;i>=0;--i){const WindowTime time(i*dt/4);assert(cached.at(time)==linear(history,time));}
 for(int i=0;i<1024;++i){const WindowTime time(((i*541)%1025)*dt/4);assert(cached.at(time)==linear(history,time));}
 for(long double value:{std::numeric_limits<long double>::quiet_NaN(),std::numeric_limits<long double>::infinity(),-std::numeric_limits<long double>::infinity()}){
  const WindowTime time(value);assert(cached.at(time)==linear(history,time));assert(cached.at(WindowTime(32*dt))==linear(history,WindowTime(32*dt)));}
 WindowTime begin;while(begin<history.localEnd()){
  WindowTime fast(-1),reference(-1);std::string fastError,referenceError;
  const auto end=history.localEnd();
  assert(cached.slabEndLocal(begin,end,.05,1,fast,fastError)==materialDriveSlabEndLocal(history,begin,end,.05,1,reference,referenceError));
  assert(fast==reference&&fastError==referenceError&&fast>begin);begin=fast;
 }
 WindowTime unchanged(7);assert(!cached.slabEndLocal(WindowTime{},history.localEnd(),-1,1,unchanged,error));assert(unchanged==WindowTime(7));
 // The cursor is a consumed-prefix hint, not a binary search. Unsorted end
 // values therefore retain the existing first-match behavior as well.
 IntervalHistory irregular;window.begin=0;window.end=4;assert(irregular.begin(window,error));
 for(double end:{3.,1.,2.,4.}){GasIntervalRecord r;r.localBegin=WindowTime{};r.localEnd=WindowTime(end);r.actualDt=end;assert(irregular.appendAccepted(r,error));}
 MaterialDriveCursor irregularCursor(irregular);
 for(double time:{0.,1.,2.,3.,3.5,4.,1.5})assert(irregularCursor.at(WindowTime(time))==linear(irregular,WindowTime(time)));
}
""")
    subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-fsanitize=undefined','-fno-sanitize-recover=undefined','-I'+str(SOURCE_ROOT/'applications/CHMT'),'-I'+str(SOURCE_ROOT/'common'),str(source),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
