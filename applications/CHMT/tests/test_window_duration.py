"""Paid local durations remain observable below the absolute clock spacing."""
from pathlib import Path
import subprocess
APP = Path(__file__).resolve().parents[1]
def test_compensated_window_duration(tmp_path):
    source = tmp_path / 'duration.cpp'
    source.write_text(r'''
#include "coupling/WindowTime.H"
#include <cassert>
#include <cmath>
int main(){
 using namespace chmt;
 WindowTime elapsed;
 const double origin=0x1p40, dt=0x1p-20;
 assert(origin+dt==origin);
 elapsed+=dt;
 assert(elapsed>WindowTime{});
 assert(windowDuration(elapsed,WindowTime{})==dt);
 WindowTime large(1.0L);large+=0x1p-100;
 assert(large>WindowTime(1.0L));
 assert(windowDuration(large,WindowTime(1.0L))==0x1p-100);
 large-=0x1p-100;assert(large==WindowTime(1.0L));
 WindowTime total; for(int i=0;i<1024;++i)total+=dt;
 assert(windowDuration(total,WindowTime{})==1024*dt);
 assert(windowAbsolute(origin,total)==origin+1024*dt);
}
''')
    binary=tmp_path/'duration'
    p=subprocess.run(['g++','-std=c++17','-O2','-I'+str(APP),str(source),'-o',str(binary)],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    subprocess.run([str(binary)],check=True)

def test_same_absolute_time_history_wall_and_donor(tmp_path):
    source=tmp_path/'history.cpp'
    source.write_text(r'''
#include "gpu/GasWindowProgram.H"
#include "coupling/IntervalAudit.H"
#include "materials/MaterialDrive.H"
#include <cassert>
int main(){
 using namespace chmt;std::string error;
 const double origin=0x1p40,dt=0x1p-20;
 CouplingInterval interval;interval.begin=origin;interval.end=origin+256*dt;
 IntervalHistory history;assert(history.begin(interval,error));
 MaterialDonorReserve reserve;assert(reserve.reset({}, {}, origin,error));
 MaterialDonorPlan plan;plan.interval=interval;
 MaterialReserveKnot a,z;a.time=origin;a.localTime=WindowTime{};
 z.time=interval.end;z.localTime=intervalDuration(interval);plan.knots={a,z};
 assert(reserve.setPlan(plan,error));
 for(int i=0;i<256;++i){
  GasIntervalRecord record;record.microSequence=i+1;record.actualDt=dt;
  record.localBegin=history.localEnd();record.localEnd=record.localBegin+dt;
  record.begin=windowAbsolute(origin,record.localBegin);record.end=windowAbsolute(origin,record.localEnd);
  record.radiationEnergy={dt};record.radiationReceiver={ConsumeSolid};
  if(i==0)assert(record.begin==record.end);
  assert(reserve.apply(record,error));assert(history.appendAccepted(record,error));
 }
 assert(history.complete());auto cursor=history.cursor();std::vector<ExchangePacket> packets;std::vector<Real> radiation;
 assert(cursor.takeThroughLocal(WindowTime(dt/2),packets,radiation,error));assert(radiation[0]==dt/2);
 assert(cursor.takeThroughLocal(history.localEnd(),packets,radiation,error));assert(radiation[0]==256*dt-dt/2);assert(cursor.complete());
 WallProgram wall;wall.interval=interval;WallKnot first,last;
 first.time=origin;first.localTime=WindowTime{};last.time=interval.end;last.localTime=intervalDuration(interval);
 first.faces.resize(1);last.faces.resize(1);first.faces[0].temperature=100;last.faces[0].temperature=356;
 wall.knots={first,last};WallKnot sample;
 assert(sampleGasWallProgramLocal(wall,WindowTime(dt),sample,error));assert(sample.time==origin);assert(sample.faces[0].temperature==101);
 assert(intervalSampleWallLocal(wall,WindowTime(dt),sample,error));assert(sample.faces[0].temperature==101);
 WindowTime end;std::uint64_t count=0;assert(materialDriveSlabEndLocal(history,WindowTime{},history.localEnd(),.05,1,end,count,error));assert(end==history.localEnd());
}
''')
    binary=tmp_path/'history'
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(APP),'-I'+str(APP.parents[1]/'common'),str(source),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
