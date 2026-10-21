// Preserve every original quadrature point and left-to-right FP64 sum.
#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
#include "common/gasWall/reconstruction/ConservativeProfile.H"
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <cassert>
#include <cstring>
#include <iostream>
struct Barrier {std::mutex mutex;std::condition_variable cv;int lanes,arrived=0,generation=0;explicit Barrier(int n):lanes(n){}void wait(){std::unique_lock<std::mutex>lock(mutex);int g=generation;if(++arrived==lanes){arrived=0;++generation;cv.notify_all();}else cv.wait(lock,[&]{return generation!=g;});}};
struct Team {Barrier&barrier;int lane,lanes;double*shared=nullptr;double broadcast(double v)const{if(leader())*shared=v;sync();v=*shared;sync();return v;}int rank()const{return lane;}int size()const{return lanes;}bool leader()const{return lane==0;}void sync()const{barrier.wait();}};
template<int M>void check(int lanes,bool endpoints,bool algebraic){
 using namespace ugkwp::gaswall;using namespace ugkwp::gaswall::reconstruction;
 CapturedSstWall fixture;auto cfg=fixture.config;cfg.nodes=M;cfg.coherentDiscretization=true;WallWorkspace<double,2,M>w{};
 for(int i=0;i<M;++i){double t=double(i)/(M-1);w.y[i]=fixture.input.matchingDistance*t*t;w.state[i][0]=std::log1p(20*t);w.state[i][1]=-.2*t;w.state[i][2]=35*t;w.state[i][3]=.8-.1*t;w.state[i][4]=.01+std::sin(2*t);w.state[i][5]=.01+t;}
 static ProfileScratch scratch;Barrier barrier(lanes);std::vector<std::thread>threads;std::vector<Q>results(lanes);
 Profile serial(fixture.input,cfg,w);Segment seg{};seg.lo=0;seg.hi=.63*w.y[M-1];seg.aRho=.23;seg.aT=-.13;seg.aK=.04;seg.aOmega=-.17;seg.aY=.12;seg.aU[0]=-1.3;seg.aU[1]=2.1;seg.algebraicOmega=algebraic;serial.segments.push_back(seg);
 if(endpoints){auto left=serial.base(seg.lo),right=serial.base(seg.hi);left.rho*=.97;right.T*=1.01;right.U[0]-=.5;serial.setEndpoints(0,left,right);}
 const auto expected=serial.average(serial.segments[0]);
 for(int lane=0;lane<lanes;++lane)threads.emplace_back([&,lane]{Profile p=serial;results[lane]=p.averageTeam(p.segments[0],scratch,Team{barrier,lane,lanes});});
 for(auto&t:threads)t.join();for(int j=0;j<9;++j){assert(std::isfinite(results[0][j]));assert(std::memcmp(&results[0][j],&expected[j],sizeof(double))==0);}
}
template<int M>void checkTraces(int lanes,int variant){
 using namespace ugkwp::gaswall;using namespace ugkwp::gaswall::reconstruction;
 CapturedSstWall fixture;auto cfg=fixture.config;cfg.nodes=M;cfg.coherentDiscretization=true;WallWorkspace<double,2,M>w{};
 for(int i=0;i<M;++i){double t=double(i)/(M-1);w.y[i]=fixture.input.matchingDistance*t*t;w.state[i][0]=std::log1p((20+variant)*t);w.state[i][1]=-.2*t;w.state[i][2]=(35+variant)*t;w.state[i][3]=.8-.1*t;w.state[i][4]=.01+std::sin(2*t);w.state[i][5]=.01+t;}
 Profile original(fixture.input,cfg,w);Q means[2];
 for(int i=0;i<2;++i){Segment seg{};seg.lo=i?.43*w.y[M-1]:0;seg.hi=i?w.y[M-1]:.43*w.y[M-1];seg.algebraicOmega=i==0;means[i]=original.baseAverage(seg);seg.target=means[i];seg.target[4]*=1.01;seg.target[1]+=.1*seg.target[0];original.segments.push_back(seg);}
 Profile serial=original;serial.reconstructInternalTraces();static ProfileScratch scratch;Barrier barrier(lanes);double shared=0;std::vector<std::thread>threads;
 for(int lane=0;lane<lanes;++lane)threads.emplace_back([&,lane]{Profile p=original;p.reconstructInternalTracesTeam(scratch,Team{barrier,lane,lanes,&shared});for(int i=0;i<2;++i){for(int j=0;j<9;++j)assert(std::memcmp(&scratch.baseMeans[i][j],&means[i][j],sizeof(double))==0);for(double t:{0.,.2,.5,.8,1.}){double y=p.segments[i].lo+t*(p.segments[i].hi-p.segments[i].lo);auto got=p.pointIn(y,p.segments[i]),want=serial.pointIn(y,serial.segments[i]);assert(got.rho==want.rho&&got.T==want.T&&got.k==want.k&&got.omega==want.omega);for(int j=0;j<3;++j)assert(got.U[j]==want.U[j]);for(int j=0;j<2;++j)assert(got.Y[j]==want.Y[j]);}p.segments[i].aRho=.2;p.segments[i].aT=-.1;p.segments[i].aK=.3;p.segments[i].aU[0]=-.5;auto after=p.baseAverage(p.segments[i]);for(int j=0;j<9;++j)assert(std::memcmp(&after[j],&scratch.baseMeans[i][j],sizeof(double))==0);}});
 for(auto&t:threads)t.join();
}
int main(){for(int lanes:{1,4,7,32})for(bool endpoints:{false,true})for(bool algebraic:{false,true}){check<24>(lanes,endpoints,algebraic);check<48>(lanes,endpoints,algebraic);check<128>(lanes,endpoints,algebraic);}for(int lanes:{1,4,7,32})for(int variant:{0,1}){checkTraces<24>(lanes,variant);checkTraces<48>(lanes,variant);checkTraces<128>(lanes,variant);}std::cout<<"48 quadrature and24 trace/cache cases passed bitwise\n";}
