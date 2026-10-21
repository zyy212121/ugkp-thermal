// CPU collective-control-flow check for borrowed slope initialization. This is
// not a CUDA race/synchronization qualification.
#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
#include "common/gasWall/reconstruction/ConservativeProfile.H"
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <cassert>
#include <iostream>
#include <limits>
struct Barrier {
 std::mutex mutex;std::condition_variable cv;int lanes,arrived=0,generation=0;
 explicit Barrier(int n):lanes(n){}
 void wait(){std::unique_lock<std::mutex> lock(mutex);int g=generation;if(++arrived==lanes){arrived=0;++generation;cv.notify_all();}else cv.wait(lock,[&]{return generation!=g;});}
};
struct Team {Barrier&barrier;int lane,lanes;int rank()const{return lane;}int size()const{return lanes;}void sync()const{barrier.wait();}};
template<int M> void check(int lanes){
 using namespace ugkwp::gaswall;using namespace ugkwp::gaswall::reconstruction;
 CapturedSstWall m;auto cfg=m.config;cfg.nodes=M;cfg.coherentDiscretization=true;WallWorkspace<double,2,M>w{};
 for(int i=0;i<M;++i){double t=double(i)/(M-1);w.y[i]=m.input.matchingDistance*t*t;w.state[i][0]=std::log1p(20*t);w.state[i][1]=.1*t;w.state[i][2]=-5*t;w.state[i][3]=.8;w.state[i][4]=std::sin(3*t);w.state[i][5]=t*(1+.1*t);}
 double slopes[M][3];for(auto&r:slopes)for(auto&v:r)v=std::numeric_limits<double>::quiet_NaN();Barrier b(lanes);std::vector<std::thread> threads;
 for(int lane=0;lane<lanes;++lane)threads.emplace_back([&,lane]{Profile cached(m.input,cfg,w,slopes,Team{b,lane,lanes});Profile direct(m.input,cfg,w);for(int i=0;i<M;++i)for(int j:{2,4,5})assert(cached.nodeSlope(i,j)==direct.nodeSlope(i,j));});
 for(auto&t:threads)t.join();for(auto&r:slopes)for(auto v:r)assert(std::isfinite(v));
}
int main(){for(int lanes:{1,4,7,32}){check<24>(lanes);check<48>(lanes);check<128>(lanes);}std::cout<<"12 capacity/team slope-cache cases passed\n";}
