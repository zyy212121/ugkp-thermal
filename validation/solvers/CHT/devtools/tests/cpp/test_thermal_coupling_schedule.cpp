#include "../../../../../../common/gpu/PhysicalEventClock.H"

#include <cstdlib>
#include <iostream>

int main()
{
    using ugkp::gpu::reachedEventTime;
    if (!reachedEventTime(std::nextafter(1.1,1.),1.1,0.1) ||
        !reachedEventTime(1.1,1.1,0.1) ||
        reachedEventTime(1.0999999999,1.1,0.1) ||
        reachedEventTime(1.09,1.1,0.1))
        return EXIT_FAILURE;
    using ugkp::gpu::nextPeriodicTime;
    using ugkp::gpu::stepToEvent;
    if (nextPeriodicTime(0., 0.005, 0.29500248649130967) != 0.3
        || nextPeriodicTime(0., 0.005, std::nextafter(0.295,0.)) != 0.3)
        return EXIT_FAILURE;
    const double origin=std::ldexp(1.,40);
    const double tick=std::nextafter(origin,INFINITY)-origin;
    if (nextPeriodicTime(origin,tick,origin)!=origin+tick ||
        ugkp::gpu::eventDeadline(origin,origin+tick,origin+2*tick)!=origin+tick ||
        stepToEvent(origin,tick,origin+2*tick).end!=origin+tick)
        return EXIT_FAILURE;
    const double nominal = 7e-5;
    double t = 0.29500248649130967;
    int steps = 0;
    while (t < 0.3 && ++steps < 100) {
        const auto step = stepToEvent(t,nominal,0.3);
        if (!(step.dt > 0 && step.dt <= nominal && step.end > t && step.end <= 0.3))
            return EXIT_FAILURE;
        t = step.end;
    }
    if (t != 0.3 || stepToEvent(t,nominal,0.305).dt != nominal)
        return EXIT_FAILURE;
    using namespace ugkp::gpu;
    const auto march = [](double contact,double radiation,bool enabled,
        const std::initializer_list<double>& events,const std::initializer_list<bool>& kinds) {
        const auto clock=makeOutputCouplingClock(.01,contact,radiation,enabled);
        constexpr double start=1.5;
        double heatTime=start,radiationTime=start;
        auto kind=kinds.begin();
        for(double offset:events) {
            const double target=nextFullOrContactEvent(clock,start,heatTime);
            if(!sameEventTime(target,start+offset))return false;
            const double write=nextPeriodicTime(start,.01,heatTime);
            const auto step=stepToEvent(heatTime,.1,eventDeadline(heatTime,target,write));
            if(!(step.dt>0) || !sameEventTime(step.end,target))return false;
            const bool full=outputRelativeRadiationDue(clock,start,radiationTime,step.end,
                sameEventTime(step.end,write));
            if(full!=*kind++)return false;
            heatTime=step.end;if(full)radiationTime=step.end;
        }
        return true;
    };
    if(!march(.5,.2,true,{.002,.004,.006,.008,.01,.012},{true,true,true,true,true,true})
        || !march(.2,.5,true,{.002,.004,.005,.006,.008,.01,.012},{false,false,true,false,false,true,false})
        || !march(.3,.5,true,{.003,.005,.006,.009,.01,.013},{false,true,false,false,true,false})
        || !march(.5,.2,false,{.005,.01,.015},{false,false,false})
        || !march(100,100,true,{.01,.02},{true,true}))
        return EXIT_FAILURE;
    const auto aligned=makeOutputCouplingClock(.01,.3,.2,true);
    for(double saved:{1.51,std::nextafter(1.51,0.),std::nextafter(1.51,2.)}) {
        const double next=nextFullOrContactEvent(aligned,1.5,saved);
        if(!sameEventTime(next,1.512)
            || outputRelativeRadiationDue(aligned,1.5,saved,std::nextafter(saved,2.),false))
            return EXIT_FAILURE;
    }
    const auto shortWindow=makeOutputCouplingClock(.01,.5,.5,true);
    if(!outputRelativeRadiationDue(shortWindow,1.5,1.50,1.503,true))
        return EXIT_FAILURE;
    std::cout << "PASS\n";
    return EXIT_SUCCESS;
}
