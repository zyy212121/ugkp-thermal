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
    std::cout << "PASS\n";
    return EXIT_SUCCESS;
}
