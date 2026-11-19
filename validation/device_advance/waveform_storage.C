#include "coupling/IntervalAudit.H"
#include <cassert>
#include <iostream>
#include <sys/resource.h>

using namespace chmt;
static WallKnot knot(const CouplingInterval& interval,int index,int steps){
    WallKnot value;value.localTime=WindowTime(double(index)/steps);
    value.time=windowAbsolute(interval.begin,value.localTime);
    value.faces.resize(2);value.gasPoints.resize(7);value.solidPoints.resize(5);
    for(std::size_t face=0;face<value.faces.size();++face){auto& item=value.faces[face];
        item.temperature=300.125+17.37*index/steps+face*.17;
        for(int species=0;species<Ns;++species){
            item.speciesRate[species]=(.1+face+.123*species)*(1+double(index%13)/19);
            item.poreRate[species]=-.037*index/steps;item.poreSweepRate[species]=.019*(index%7);
        }
        for(int c=0;c<Nc;++c)item.condensedRate[c]=-.13*(1+c)*(1+double(index%11)/23);
    }
    for(std::size_t i=0;i<value.gasPoints.size();++i)value.gasPoints[i]={.13*i+.017*index/steps,.29*i-.002*(index%5),.71*i};
    for(std::size_t i=0;i<value.solidPoints.size();++i)value.solidPoints[i]={-.7*i,.25*i+.013*index/steps,.21*i};
    return value;
}
static bool equal(const WallProgramComparison& a,const WallProgramComparison& b){
    return a.thermalError==b.thermalError&&a.massPredictionError==b.massPredictionError
        &&a.geometryError==b.geometryError&&a.regimeMatches==b.regimeMatches;
}
static void exercise(int steps,bool retained){
    WallProgram reference;reference.interval.begin=1.e9;reference.interval.end=1.e9+1;
    reference.surface.area={.00123,.00791};
    reference.knots={knot(reference.interval,0,steps),knot(reference.interval,1,3),
        knot(reference.interval,3,5),knot(reference.interval,steps,steps)};
    WallProgram current;current.interval=reference.interval;current.surface=reference.surface;
    WallProgram full=current;WallProgramComparison streamed;
    Tolerances tolerance;tolerance.absoluteTemperature=1.e-8;tolerance.relativeTemperature=1.e-7;
    tolerance.absoluteMass=1.e-12;tolerance.relativeMass=1.e-7;
    tolerance.absoluteGeometry=1.e-12;tolerance.relativeGeometry=1.e-7;
    std::string error;
    for(int step=0;step<=steps;++step){
        const auto value=knot(reference.interval,step,steps);
        current.knots.push_back(value);if(retained)full.knots.push_back(value);
        assert(current.knots.size()<=3);
        if(current.knots.size()==3){
            assert(accumulateWallProgramSegment(reference,current,tolerance,streamed,error));
            current.knots.erase(current.knots.begin());
        }
    }


    current.knots.back().faces[0].temperature+=777;
    if(retained)full.knots.back().faces[0].temperature+=777;
    assert(accumulateWallProgramSegment(reference,current,tolerance,streamed,error));
    if(retained){WallProgramComparison expected;
        assert(compareWallPrograms(reference,full,tolerance,expected,error));
        assert(equal(streamed,expected));
    }
    assert(chmt::finite(streamed.thermalError)&&chmt::finite(streamed.massPredictionError)&&chmt::finite(streamed.geometryError));

    current.knots[1].gasPoints.pop_back();
    assert(!accumulateWallProgramSegment(reference,current,tolerance,streamed,error));
    assert(error=="wall interpolation field layout is incomplete");
    std::cout<<"waveform steps="<<steps<<" retained_reference="<<retained
        <<" maximum_live_knots=3 exact_full_comparison="<<retained<<'\n';
}
int main(){
    exercise(2,true);exercise(3,true);exercise(257,true);exercise(100003,false);
    rusage usage{};assert(getrusage(RUSAGE_SELF,&usage)==0);
    assert(usage.ru_maxrss<128*1024);
    std::cout<<"peak_rss_kib="<<usage.ru_maxrss<<" bounded_storage_passed\n";
}
