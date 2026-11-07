#include "gas_drive_probe.H"
#include "materials/MaterialDrive.H"
#include <cassert>
#include <iostream>
#include <random>
#include <limits>
using namespace chmt;
using gasDriveProbe::Pair;
GasIntervalRecord record(const std::vector<CoupledSurfaceDownload>& data,const std::vector<CoupledWallDownload>& wall,int index){
    GasIntervalRecord r;r.actualDt=1;r.begin=index;r.end=index+1;r.localBegin=WindowTime(index);r.localEnd=WindowTime(index+1);r.microSequence=index+1;
    for(const auto& f:data){r.gasTrace.push_back(f.bulk);r.gasGradient.push_back(f.gradient);r.gasTraction.push_back(f.traction);r.radiationReceiver.push_back(f.radiationReceiver);}
    for(const auto& w:wall)r.gasWallMatching.push_back(w.matching);return r;
}
int main(){
    std::vector<Pair> cases;std::mt19937_64 random(731982);
    auto value=[&]{return double(random()%10001)/10000.;};
    for(int count:{0,1,3,17,129})for(bool matching:{false,true})for(double scale:{1e-30,1.,1e30})for(int change=0;change<20;++change){
        Pair p;p.tractionScale=std::max(scale,1e-30);p.held.resize(count);if(matching)p.heldWall.resize(count);
        for(int i=0;i<count;++i){auto& f=p.held[i];auto& b=f.bulk;b.temperature=300+value();b.pressure=1e5*(1+value());b.rho=1+value();b.soundSpeed=300+value();
            b.velocity={scale*value(),scale*value(),scale*value()};b.Y[0]=value();b.Y[1]=1-b.Y[0];f.traction={scale*value(),scale*value(),scale*value()};
            for(auto& g:f.gradient.velocity)g={scale*value(),scale*value(),scale*value()};f.radiationReceiver=ConsumeSolid;
            if(matching){auto& m=p.heldWall[i].matching;m.pressure=b.pressure;m.mechanicalPressure=b.pressure+scale;m.state.temperature=b.temperature;
                for(int k=0;k<3;++k)m.state.velocity[k]=scale*value();m.state.massFraction[0]=b.Y[0];m.state.massFraction[1]=b.Y[1];m.state.k=scale*value();m.state.omega=scale*value();}}
        p.sample=p.held;p.sampleWall=p.heldWall;
        if(count){const auto f=random()%count;auto& s=p.sample[f];
            switch(change){
            case 0:break;case 1:s.bulk.temperature*=1.125;break;case 2:s.bulk.pressure*=1.25;break;
            case 3:s.bulk.rho*=2;break;case 4:s.bulk.velocity.x+=scale;break;case 5:s.bulk.Y[1]+=.125;break;
            case 6:if(matching)p.sampleWall[f].matching.pressure*=2;break;
            case 7:if(matching)p.sampleWall[f].matching.mechanicalPressure*=2;break;
            case 8:if(matching)p.sampleWall[f].matching.state.temperature*=2;break;
            case 9:if(matching)p.sampleWall[f].matching.state.velocity[2]+=scale;break;
            case 10:if(matching)p.sampleWall[f].matching.state.massFraction[0]+=.25;break;
            case 11:if(matching)p.sampleWall[f].matching.state.k*=2;break;
            case 12:if(matching)p.sampleWall[f].matching.state.omega*=2;break;
            case 13:s.traction.z+=scale;break;case 14:s.gradient.velocity[2].z+=scale;break;
            case 15:s.radiationReceiver=ConsumeGas;break;
            case 16:p.sample.pop_back();break;case 17:if(matching)p.sampleWall.pop_back();break;
            case 18:s.bulk.temperature=std::numeric_limits<double>::quiet_NaN();break;
            case 19:p.sample.back().gradient.velocity[2].z=std::numeric_limits<double>::infinity();break;
            }}cases.push_back(std::move(p));
    }
    const auto actual=gasDriveProbe::evaluate(cases);std::size_t comparisons=0,rejected=0,changed=0;
    for(std::size_t i=0;i<cases.size();++i){const auto& p=cases[i];const auto& gpu=actual[i];
        const ResidentMaterialDriveRecord held{p.held.data(),p.heldWall.data(),p.held.size(),p.heldWall.size()};
        const ResidentMaterialDriveRecord sample{p.sample.data(),p.sampleWall.data(),p.sample.size(),p.sampleWall.size()};
        const auto host=ugkwp::compareGasDrive<Ns>(held,sample,p.tractionScale);
        assert(host.layoutMatches==gpu.layoutMatches&&host.receiverChanged==gpu.receiverChanged);
        assert((std::isnan(host.change)&&std::isnan(gpu.change))||host.change==gpu.change);
        IntervalHistory history;CouplingInterval interval;interval.end=2;std::string error;assert(history.begin(interval,error));
        assert(history.appendAccepted(record(p.held,p.heldWall,0),error));assert(history.appendAccepted(record(p.sample,p.sampleWall,1),error));
        std::vector<double> thresholds{0,.05,1,100};
        if(std::isfinite(host.change)&&host.change>=0){thresholds.push_back(host.change);thresholds.push_back(std::nextafter(host.change,std::numeric_limits<double>::infinity()));
            if(host.change>0)thresholds.push_back(std::nextafter(host.change,0.));}
        for(double threshold:thresholds){if(!std::isfinite(threshold))continue;
            WindowTime end(-1);MaterialDriveCursor native(history);const bool ok=native.slabEndLocal(WindowTime{},WindowTime(2),threshold,p.tractionScale,end,error);
            const bool expected=host.layoutMatches&&std::isfinite(host.change);assert(ok==expected);
            if(ok){const bool bound=gpu.change>threshold||gpu.receiverChanged;assert(end==WindowTime(bound?1:2));changed+=bound;}
            else {++rejected;assert(error==(host.layoutMatches?"nonfinite gas driving change":"gas drive layout changed"));assert(end==WindowTime(-1));}
            ++comparisons;
        }
    }

    std::vector<gasDriveProbe::SelectionTest> tests;
    for(std::size_t i=0;i<cases.size();++i)for(int mode=0;mode<20;++mode){
        if(mode&&i%20!=0&&i%20!=15)continue;
        gasDriveProbe::SelectionTest test;test.pair=i;auto& c=test.controls;
        c.begin=ugkp::CompensatedTime(0);c.requested=c.total=ugkp::CompensatedTime(2);c.tolerance=.05;c.tractionScale=cases[i].tractionScale;
        switch(mode){
        case 1:c.begin=ugkp::CompensatedTime(.25);c.requested=ugkp::CompensatedTime(.75);break;
        case 2:c.begin=ugkp::CompensatedTime(1);break;
        case 3:c.begin=ugkp::CompensatedTime(1.25);c.requested=ugkp::CompensatedTime(1.75);break;
        case 4:c.requested=ugkp::CompensatedTime(1);break;
        case 5:c.begin=ugkp::CompensatedTime(-1);break;
        case 6:c.tolerance=-1;break;case 7:c.tractionScale=0;break;
        case 8:c.requested=ugkp::CompensatedTime(3);break;
        case 9:test.frames=0;break;case 10:c.total=ugkp::CompensatedTime(3);break;
        case 11:test.intervals[0].end=ugkp::CompensatedTime(1.5);test.intervals[0].actualDt=1.5;
            test.intervals[1].begin=ugkp::CompensatedTime(.5);test.intervals[1].actualDt=1.5;break;
        case 12:test.intervals[0].end=ugkp::CompensatedTime(2);test.intervals[0].actualDt=2;
            test.intervals[1].begin=ugkp::CompensatedTime(0);test.intervals[1].actualDt=2;break;
        case 13:c.begin=ugkp::CompensatedTime(2);break;
        case 14:c.begin=ugkp::CompensatedTime(std::numeric_limits<double>::quiet_NaN());break;
        case 15:c.tractionScale=std::numeric_limits<double>::quiet_NaN();break;
        case 16:c.tolerance=std::numeric_limits<double>::infinity();break;
        case 17:c.requested=ugkp::CompensatedTime(0);break;
        case 18:c.requested=ugkp::CompensatedTime(1);c.requested.correction=1e-20;break;
        case 19:c.requested=ugkp::CompensatedTime(1);c.requested.correction=-1e-20;break;
        }tests.push_back(test);
    }
    const auto selections=gasDriveProbe::select(cases,tests);std::size_t selectionErrors=0;
    const auto nativeTime=[](ugkp::CompensatedTime t){WindowTime result;result.sum=t.sum;result.correction=t.correction;return result;};
    for(std::size_t i=0;i<tests.size();++i){const auto& test=tests[i];const auto& p=cases[test.pair];
        const auto& c=test.controls;const auto& actual=selections[i];IntervalHistory history;CouplingInterval interval;interval.end=c.total.value();
        std::string error;assert(history.begin(interval,error));
        for(std::size_t f=0;f<test.frames;++f){auto r=record(f?p.sample:p.held,f?p.sampleWall:p.heldWall,f);
            r.localBegin=nativeTime(test.intervals[f].begin);r.localEnd=nativeTime(test.intervals[f].end);r.actualDt=test.intervals[f].actualDt;
            r.begin=windowAbsolute(0,r.localBegin);r.end=windowAbsolute(0,r.localEnd);assert(history.appendAccepted(r,error));}
        WindowTime expected(-17);MaterialDriveCursor native(history);
        const bool ok=native.slabEndLocal(nativeTime(c.begin),nativeTime(c.requested),c.tolerance,c.tractionScale,expected,error);
        using Error=ugkwp::GasDriveSelectionError;assert(ok==(actual.error==Error::None));
        assert(nativeTime(actual.end)==expected);
        if(!ok){++selectionErrors;
            if(error=="invalid material driving control")assert(actual.error==Error::InvalidControl);
            else if(error=="no driving sample at material slab start")assert(actual.error==Error::NoSample);
            else if(error=="gas drive layout changed")assert(actual.error==Error::LayoutChanged);
            else if(error=="nonfinite gas driving change")assert(actual.error==Error::NonfiniteChange);
            else {assert(error=="driving slab does not advance");assert(actual.error==Error::NonAdvancing);}
        }
    }
    assert(gasDriveProbe::select({},{}).empty());
    std::cout<<"PASS "<<tests.size()<<" resident material interval selections, "<<selectionErrors<<" rejected, exact native endpoints/errors including unordered and sub-ULP controls\n";
    assert(gasDriveProbe::evaluate({}).empty());
    std::cout<<"PASS "<<cases.size()<<" host/GPU driving pairs, "<<comparisons<<" native slab decisions, "<<rejected<<" rejected and "<<changed<<" bounded; exact ordered differences, three graph replays, empty input\n";
}
