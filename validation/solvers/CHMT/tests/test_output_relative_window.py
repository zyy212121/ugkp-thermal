from validation.tools.paths import SOURCE_ROOT
import subprocess

def test_output_relative_windows(tmp_path):
    source=tmp_path/'clock.cpp'
    source.write_text(r"""
#include "coupling/CouplingWindow.H"
#include <cassert>
#include <cmath>
#include <vector>
using namespace chmt;
using namespace ugkp::gpu;
void sequence(double contact,double radiation,bool enabled,
              const std::vector<double>& ends,const std::vector<bool>& due){
    CouplingControls controls;
    controls.scheduleOrigin=10;
    controls.clock=makeOutputCouplingClock(1,contact,radiation,enabled);
    CouplingWindowController owner;
    double now=10,previous=10;
    std::string error;
    for(unsigned i=0;i<ends.size();++i){
        assert(owner.begin(now,11,i+1,controls,error));
        const auto window=owner.interval();
        assert(window.begin==now && window.identity.sequence==i+1);
        assert(sameEventTime(window.end,10+ends[i]));
        const bool output=CouplingWindowController::outputDue(window,controls);
        assert(output==(i+1==ends.size()));
        const bool radiate=outputRelativeRadiationDue(controls.clock,10,previous,window.end,output);
        assert(radiate==due[i]);
        if(radiate)previous=window.end;
        now=window.end;
    }
    assert(CouplingWindowController::outputIndex(now,controls)==1);
}
int main(){
    sequence(.2,.5,true,{.2,.4,.5,.6,.8,1},{false,false,true,false,false,true});
    sequence(.5,.2,true,{.2,.4,.6,.8,1},{true,true,true,true,true});
    sequence(.5,.5,true,{.5,1},{true,true});
    sequence(2,3,true,{1},{true});
    sequence(.2,.5,false,{.2,.4,.6,.8,1},{false,false,false,false,false});
    CouplingControls controls;
    controls.clock=makeOutputCouplingClock(1,.2,.5,true);
    controls.scheduleOrigin=10;
    CouplingWindowController owner;std::string error;
    assert(owner.begin(10.4,10.45,3,controls,error));
    assert(owner.interval().end==10.45);
    assert(!CouplingWindowController::outputDue(owner.interval(),controls));
    assert(outputRelativeRadiationDue(controls.clock,10,10,10.45,true));
    assert(!outputRelativeRadiationDue(controls.clock,10,10,10.45,false));
    assert(owner.begin(10.45,11,4,controls,error));
    assert(owner.interval().end==10.5);
    assert(outputRelativeRadiationDue(controls.clock,10,10.45,10.5,false));
    assert(owner.begin(std::nextafter(10.5,0.),11,5,controls,error));
    assert(sameEventTime(owner.interval().end,10.6));
    assert(owner.begin(11,12,7,controls,error));
    assert(sameEventTime(owner.interval().end,11.2));
    auto gas=stepToEvent(10.49,.02,10.5);
    assert(gas.end==10.5 && gas.dt==10.5-10.49);
}
""")
    binary=tmp_path/'clock'
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(SOURCE_ROOT/'applications/CHMT'),
                    str(source),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
