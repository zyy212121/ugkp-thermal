#include "gasTransport/MixtureThermo.H"
#include <cstdio>
struct Quadratic {
    ugkwp::ThermoStatus operator()(double t,double& e,double& c)const {
        e=t*t;c=2*t;return ugkwp::ThermoStatus::Success;
    }
};
int main(){
    ugkp::diagnostics::DeviceContext diagnostics{};
    ugkwp::ThermoInversionControls<double> controls;
    controls.maximumIterations=2;
    double t=1;
    for(int i=0;i<3;++i){
        auto status=ugkwp::invertCaloricEnergy(Quadratic{},2.,.1,10.,controls,t,nullptr,&diagnostics);
        std::printf("call=%d success=%d temperature=%.17g residual=%.17g warnings=%u\n",i+1,status==ugkwp::ThermoStatus::Success,t,t*t-2.,diagnostics.warnings[1].count);
    }
}