"""Actual time-weighted sampling kernels and physical phase inventories on CPU."""
from test_mixture_state import compile_probe, ROOT
from test_mixture_transport import fixture


def sampling_fixture():
    return fixture()+r'''
struct SamplingState:State {
 static constexpr int n=32;
 double rhoSolid=100,epsS[2]={.1,.1},rhoUsx[2]={},rhoUsy[2]={},rhoUsz[2]={},rhoEs[2]={},rhoHp[2]={},rhoDs[2]={};
 double steadySampleSum[n]={},steadySampleSquare[n]={},steadySampleBegin[n]={},steadySampleMean[n]={},steadySampleScratch[5*n]={};
};
#include "SAMPLING_HEADER"
void closeSample(double a,double b,const char*why){if(fabs(a-b)>1e-11*fmax(1.,fabs(b))){std::fprintf(stderr,"%s %.17g != %.17g\n",why,a,b);std::abort();}}
'''.replace('SAMPLING_HEADER',str(ROOT/'applications/gasUGKPSteady/private_backend/SteadySamplingMoments.cuh'))


def test_time_weighted_means_rms_and_physical_phase_inventories(tmp_path):
    compile_probe(tmp_path,sampling_fixture()+r'''
int main(){SamplingState s;initialise(s);s.V[0]=.7;s.V[1]=1.3;
for(int c=0;c<2;++c){s.rho[c]=2;s.gasSpecies.rho[c]=.5;s.gasSpecies.rho[2+c]=1.5;s.steadySampleMean[c]=4;threadIdx.x=c;beginSteadySampleWindowKernel(&s);accumulateSteadySampleWindowKernel(&s,1.);}
for(int c=0;c<2;++c){s.rho[c]=6;s.gasSpecies.rho[c]=1.5;s.gasSpecies.rho[2+c]=4.5;s.epsS[c]=.4;threadIdx.x=c;accumulateSteadySampleWindowKernel(&s,3.);finishSteadySampleWindowKernel(&s,4.);
closeSample(s.steadySampleMean[c],5,"time weighted rho mean");closeSample(s.steadySampleScratch[c],.2,"previous-window relative mean change");closeSample(s.steadySampleScratch[32+c],sqrt(3.)/5,"time weighted RMS not standard error");
closeSample(s.steadySampleScratch[64+c],2*.9*s.V[c],"initial gas physical inventory");closeSample(s.steadySampleScratch[96+c],6*.6*s.V[c],"final gas physical inventory");
closeSample(s.steadySampleScratch[64+14+c],10*s.V[c],"particle inventory no extra porosity");closeSample(s.steadySampleScratch[96+14+c],40*s.V[c],"final particle inventory");
closeSample(s.steadySampleMean[28+c],1.25,"species mean");closeSample(s.steadySampleMean[30+c],3.75,"species mean closure");
closeSample(s.steadySampleScratch[64+28+c],.5*.9*s.V[c],"species initial physical inventory");closeSample(s.steadySampleScratch[96+28+c],1.5*.6*s.V[c],"species final physical inventory");
closeSample(s.steadySampleScratch[128+c],5*s.V[c]*.6,"characteristic inventory");}
}
''')


def test_energy_rms_retains_small_variation_on_large_formation_baseline(tmp_path):
    compile_probe(tmp_path,sampling_fixture()+r'''
int main(){SamplingState s;initialise(s);const int c=0,j=8;s.rhoE[c]=-1e12+2;s.steadySampleMean[j]=-1e12+1;threadIdx.x=c;
beginSteadySampleWindowKernel(&s);accumulateSteadySampleWindowKernel(&s,1.);s.rhoE[c]=-1e12+6;accumulateSteadySampleWindowKernel(&s,3.);finishSteadySampleWindowKernel(&s,4.);
const double scale=fabs(s.p[c]);closeSample(s.steadySampleMean[j],-1e12+5,"formation-inclusive mean");closeSample(s.steadySampleScratch[j],4/scale,"energy change gauge independent");closeSample(s.steadySampleScratch[32+j],sqrt(3.)/scale,"formation offset erased RMS");
}
''')
