"""Precision publication must diagnose actual nonfinite narrowing, not return success."""
import subprocess
from validation.tools.paths import SOURCE_ROOT as ROOT


def test_actual_float_publication_reports_overflow(tmp_path):
    src=tmp_path/'precision.cpp'
    src.write_text(r'''
#include "gasWall/WallPrecision.H"
#include <cassert>
#include <limits>
int main(){
 using namespace ugkwp::gaswall;
 ugkp::diagnostics::DeviceContext context{};
 WallInput<float,2> input;input.diagnostics=&context;input.diagnosticEntity=17;
 WallOutput<double,2> source;WallOutput<float,2> output;
 source.traceTemperature=300;source.traceDensity=1;source.traceMassFraction[0]=.7;source.traceMassFraction[1]=.3;
 assert(detail::publishWallPrecision(input,source,output));
 assert(output.traceTemperature==300&&!context.fatalClaim);
 assert(!output.chemistryMismatchAvailable&&std::isnan(output.chemistryMismatch[0]));
 source.conductiveHeatFlux=std::numeric_limits<double>::max();
 assert(!detail::publishWallPrecision(input,source,output));
 assert(context.fatalClaim&&context.firstFatal.code==ugkp::diagnostics::Code::NonFinite);
 assert(context.firstFatal.entityIndex==17);
}
''')
    exe=src.with_suffix('')
    subprocess.run(['g++','-std=c++17','-I'+str(ROOT/'common'),str(src),'-o',str(exe)],check=True,capture_output=True,text=True)
    result=subprocess.run([str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
