"""A runtime catalogue entry must point at its producer, not a nearby site number."""
import subprocess
from validation.tools.paths import SOURCE_ROOT

def test_zero_diffusion_warning_catalogue_points_to_its_producer(tmp_path):
 source=tmp_path/'catalogue.cpp';source.write_text(r'''#include "common/DiagnosticsCatalogue.H"
#include <iostream>
int main(){for(const auto& s:ugkp::diagnostics::sourceCatalogue())if(s.siteId==3807){std::cout<<s.file<<"\n"<<s.function<<"\n"<<s.line<<"\n";return 0;}return 2;}
''')
 exe=tmp_path/'catalogue';q=subprocess.run(['g++','-std=c++14','-O2','-I'+str(SOURCE_ROOT),str(source),'-o',str(exe)],capture_output=True,text=True)
 assert q.returncode==0,q.stdout+q.stderr
 q=subprocess.run([str(exe)],capture_output=True,text=True);assert q.returncode==0,q.stdout+q.stderr
 file,function,line=q.stdout.splitlines()
 assert file=='applications/CHMT/ablation/InterfaceComposition.H'
 assert function=='linearComposition'
 producer=(SOURCE_ROOT/file).read_text().splitlines()[int(line)-1]
 assert 'reportPhysicsWarning(p,ugkp::diagnostics::Code::AccuracyResidual,3807,finalResidual,tolerance)' in producer
