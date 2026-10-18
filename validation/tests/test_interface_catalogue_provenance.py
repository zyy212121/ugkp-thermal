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

def test_all_moved_interface_diagnostics_identify_the_actual_producer(tmp_path):
 import re
 source=tmp_path/'all_catalogue.cpp';source.write_text(r'''#include "common/DiagnosticsCatalogue.H"
#include <iostream>
#include <cstring>
int main(){for(const auto& s:ugkp::diagnostics::sourceCatalogue())if(s.file.find("ablation/Interface")!=std::string::npos)std::cout<<s.siteId<<"|"<<s.file<<"|"<<s.function<<"|"<<s.line<<"\n";}
''')
 exe=tmp_path/'all_catalogue'
 q=subprocess.run(['g++','-std=c++14','-O2','-I'+str(SOURCE_ROOT),str(source),'-o',str(exe)],capture_output=True,text=True);assert q.returncode==0,q.stderr
 q=subprocess.run([str(exe)],capture_output=True,text=True);assert q.returncode==0,q.stderr
 count=0
 for entry in q.stdout.splitlines():
  site,file,function,line=entry.split('|');lines=(SOURCE_ROOT/file).read_text().splitlines();number=int(line)
  assert re.search(r'\b'+site+r'\b',lines[number-1]),entry+' does not identify the actual producer line'
  window='\n'.join(lines[max(0,number-4):number+3])
  assert re.search(r'(?:interfaceFailure|clipPhysicsMassFraction|reportPhysicsWarning|makeRecord)\s*\([^;]*\b'+site+r'\b',window),entry+' is not a complete diagnostic producer'
  functions=re.findall(r'CHMT_HD\s+inline\s+[^\n]*?\b(\w+)\s*\(', '\n'.join(lines[:number]))
  assert functions and functions[-1]==function,(entry,functions[-1:])
  count+=1
 assert count>=30
