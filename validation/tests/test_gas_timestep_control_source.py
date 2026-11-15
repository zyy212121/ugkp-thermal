"""Runtime controlDict owns the adaptive switch; scheme owns numeric limits."""
from pathlib import Path
import os,subprocess
import pytest
from validation.tools.paths import SOURCE_ROOT
ROOT=SOURCE_ROOT
@pytest.fixture(scope='module')
def parser(tmp_path_factory):
    out=tmp_path_factory.mktemp('time-control')
    source=out/'parse.C'
    source.write_text(r"""
#include "argList.H"
#include "Time.H"
#include "FoamGasTimeStepControls.H"
int main(int argc,char**argv){
 Foam::argList args(argc,argv);
 Foam::Time time(Foam::Time::controlDictName,args);
 const auto c=ugkp::solverControls::readGasTimeStepControls(time);
 Foam::Info << "RESULT " << int(c.adjustTimeStep) << ' ' << c.maxCo << ' ' << c.maxDeltaT << Foam::endl;
}
""")
    foam=Path(os.environ.get('WM_PROJECT_DIR','/opt/openfoam10'))
    lib=Path(os.environ.get('FOAM_LIBBIN',str(foam/'platforms/linux64GccDPInt32Opt/lib')))
    exe=out/'parse'
    cmd=['g++','-std=c++17','-DWM_DP','-DWM_LABEL_SIZE=32','-DNoRepository','-I'+str(ROOT/'common'),
         '-I'+str(foam/'src/OpenFOAM/lnInclude'),'-I'+str(foam/'src/finiteVolume/lnInclude'),
         '-I'+str(foam/'src/meshTools/lnInclude'),'-I'+str(foam/'src/OSspecific/POSIX/lnInclude'),
         str(source),'-L'+str(lib),'-Wl,-rpath,'+str(lib),'-lfiniteVolume','-lmeshTools','-lOpenFOAM','-L'+str(lib/'dummy'),'-Wl,-rpath,'+str(lib/'dummy'),'-lPstream','-o',str(exe)]
    q=subprocess.run(cmd,capture_output=True,text=True)
    assert q.returncode==0,q.stdout+q.stderr
    return exe

@pytest.mark.parametrize('control,scheme,expected',[('yes','false',1),('no','true',0),('','true',0)])
def test_switch_source_and_numeric_limits(parser,tmp_path,control,scheme,expected):
    (tmp_path/'system').mkdir();(tmp_path/'constant').mkdir()
    header=lambda obj:'FoamFile {version 2.0; format ascii; class dictionary; object '+obj+';}\n'
    (tmp_path/'system/controlDict').write_text(header('controlDict')+'application probe; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT .01; writeControl timeStep; writeInterval 1; '+('adjustTimeStep '+control+';' if control else ''))
    (tmp_path/'constant/scheme').write_text(header('scheme')+'precision 64; gasTimeStep {adjustTimeStep '+scheme+'; maxCo .7; maxDeltaT .001;}')
    q=subprocess.run([str(parser),'-case',str(tmp_path)],capture_output=True,text=True)
    assert q.returncode==0,q.stdout+q.stderr
    assert f'RESULT {expected} 0.7 0.001' in q.stdout,q.stdout
