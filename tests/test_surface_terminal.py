"""Bounded actual surface producer evidence; no geometry or GPU execution claim."""
from pathlib import Path
import subprocess, io, tarfile
ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'applications/CHMT'
BASE='89ed80b59020d57c444143245de33fc546846e5e'


def compile_run(tmp_path,name,code,includes=(),owner=None):
    source=tmp_path/(name+'.cpp');source.write_text(code);exe=tmp_path/(name+'.exe')
    cmd=['g++','-std=c++14','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror','-pedantic',
         '-Wno-misleading-indentation','-Wno-mismatched-new-delete',
         *['-I'+str(p) for p in includes],'-I'+str(ROOT/'tests/fixtures/surface_terminal'),
         '-I'+str(APP),'-I'+str(ROOT/'common'),str(source),*([str(owner)] if owner else []),'-o',str(exe)]
    build=subprocess.run(cmd,text=True,capture_output=True)
    (tmp_path/(name+'.build.log')).write_text(' '.join(cmd)+'\n'+build.stdout+build.stderr)
    assert build.returncode==0,build.stderr
    result=subprocess.run([str(exe)],capture_output=True,text=True,timeout=60)
    (tmp_path/(name+'.run.log')).write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stderr
    return result.stdout


def instrument(source):
    source=source.replace('namespace chmt {','static bool forceWindow=false,forceNoBracket=false,forceFlatResidual=false;static double failAbove=1e100,lastCompleteT=0,lastResidualY[chmt::Ns]{};static unsigned long long surfaceCalls=0,residualCalls=0;static int maxGrid=0;\nnamespace chmt {',1)
    a=source.index('CHMT_HD inline bool evaluateSurfaceTemperature(');b=source.index('// Never accept a root',a)
    body=source[a:b];start=body.index('{');body=body[:start+1]+'++surfaceCalls;'+body[start+1:]
    body=body.replace('    out=result; return true;','    if((forceWindow&&(topT<699||topT>714))||topT>failAbove)return false;\n    lastCompleteT=topT;out=result; return true;',1)
    source=source[:a]+body+source[b:]
    a=source.index('CHMT_HD inline bool compositionResidual(');b=source.index('CHMT_HD inline bool interfaceComposition(',a)
    body=source[a:b].replace('    return true;', '    ++residualCalls;for(int s=0;s<Ns;++s){lastResidualY[s]=Y[s];if(forceFlatResidual)residual[s]=1;}\n    return true;',1)
    source=source[:a]+body+source[b:]
    source=source.replace('if (valid && previousValid','if (valid && previousValid && !forceNoBracket')
    return source.replace('for (int subdivisions=32; subdivisions<=512; subdivisions*=4) {',
        'for (int subdivisions=32; subdivisions<=512; subdivisions*=4) { maxGrid=subdivisions;')


def test_old_success_late_grid_exact_fields_and_counts(tmp_path):
    baseline=tmp_path/'base';baseline.mkdir()
    archive=subprocess.check_output(['git','archive',BASE,'applications/CHMT','common'],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:tar.extractall(baseline,filter='data')
    output=[]
    for label,root in [('base',baseline),('current',ROOT)]:
        app=root/'applications/CHMT';overlay=tmp_path/(label+'-overlay/ablation');overlay.mkdir(parents=True)
        (overlay/'InterfaceMath.H').write_text(instrument((app/'ablation/InterfaceMath.H').read_text()).replace('\"../../../common/gasWall/WallModel.H\"','\"'+str(root/'common/gasWall/WallModel.H')+'\"'))
        (overlay/'CpuSurfaceInterface.H').write_text((app/'ablation/CpuSurfaceInterface.H').read_text())
        owner=tmp_path/(label+'-owner.C');owner.write_text((app/'ablation/CpuSurfaceInterface.C').read_text())
        # One TU shares instrumentation counters with the real CPU producer.
        code=(ROOT/'tests/fixtures/surface_terminal/parity.cpp').read_text()+'\n#include "'+str(owner)+'"\n'
        output.append(compile_run(tmp_path,label,code,[overlay.parent,app,root/'common']))
    assert output[0]==output[1]


def test_terminal_evaluated_candidates_and_cpu_tail(tmp_path):
    overlay=tmp_path/'overlay/ablation';overlay.mkdir(parents=True)
    source=instrument((APP/'ablation/InterfaceMath.H').read_text()).replace(
        '"../../../common/gasWall/WallModel.H"','"'+str(ROOT/'common/gasWall/WallModel.H')+'"')
    (overlay/'InterfaceMath.H').write_text(source)
    (overlay/'CpuSurfaceInterface.H').write_text((APP/'ablation/CpuSurfaceInterface.H').read_text())
    code=(ROOT/'tests/fixtures/surface_terminal/terminal.cpp').read_text()+'\n#include "'+str(APP/'ablation/CpuSurfaceInterface.C')+'"\n'
    compile_run(tmp_path,'terminal',code,[overlay.parent])


def test_context_consumer_supplied_values_and_missing_storage(tmp_path):
    header=tmp_path/'WallClosureHost.H'
    source=(APP/'ablation/WallClosureHost.H').read_text().replace('class WallClosureHost {','class WallClosureHost { public:',1)
    source=source.replace('"../../../common/gasWall/WallGeometryBuilder.H"','"'+str(ROOT/'common/gasWall/WallGeometryBuilder.H')+'"')
    header.write_text(source)
    code=r'''
#include "HEADER"
#include <cassert>
int main(){using namespace chmt;WallClosureHost owner;
 owner.ready_=true;owner.cache_.descriptors.resize(1);auto& d=owner.cache_.descriptors[0];d.matchingDistance=.7;d.ownerDistance=.2;
 std::vector<GasWallMatchingSample> samples(1);samples[0].pressure=123;samples[0].mechanicalPressure=-4;samples[0].state.temperature=500;
 auto config=defaultWallModel();config.model=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport;
 std::vector<GasWallClosureContext> out;std::string error;bool numerical=false;
 assert(!owner.contexts(samples,{},config,out,error,&numerical)&&numerical);
 assert(owner.contexts<true>(samples,{},config,out,error,&numerical));
 assert(out[0].mechanicalPressure==-4&&out[0].matchingPressure==123&&out[0].matchingDistance==.7&&out[0].ownerDistance==.2);
 config.model=ugkwp::gaswall::BoundaryLayerModel::ReactingSst;
 for(int nodes:{-7,129}){config.nodes=nodes;assert(!owner.contexts(samples,{},config,out,error,&numerical)&&numerical);
  assert(owner.contexts<true>(samples,{},config,out,error,&numerical));assert(out[0].workspace&&out[0].workspaceCapacity==(nodes<24?24:128));}
 // No wall solver is run with a deliberately out-of-range node count.
 samples.clear();assert(owner.contexts<true>(samples,{},config,out,error,&numerical)&&out.empty());
 samples.resize(1);owner.cache_.descriptors.clear();assert(!owner.contexts<true>(samples,{},config,out,error,&numerical)&&!numerical);
}
'''.replace('HEADER',str(header))
    compile_run(tmp_path,'contexts',code)
