"""Output-event profiles preserve the actual solved nodes and species ordering."""
from test_wall_model import compile_run


def test_actual_profile_nodes_and_dependent_species(tmp_path):
    compile_run(tmp_path, r'''
 Model m; WallModelConfig<double> c; c.nodes=24;
 WallWorkspace<double,2,24> w; WallOutput<double,2> o; WallStatus status;
 m.in.matching.massFraction[0]=.2; m.in.matching.massFraction[1]=.8;
 m.in.velocity[0]=3; m.in.velocity[2]=-2;
 assert(evaluateWallModel(m.in,c,w,o,status));
 for(int node=0;node<c.nodes;++node) {
   WallProfileNode<double,2> p;
   assert(reactingWallProfileNode(m.in,c,w,node,p));
   assert(p.y==w.y[node]); assert(p.temperature==m.in.temperature+w.state[node][2]);
   assert(p.massFraction[0]==w.state[node][3]);
   assert(p.massFraction[1]==1-w.state[node][3]);
   assert(near(p.velocity[0],m.in.velocity[0]+w.state[node][0]));
   assert(near(p.velocity[2],m.in.velocity[2]-w.state[node][1]));
 }
 ''')


def test_capture_on_off_preserves_physical_result_and_all_species(tmp_path):
    from test_wall_model import BASE, ROOT
    import subprocess
    source = tmp_path / 'capture.cpp'
    source.write_text(BASE + r'''
#include "common/gasTransport/GasBoundaryLayerModelState.H"
#include <cstring>
int main(){
 Model m;WallModelConfig<double> c;c.nodes=24;
 m.in.model.mode=GasMode::MixtureChemistry;
 WallWorkspace<double,2,24> off,on;WallOutput<double,2> a,b;WallStatus sa,sb;
 double data[24*9];for(double& x:data)x=-12345;
 assert(ugkwp::evaluateGasBoundaryLayerWorkspace<false>(m.in,c,off,a,sa,data));
 for(double x:data)assert(x==-12345); // default specialization performs no capture
 assert(ugkwp::evaluateGasBoundaryLayerWorkspace<true>(m.in,c,on,b,sb,data));
 assert(std::memcmp(&a,&b,sizeof(a))==0);
 assert(std::memcmp(off.state,on.state,sizeof(off.state))==0);
 assert(sa.code==sb.code&&sa.iteration==sb.iteration&&sa.residual==sb.residual);
 for(int node=0;node<24;++node){WallProfileNode<double,2> p;
   assert(reactingWallProfileNode(m.in,c,on,node,p));const double* row=data+9*node;
   assert(row[0]==on.y[node]);assert(row[4]==p.temperature);
   assert(row[7]==p.massFraction[0]&&row[8]==p.massFraction[1]);
   assert(near(row[7]+row[8],1));
 }
}
''')
    exe=tmp_path/'capture'
    result=subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT),'-I'+str(ROOT/'common'),str(source),'-o',str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    result=subprocess.run([str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_csv_metadata_species_names_and_distinct_stage_write_times(tmp_path):
    from test_wall_model import ROOT
    import subprocess
    import csv
    source=tmp_path/'csv.cpp'
    source.write_text(r'''
#include "common/gasTransport/GasBoundaryLayerProfileOutput.H"
#include <iostream>
int main(){std::vector<int> faces{17};std::vector<double> centres{.1,.2,.3},geometry{0,1,0,.1,.2,1,.1};
 std::vector<std::string> species{"air","CO2"};double rows[24*9];
 for(int node=0;node<24;++node)for(int col=0;col<9;++col)rows[9*node+col]=100*node+col;
 ugkwp::writeGasBoundaryLayerProfiles(std::cout,faces,centres,geometry,species,24,rows,1.125,1.25);
}
''')
    exe=tmp_path/'csv'
    subprocess.run(['g++','-std=c++17','-I'+str(ROOT),str(source),'-o',str(exe)],check=True)
    output=subprocess.run([str(exe)],capture_output=True,text=True,check=True).stdout
    rows=list(csv.DictReader(output.splitlines()))
    assert len(rows)==24
    for i,row in enumerate(rows):
        assert int(row['face'])==17 and int(row['node'])==i
        assert float(row['faceX'])==.1 and float(row['normalY'])==1
        assert float(row['stageTime'])==1.125 and float(row['outputTime'])==1.25
        assert float(row['Y_air'])==100*i+7 and float(row['Y_CO2'])==100*i+8


def test_profile_switch_is_startup_only_and_download_stays_in_write_path():
    from test_wall_model import ROOT
    adapter=(ROOT/'applications/gasUGKP/gpu/BoundaryLayerInput.H').read_text()
    driver=(ROOT/'applications/gasUGKP/diluteUgkwpFoam.C').read_text()
    assert adapter.count('lookupOrDefault<bool>("writeProfiles",false)')==1
    constructor=adapter.split('    BoundaryLayerInput(',1)[1].split('    bool active()',1)[0]
    assert 'lookupOrDefault<bool>("writeProfiles",false)' in constructor
    write=adapter.split('    void writeDiagnostics(',1)[1]
    assert '(this->*profileWriter_)(time,handle)' in write
    assert 'profileWriter_=&BoundaryLayerInput::skipProfiles' in adapter
    branch=driver.split('if (runTime.writeTime())',1)[1]
    assert 'resident.writeBoundaryLayerDiagnostics(boundaryLayerInput, runTime);' in branch
    assert driver.count('resident.writeBoundaryLayerDiagnostics(')==1
