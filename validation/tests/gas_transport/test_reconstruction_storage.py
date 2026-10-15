import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture


@pytest.mark.parametrize('native', [False, True])
def test_reconstruction_consumed_face_storage(tmp_path,native):
    body=r'''
int main(){
for(int mode=0;mode<15;++mode){
 State s;initialise(s);s.gasReconstruction=1;
 int f=0,c=0;
 switch(mode){
 case 0:s.gasSpecies.soundSpeed=nullptr;break;
 case 1:s.faceWeight=nullptr;break;
 case 2:s.Sfx=nullptr;break;
 case 3:s.Sfy=nullptr;break;
 case 4:s.Sfz=nullptr;break;
 case 5:s.magSf=nullptr;break;
 case 6:s.faceNeighbour=nullptr;break;
 case 7:s.faceOwner=nullptr;break;
 case 8:s.faceOwner[0]=s.nCells;break;
 case 9:s.faceNeighbour[0]=s.nCells;break;
 case 10:f=1;s.facePeriodicPair=nullptr;break;
 case 11:f=1;s.facePeriodicPair[f]=2;s.faceNeighbour=nullptr;break;
 case 12:f=1;c=1;s.facePeriodicPair[f]=2;s.faceNeighbour[f]=c;s.facePeriodicDx=nullptr;break;
 case 13:f=1;c=1;s.facePeriodicPair[f]=2;s.faceNeighbour[f]=c;s.facePeriodicDy=nullptr;break;
 case 14:f=1;c=1;s.facePeriodicPair[f]=2;s.faceNeighbour[f]=c;s.facePeriodicDz=nullptr;break;
 }
 GasPrimDevice output{11,12,13,14,15,16},before=output;Real y[2]={21,22};bool evaluated=true;
 const bool ok=reconstructGasMixtureFace(s,c,f,output,y,NATIVE,&evaluated);
 ck(!ok&&!evaluated&&s.gasSpecies.faceStatus[f]==int(ugkwp::GasTransportCode::InvalidStorage),"missing reconstruction face resource");
 ck(std::memcmp(&before,&output,sizeof output)==0&&y[0]==21&&y[1]==22,"failed reconstruction published partial output");
}
}'''.replace('NATIVE','true' if native else 'false')
    compile_probe(tmp_path,fixture()+body,sanitizers=True)


@pytest.mark.parametrize('native', [False, True])
def test_reconstruction_keeps_unconsumed_storage_optional(tmp_path,native):
    body=r'''
int main(){for(int mode=0;mode<4;++mode){
 State s;initialise(s);s.gasReconstruction=mode==0?0:1;
 int f=mode==0?0:1,c=0;
 if(mode>=2){s.facePeriodicPair[f]=2;s.faceNeighbour[f]=1;}
 if(mode==3)s.gasSpecies.mode=ugkwp::GasMode::SingleLegacy;
 GasPrimDevice expected{},actual{};Real ey[2]{},ay[2]{};bool evaluated=false;
 if(mode==3)expected=reconstructGasCellToFace(s,c,f,NATIVE,&evaluated);
 else ck(reconstructGasMixtureFace(s,c,f,expected,ey,NATIVE,&evaluated),"original reconstruction control");
 ck(evaluated,"original control not evaluated");
 if(mode==0){s.faceOwner=nullptr;s.faceNeighbour=nullptr;s.facePeriodicPair=nullptr;s.Cx=nullptr;s.gasSpecies.gradX=nullptr;s.gasSpecies.thermo.species=nullptr;}
 if(mode<=1){s.faceOwner=nullptr;s.faceNeighbour=nullptr;s.gasSpecies.soundSpeed=nullptr;s.faceWeight=nullptr;s.Sfx=s.Sfy=s.Sfz=s.magSf=nullptr;}
 s.facePeriodicDx=s.facePeriodicDy=s.facePeriodicDz=nullptr;
 if(mode==3)s.Tgas=nullptr;
 evaluated=false;
 if(mode==3)actual=reconstructGasCellToFace(s,c,f,NATIVE,&evaluated);
 else ck(reconstructGasMixtureFace(s,c,f,actual,ay,NATIVE,&evaluated),"unused storage became required");
 ck(evaluated&&s.gasSpecies.faceStatus[f]==0,"valid optional-storage reconstruction failed");
 ck(std::memcmp(&actual,&expected,sizeof actual)==0,"optional storage changed reconstructed primitives");
 if(mode!=3)ck(std::memcmp(ay,ey,sizeof ay)==0,"optional storage changed reconstructed composition");
}}'''.replace('NATIVE','true' if native else 'false')
    compile_probe(tmp_path,fixture()+body,sanitizers=True)


@pytest.mark.parametrize('native', [False, True])
def test_face_flux_reports_missing_topology_before_reconstruction(tmp_path,native):
    body=r'''
int main(){for(int mode=0;mode<4;++mode){State s;initialise(s);int f=mode==0?0:1;
 if(mode==0)s.faceNeighbour=nullptr;
 if(mode==1)s.facePeriodicPair=nullptr;
 if(mode==2){s.facePeriodicPair[f]=2;s.faceNeighbour=nullptr;}
 if(mode==3)s.riemannBoundaryKind=nullptr;
 Real mass,px,py,pz,energy;
 const bool ok=computeRiemannGasFaceFluxDevice<false>(s,f,mass,px,py,pz,energy,NATIVE);
 ck(!ok&&s.gasSpecies.faceStatus[f]==int(ugkwp::GasTransportCode::InvalidStorage),"missing face topology not reported");
}}'''.replace('NATIVE','true' if native else 'false')
    compile_probe(tmp_path,fixture()+body,sanitizers=True)
