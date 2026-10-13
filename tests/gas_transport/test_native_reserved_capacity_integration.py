"""Selected native GasOnly numerical route with real model/SST configuration."""
import subprocess
from native_gas_integration_fixture import ROOT,native_fixture,MODEL_SETUP,COMPARE_SPECIES

def test_reserved_storage_does_not_select_coupling_or_change_gas_candidate(tmp_path):
    code=native_fixture()+MODEL_SETUP+COMPARE_SPECIES+r'''
int main(){for(int turbulence:{0,3}){
 DeviceState a,b;setupNative(a,0,turbulence);setupNative(b,7,turbulence);
 int count=0,cells[7]={},status[7]={};double particles[13][7];for(int j=0;j<13;++j)for(int i=0;i<7;++i)particles[j][i]=100*j+i;
 b.particleCountDevice=&count;b.pCellId=cells;b.pStatus=status;
 double**ptrs[]={&b.px,&b.py,&b.pz,&b.pux,&b.puy,&b.puz,&b.puxOld,&b.puyOld,&b.puzOld,&b.pT,&b.pTheta,&b.pd,&b.pm};for(int j=0;j<13;++j)*ptrs[j]=particles[j];
 b.particlesMayBePresent=true; // GasOnly remains callable after prior particle routing.
 const double before=a.rhoE[0],dt=1e-6;
 assert(ugkwpGpuResidentStrictAdvanceGasOnly(&a,dt,1e20)==0&&ugkwpGpuResidentStrictAdvanceGasOnly(&b,dt,1e20)==0);
 assert(a.rhoE[0]!=before&&b.particleCapacity==7&&count==0);
 for(int j=0;j<13;++j)for(int i=0;i<7;++i)assert(particles[j][i]==100*j+i);
 for(int i=0;i<7;++i)assert(cells[i]==0&&status[i]==0);
 compareSpeciesFields(a,b);
 auto left=sharedGasTrialDetail::cellFields(&a),right=sharedGasTrialDetail::cellFields(&b);
 for(std::size_t k=0;k<left.size();++k)if(left[k].pointer)assert(std::memcmp(left[k].pointer,right[k].pointer,left[k].count*sizeof(double))==0);
 left=sharedGasTrialDetail::faceFields(&a);right=sharedGasTrialDetail::faceFields(&b);for(std::size_t k=0;k<left.size();++k)assert(std::memcmp(left[k].pointer,right[k].pointer,left[k].count*sizeof(double))==0);
 const double completed=b.rhoE[0];const int beforeCombined=launchCount;assert(ugkwpGpuResidentStrictAdvance(&b,dt,1e20)!=0&&b.rhoE[0]==completed&&last.find("coupling")!=std::string::npos&&launchCount==beforeCombined);
 // Actual launch error returns failure and restores the completed earlier request.
 failLaunch=launchCount+1;assert(ugkwpGpuResidentStrictAdvanceGasOnly(&b,dt,1e20)!=0&&b.rhoE[0]==completed);failLaunch=0;
 releaseSharedGasTrialStorage(a.gasTrial);releaseSharedGasTrialStorage(b.gasTrial);releaseSharedGasSpecies(a.gasSpecies);releaseSharedGasSpecies(b.gasSpecies);assert(allocations==0);
}}
'''
    source=tmp_path/'native.cpp';source.write_text(code);binary=tmp_path/'native'
    b=subprocess.run(['g++','-std=c++17','-O0','-I'+str(ROOT),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),'-I'+str(ROOT/'tests/gas_transport'),str(source),'-o',str(binary)],capture_output=True,text=True)
    assert b.returncode==0,b.stderr
    r=subprocess.run([str(binary)],capture_output=True,text=True);assert r.returncode==0,r.stdout+r.stderr
