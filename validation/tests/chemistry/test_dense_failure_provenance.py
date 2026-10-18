"""Run the production CPU dense solve and inspect its first failing quantity."""
import subprocess
from validation.tools.paths import SOURCE_ROOT


def test_dense_failure_provenance(tmp_path):
    source = tmp_path / 'dense.cpp'
    source.write_text(r'''
#include "common/operators/advanceGasChemistryKernel.cuh"
#include <cassert>
#include <limits>
using namespace ugkwp;
int main(){
 double a[4]={1,0,0,0},b[2]={1,2}; DenseLinearFailure f; bool done=true;
 assert((!gasChemistryDetail::solveDense<double,2>(2,a,b,&done,&f)));
 assert(!done && f.quantity==DenseLinearQuantity::RowScale && f.row==1 && f.value==0);
 double c[4]={1,2,2,4},d[2]={1,2};
 assert((!gasChemistryDetail::solveDense<double,2>(2,c,d,nullptr,&f)));
 assert(f.quantity==DenseLinearQuantity::Pivot && f.row==1 && f.column==1 && f.value==0);
 double e[4]={1,0,std::numeric_limits<double>::infinity(),1},g[2]={1,2};
 assert((!gasChemistryDetail::solveDense<double,2>(2,e,g,nullptr,&f)));
 assert(f.quantity==DenseLinearQuantity::MatrixEntry && f.row==1 && f.column==0 && std::isinf(f.value));
 double h[4]={1,0,0,1},i[2]={1,std::numeric_limits<double>::quiet_NaN()};
 assert((!gasChemistryDetail::solveDense<double,2>(2,h,i,nullptr,&f)));
 assert(f.quantity==DenseLinearQuantity::RightHandSide && f.row==1 && std::isnan(f.value));
 double j[4]={2,0,0,4},k[2]={6,8};
 assert((gasChemistryDetail::solveDense<double,2>(2,j,k,&done,&f)));
 assert(done && f.quantity==DenseLinearQuantity::None && k[0]==3 && k[1]==2);
 struct State : GasStateView<double,double,int> { GasSpeciesState<double,2> gasSpecies; } state;
 ugkp::diagnostics::DeviceContext diagnostics{}; state.diagnostics=&diagnostics; state.nCells=1;
 ChemistryStatus status;status.code=ChemistryCode::SingularJacobian;
 status.linearFailure={DenseLinearQuantity::MatrixEntry,1,0,std::numeric_limits<double>::infinity()};
 gasChemistryAdapterDetail::recordFailure(state,0,status);
 const auto& record=diagnostics.firstFatal;
 assert(ugkp::diagnostics::hasFatal(&diagnostics));
 assert(record.sourceSiteId==2207 && record.code==ugkp::diagnostics::Code::NonFinite);
 assert(record.component==1 && record.reference==0 && std::isinf(record.observed));
 assert(record.metadata==unsigned(DenseLinearQuantity::MatrixEntry));
}
''')
    binary = tmp_path / 'dense'
    subprocess.run(['g++', '-std=c++17', '-I', str(SOURCE_ROOT), str(source), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
