#include "core/HostAuditReduction.H"
#include "DiagnosticsHost.H"
#include "argList.H"
#include <cfenv>
#include <fenv.h>
#include <iostream>
#include <cstdlib>
void require(bool value,const char*message){if(!value){std::cerr<<message<<'\n';std::exit(2);}}
int main(int argc,char**argv){Foam::argList args(argc,argv);feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);const int mask=fegetexcept();std::feclearexcept(FE_ALL_EXCEPT);
using namespace chmt;const long double m=std::numeric_limits<Real>::max();const int ranks=Foam::Pstream::nProcs();
const long double large=hostAuditGlobalSum(hostAuditAdd(m,m));
require(std::isfinite(large)&&std::abs(large/(2*m*ranks)-1)<1e-14L,"wide global sum lost finite audit");
require(std::isinf(hostAuditNarrow(large)),"safe audit narrowing missing");
require(hostAuditGlobalSum(1.25L)==1.25L*ranks,"ordinary audit sum changed");
const long double product=hostAuditGlobalSum(m*m);require(std::isfinite(product)&&std::isinf(hostAuditNarrow(product)),"wide audit product lost");
const long double sign=Foam::Pstream::myProcNo()%2?-1:1;
const long double cancellation=hostAuditGlobalSum(sign*m);require(ranks==1?cancellation==m:cancellation==0,"global signed cancellation failed");
const auto infinity=std::numeric_limits<long double>::infinity();
require(std::isnan(hostAuditAdd(infinity,-infinity)),"opposite infinities need quiet sentinel");
require(std::isinf(hostAuditAdd(std::numeric_limits<long double>::max(),std::numeric_limits<long double>::max())),"wide sum overflow not classified");
const long double nonfinite=hostAuditGlobalSum(Foam::Pstream::myProcNo()==0?infinity:0);
require(std::isinf(nonfinite)&&nonfinite>0,"global audit infinity lost");
const long double mixed=hostAuditGlobalSum(Foam::Pstream::myProcNo()==0?infinity:-infinity);require(ranks==1?std::isinf(mixed):std::isnan(mixed),"mixed global infinity was not classified");
ugkp::diagnostics::DeviceContext d{};ugkp::diagnostics::recordWarning(&d,ugkp::diagnostics::makeRecord(ugkp::diagnostics::Code::ConservationResidual,777101,0,0,ugkp::diagnostics::EntityKind::None,-1,-1,hostAuditNarrow(large)));
const auto snapshot=ugkp::diagnostics::takeSnapshot(d);require(!snapshot.hasFatal&&!snapshot.warnings.empty(),"audit became fatal");
require(fegetexcept()==mask&&!std::fetestexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW),"audit changed FPE mask/flags");
if(Foam::Pstream::master())std::cout<<"PASS host audit ranks="<<ranks<<'\n';
}
