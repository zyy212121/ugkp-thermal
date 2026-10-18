#pragma once
#include "ChemistryTypes.H"
// Row-equilibrated partial-pivot LU retained for multiple Newton right-hand
// sides. Trial corrections reuse this factorization; no Jacobian is rebuilt.
namespace ugkwp { namespace gasChemistryDetail {
// Detect an unrepresentable quotient without executing a trapping division.
// Normalized same-precision mantissas cannot lie in the half-ulp gap between
// max() and its overflow boundary, so the exponent/tie comparison is exact.
template<class R> UGKWP_GAS_HD inline bool newtonQuotient(R a,R b,R& result){
 if(!gasFinite(a)||!gasFinite(b)||b==0){result=b==0?std::numeric_limits<R>::infinity():(!gasFinite(a)?a:b);return false;}
 if(a==0){result=a/b;return true;}
 int ea=0,eb=0;const R ma=::frexp(gasThermoDetail::absValue(a),&ea),mb=::frexp(gasThermoDetail::absValue(b),&eb);
 const int exponent=ea-eb;
 if(exponent>std::numeric_limits<R>::max_exponent||(exponent==std::numeric_limits<R>::max_exponent&&ma>=mb)){
  result=(a<0)!=(b<0)?-std::numeric_limits<R>::infinity():std::numeric_limits<R>::infinity();return false;
 }
 result=a/b;return gasFinite(result);
}
template<class R> UGKWP_GAS_HD inline bool newtonDifferenceProduct(R a,R b,R c,R& result){
 if(!gasFinite(a)||!gasFinite(b)||!gasFinite(c)){result=!gasFinite(a)?a:(!gasFinite(b)?b:c);return false;}
 const R limit=::ldexp(R(1),std::numeric_limits<R>::max_exponent/4);
 if(gasThermoDetail::absValue(a)<=std::numeric_limits<R>::max()/R(4)&&gasThermoDetail::absValue(b)<limit&&gasThermoDetail::absValue(c)<limit){result=a-b*c;return gasFinite(result);}
 // Power-of-two scaling plus FMA preserves a finite cancellation even when
 // forming b*c first would overflow. Only a genuinely unrepresentable final
 // result is reported as infinity; the normal path keeps its old arithmetic.
 int ea=0,eb=0,ec=0;const R ma=::frexp(a,&ea),mb=::frexp(b,&eb),mc=::frexp(c,&ec);
 const int productExponent=eb+ec,exponent=ea>productExponent?ea:productExponent;
 const R scaled=::fma(-::ldexp(mb,productExponent-exponent),mc,::ldexp(ma,ea-exponent));
 if(scaled==0){result=scaled;return true;}
 int shift=0;(void)::frexp(scaled,&shift);
 if(exponent+shift>std::numeric_limits<R>::max_exponent){result=scaled<0?-std::numeric_limits<R>::infinity():std::numeric_limits<R>::infinity();return false;}
 result=::ldexp(scaled,exponent);return gasFinite(result);
}
template<class R,int N> UGKWP_GAS_HD inline bool factorNewton(int n,R* a,int* pivot,R* scale,DenseLinearFailure* failure=nullptr){
 if(failure)*failure=DenseLinearFailure{};
 const auto fail=[&](DenseLinearQuantity q,int row,int col,R v){if(failure){failure->quantity=q;failure->row=row;failure->column=col;failure->value=double(v);}return false;};
 if(n<0||n>N||!a||!pivot||!scale)return fail(DenseLinearQuantity::Storage,-1,-1,R(n));
 for(int i=0;i<n;++i){R s=0;for(int j=0;j<n;++j){if(!gasFinite(a[i*N+j]))return fail(DenseLinearQuantity::MatrixEntry,i,j,a[i*N+j]);s=gasThermoDetail::maxValue(s,gasThermoDetail::absValue(a[i*N+j]));}if(s==0)return fail(DenseLinearQuantity::RowScale,i,-1,s);scale[i]=s;for(int j=0;j<n;++j)a[i*N+j]/=s;}
 for(int k=0;k<n;++k){int p=k;for(int i=k+1;i<n;++i)if(gasThermoDetail::absValue(a[i*N+k])>gasThermoDetail::absValue(a[p*N+k]))p=i;pivot[k]=p;
  if(!gasFinite(a[p*N+k])||a[p*N+k]==0)return fail(DenseLinearQuantity::Pivot,p,k,a[p*N+k]);
  if(p!=k)for(int j=0;j<n;++j){R v=a[k*N+j];a[k*N+j]=a[p*N+j];a[p*N+j]=v;}
  for(int i=k+1;i<n;++i){a[i*N+k]/=a[k*N+k];for(int j=k+1;j<n;++j){R value;if(!newtonDifferenceProduct(a[i*N+j],a[i*N+k],a[k*N+j],value))return fail(DenseLinearQuantity::MatrixEntry,i,j,value);a[i*N+j]=value;}}
 }return true;
}
template<class R,int N> UGKWP_GAS_HD inline bool solveNewton(int n,const R* a,const int* pivot,const R* scale,R* b,DenseLinearFailure* failure=nullptr){
 const auto fail=[&](DenseLinearQuantity q,int row,int col,R value){if(failure){failure->quantity=q;failure->row=row;failure->column=col;failure->value=double(value);}return false;};
 if(n<0||n>N||!a||!pivot||!scale||!b)return fail(DenseLinearQuantity::Storage,-1,-1,R(n));
 for(int k=0;k<n;++k)if(pivot[k]<k||pivot[k]>=n)return fail(DenseLinearQuantity::Storage,k,-1,R(pivot[k]));
 for(int i=0;i<n;++i){if(!gasFinite(b[i]))return fail(DenseLinearQuantity::RightHandSide,i,-1,b[i]);R value;
  if(!newtonQuotient(b[i],scale[i],value))return fail(DenseLinearQuantity::ScaledRightHandSide,i,-1,value);
  b[i]=value;}
 for(int k=0;k<n;++k)if(pivot[k]!=k){R v=b[k];b[k]=b[pivot[k]];b[pivot[k]]=v;}
 for(int i=0;i<n;++i)for(int j=0;j<i;++j){R value;if(!newtonDifferenceProduct(b[i],a[i*N+j],b[j],value))return fail(DenseLinearQuantity::Solution,i,j,value);b[i]=value;}
 for(int i=n-1;i>=0;--i){for(int j=i+1;j<n;++j){R value;if(!newtonDifferenceProduct(b[i],a[i*N+j],b[j],value))return fail(DenseLinearQuantity::Solution,i,j,value);b[i]=value;}
  R value;if(!newtonQuotient(b[i],a[i*N+i],value))return fail(DenseLinearQuantity::Solution,i,i,value);b[i]=value;}
 return true;
}
}}
