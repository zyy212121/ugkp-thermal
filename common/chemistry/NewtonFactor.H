#pragma once
#include "ChemistryTypes.H"
// Row-equilibrated partial-pivot LU retained for multiple Newton right-hand
// sides. Trial corrections reuse this factorization; no Jacobian is rebuilt.
namespace ugkwp { namespace gasChemistryDetail {
template<class R,int N> UGKWP_GAS_HD inline bool factorNewton(int n,R* a,int* pivot,R* scale,DenseLinearFailure* failure=nullptr){
 if(failure)*failure=DenseLinearFailure{};
 const auto fail=[&](DenseLinearQuantity q,int row,int col,R v){if(failure){failure->quantity=q;failure->row=row;failure->column=col;failure->value=double(v);}return false;};
 if(n<0||n>N||!a||!pivot||!scale)return fail(DenseLinearQuantity::Storage,-1,-1,R(n));
 for(int i=0;i<n;++i){R s=0;for(int j=0;j<n;++j){if(!gasFinite(a[i*N+j]))return fail(DenseLinearQuantity::MatrixEntry,i,j,a[i*N+j]);s=gasThermoDetail::maxValue(s,gasThermoDetail::absValue(a[i*N+j]));}if(s==0)return fail(DenseLinearQuantity::RowScale,i,-1,s);scale[i]=s;for(int j=0;j<n;++j)a[i*N+j]/=s;}
 for(int k=0;k<n;++k){int p=k;for(int i=k+1;i<n;++i)if(gasThermoDetail::absValue(a[i*N+k])>gasThermoDetail::absValue(a[p*N+k]))p=i;pivot[k]=p;
  if(!gasFinite(a[p*N+k])||a[p*N+k]==0)return fail(DenseLinearQuantity::Pivot,p,k,a[p*N+k]);
  if(p!=k)for(int j=0;j<n;++j){R v=a[k*N+j];a[k*N+j]=a[p*N+j];a[p*N+j]=v;}
  for(int i=k+1;i<n;++i){a[i*N+k]/=a[k*N+k];for(int j=k+1;j<n;++j)a[i*N+j]-=a[i*N+k]*a[k*N+j];}
 }return true;
}
template<class R,int N> UGKWP_GAS_HD inline bool solveNewton(int n,const R* a,const int* pivot,const R* scale,R* b,DenseLinearFailure* failure=nullptr){
 for(int i=0;i<n;++i)b[i]/=scale[i];
 for(int k=0;k<n;++k)if(pivot[k]!=k){R v=b[k];b[k]=b[pivot[k]];b[pivot[k]]=v;}
 for(int i=0;i<n;++i)for(int j=0;j<i;++j)b[i]-=a[i*N+j]*b[j];
 for(int i=n-1;i>=0;--i){for(int j=i+1;j<n;++j)b[i]-=a[i*N+j]*b[j];b[i]/=a[i*N+i];if(!gasFinite(b[i])){if(failure){failure->quantity=DenseLinearQuantity::Solution;failure->row=i;failure->column=i;failure->value=double(b[i]);}return false;}}
 return true;
}
}}
