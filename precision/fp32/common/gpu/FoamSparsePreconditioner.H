#pragma once
#include "SparseConjugateGradient.H"
#include "dictionary.H"
#include "IOstreams.H"
#include <set>
#include <string>
namespace ugkp { namespace gpu {
inline const char* sparsePreconditionerName(SparsePreconditioner value) {
    return value==SparsePreconditioner::Block8?"block8":"jacobi";
}
inline SparsePreconditioner readSparsePreconditioner(const Foam::dictionary& dictionary) {
    const Foam::word name=dictionary.lookupOrDefault<Foam::word>("preconditioner","jacobi");
    if(name=="block8")return SparsePreconditioner::Block8;
    if(name!="jacobi") {
        static std::set<std::string> warned;
        if(warned.insert(name.c_str()).second)
            WarningInFunction << "Unknown sparse preconditioner " << name
                << "; retaining default jacobi." << Foam::nl;
    }
    return SparsePreconditioner::Jacobi;
}
}}
