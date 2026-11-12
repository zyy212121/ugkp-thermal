// Draft synthetic dictionary fixture. NOT COMPILED OR RUN.
// Only snapshot mode writes its disposable fixture case's constant/scheme.
#include "argList.H"
#include "Time.H"
#include "../../common/FoamSolverScheme.H"
#include <fstream>
#include <iostream>
int main(int argc,char** argv) {
    Foam::argList::addOption("scenario","name","absent, empty, precedence, snapshot, reload");
    #include "setRootCase.H"
    #include "createTime.H"
    const Foam::word mode=args.optionLookupOrDefault<Foam::word>("scenario","precedence");
    using ugkp::solverControls::startupScheme;
    const auto& s=startupScheme(runTime);
    const bool fp32=ugkp::precision::compiledBits==32;
    const Foam::label precisionModule=fp32?323:643,precisionRegion=fp32?324:644;
    int failures=0;
    auto check=[&](bool ok,const char* name) {
        std::cout<<"LOADER_CHECK name="<<name<<" pass="<<ok<<"\n";
        if(!ok)++failures;
    };
    check(&s==&startupScheme(runTime),"same_case_snapshot_identity");
    auto read=[&](const char* key,Foam::label inherited,const Foam::word& region=Foam::word::null) {
        return s.value<Foam::label>("fixture.inner",Foam::word(key),inherited,"typedDefault",region);
    };
    check(read("defaultOnly",11)==11,"effective_default_fallback");
    check(read("missingTypedKey",22)==22,"typed_default_for_missing_key");
    if(mode=="absent"||mode=="empty") {
        check(read("priority",22)==22,"no_override_compatibility");
        check(read("priority",22,Foam::word("solidA"))==22,"region_no_override_compatibility");
    } else if(mode=="reload") {
        check(read("priority",22)==77,"new_process_reads_replacement");
        check(read("priority",22,Foam::word("solidA"))==88,"new_process_region_replacement");
    } else {
        check(read("moduleOnly",22)==55,"global_module_beats_typed_default");
        check(read("priority",22)==precisionModule,"precision_module_beats_global_module");
        check(read("priority",22,Foam::word("solidA"))==precisionRegion,"precision_region_beats_global_region");
        check(read("regionWins",22,Foam::word("solidA"))==45,"global_region_beats_precision_module");
        check(read("priority",22,Foam::word("solidB"))==46,"other_region_has_independent_global_override");
        check(read("priority",22,Foam::word("solidC"))==(fp32?325:645),"precision_only_region");
        check(read("moduleOnly",22,Foam::word("solidA"))==55,"partial_region_inherits_module");
        check(read("priority",22,Foam::word("unknownRegion"))==precisionModule,"unknown_region_precision_module_fallback");
        if(mode=="snapshot") {
            const auto path=runTime.path()/runTime.constant()/"scheme";
            std::ofstream out(path.c_str());
            out<<"FoamFile { version 2.0; format ascii; class dictionary; object scheme; }\n"
               <<"fixture { inner { priority 77; moduleOnly 66; } }\n"
               <<"regions { solidA { fixture { inner { priority 88; } } } }\n";
            out.close();
            check(bool(out),"replacement_file_written");
            check(&s==&startupScheme(runTime),"snapshot_identity_after_file_change");
            check(read("priority",22)==precisionModule,"precision_module_snapshot_unchanged");
            check(read("priority",22,Foam::word("solidA"))==precisionRegion,"precision_region_snapshot_unchanged");
        }
    }
    std::cout<<"LOADER_SUMMARY scenario="<<mode<<" failures="<<failures<<"\n";
    return failures?1:0;
}
