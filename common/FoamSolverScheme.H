#ifndef ugkp_FoamSolverScheme_H
#define ugkp_FoamSolverScheme_H
#include "argList.H"
#include "Time.H"
#include "IOdictionary.H"
#include "OSspecific.H"
#include "PrecisionBuild.H"
#include <map>
#include <memory>
#include <set>
#include <string>

namespace ugkp { namespace solverControls {

// Case-level numerical controls. The dictionary is read only at startup;
// device owners receive their existing typed controls, never this dictionary.
// Top-level precision selects the launcher backend. One explicit numerical
// setting applies to either backend; omitted settings retain its typed defaults.
class Scheme {
    Foam::dictionary values_;
    Foam::fileName path_;
    mutable std::set<std::string> reported_;
    static Foam::dictionary nested(const Foam::dictionary& root,const std::string& path) {
        Foam::dictionary current(root);
        std::size_t begin=0;
        while(begin<path.size()) {
            const auto end=path.find('.',begin);
            const auto key=path.substr(begin,end==std::string::npos?end:end-begin);
            current=current.subOrEmptyDict(Foam::word(key));
            if(end==std::string::npos)break;
            begin=end+1;
        }
        return current;
    }
public:
    explicit Scheme(const Foam::Time& time):path_(time.path()/time.constant()/"scheme") {
        Foam::IOdictionary input(Foam::IOobject("scheme",time.constant(),time,
            Foam::IOobject::READ_IF_PRESENT,Foam::IOobject::NO_WRITE,false));
        values_=input;
        Foam::Info<<"SCHEME_STARTUP file="<<path_<<" present="<<Foam::isFile(path_)
            <<" precision="<<ugkp::precision::compiledBits
            <<" readOnce=1 precedence=region,module,backendDefault"<<Foam::nl;
    }
    Foam::dictionary section(const std::string& path,
        const Foam::word& region=Foam::word::null) const {
        Foam::dictionary result=nested(values_,path);
        if(!region.empty()) {
            const auto regions=values_.subOrEmptyDict("regions");
            result.merge(nested(regions.subOrEmptyDict(region),path));
        }
        return result;
    }
    Foam::dictionary module(const std::string& path,
        const Foam::word& region=Foam::word::null) const {return section(path,region);}
    template<class T> void report(const std::string& path,const Foam::word& key,
        const T& value,const bool overridden,const char* inheritedSource,
        const Foam::word& region=Foam::word::null) const {
        const std::string identity=std::string(region.c_str())+":"+path+"."+key.c_str();
        if(!reported_.insert(identity).second)return;
        Foam::Info<<"SCHEME_VALUE module="<<path.c_str()<<" key="<<key;
        if(!region.empty())Foam::Info<<" region="<<region;
        Foam::Info<<" effective="<<value<<" source=";
        if(overridden) {
            const bool regional=!region.empty()
                &&nested(values_.subOrEmptyDict("regions").subOrEmptyDict(region),path).found(key,false,false);
            Foam::Info<<"constant/scheme:";
            if(regional)Foam::Info<<"regions."<<region<<'.';
            Foam::Info<<path.c_str()<<'.'<<key;
        } else Foam::Info<<inheritedSource;
        Foam::Info<<Foam::nl;
    }
    template<class T> bool read(const std::string& path,const Foam::word& key,T& value,
        const char* inheritedSource="effectiveDefault",
        const Foam::word& region=Foam::word::null) const {
        const auto controls=section(path,region);
        const bool overridden=controls.readIfPresent(key,value,false,false);
        report(path,key,value,overridden,inheritedSource,region);
        return overridden;
    }
    template<class T> T value(const std::string& path,const Foam::word& key,T inherited,
        const char* inheritedSource="effectiveDefault",
        const Foam::word& region=Foam::word::null) const {
        read(path,key,inherited,inheritedSource,region);return inherited;
    }
};

// One immutable snapshot for each case root, including applications that create
// several region registries. Own only dictionary bytes, not a Time reference.
inline const Scheme& startupScheme(const Foam::Time& time) {
    static std::map<std::string,std::unique_ptr<const Scheme>> cases;
    const std::string key=(time.path()/time.constant()).c_str();
    auto it=cases.find(key);
    if(it==cases.end())it=cases.emplace(key,std::unique_ptr<const Scheme>(new Scheme(time))).first;
    return *it->second;
}
}}
#endif
