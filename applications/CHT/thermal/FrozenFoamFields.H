#pragma once
#include "uncollatedFileOperation.H"
#include "OSstream.H"
#include "regIOobject.H"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <zlib.h>

namespace Foam { namespace gpuThermal {
struct FrozenFoamFile {
    std::string path, bytes;
    bool compressed;
    void publish() const {
        namespace fs=std::filesystem;
        const std::string target=path+(compressed?".gz":""),temp=target+".snapshot.tmp";
        fs::create_directories(fs::path(path).parent_path());
        if(compressed){
            gzFile output=gzopen(temp.c_str(),"wb");
            if(!output)throw std::runtime_error("cannot create "+temp);
            bool ok=true;
            for(std::size_t i=0;ok&&i<bytes.size();){
                const auto n=static_cast<unsigned>(std::min<std::size_t>(bytes.size()-i,1U<<24));
                ok=gzwrite(output,bytes.data()+i,n)==int(n);
                i+=n;
            }
            if(gzclose(output)!=Z_OK||!ok)throw std::runtime_error("cannot finish "+temp);
        }else{
            std::ofstream out(temp,std::ios::binary|std::ios::trunc);
            out.write(bytes.data(),bytes.size());out.close();
            if(!out)throw std::runtime_error("cannot finish "+temp);
        }
        fs::rename(temp,target);
        const auto alternate=compressed?path:path+".gz";
        const auto state=fs::symlink_status(alternate);
        if(fs::is_regular_file(state)||fs::is_symlink(state))fs::remove(alternate);
    }
};
using FrozenFoamFiles=std::vector<FrozenFoamFile>;
inline FrozenFoamFile freezeFoamFile(const regIOobject& io,IOstream::streamFormat format,
    IOstream::versionNumber version,IOstream::compressionType compression){
    const fileName path=io.objectPath();
    std::ostringstream stream(std::ios::binary|std::ios::out);
    OSstream os(stream,path,format,version,compression);
    if(!io.writeHeader(os)||!io.writeData(os))throw std::runtime_error("cannot serialize "+std::string(path));
    IOobject::writeEndDivider(os);
    if(!os.good()||!stream)throw std::runtime_error("cannot complete "+std::string(path));
    return {path,stream.str(),compression==IOstream::COMPRESSED};
}


class SnapshotFileOperation final : public fileOperations::uncollatedFileOperation {
    inline static thread_local FrozenFoamFiles* active_=nullptr;
public:
    SnapshotFileOperation():uncollatedFileOperation(false){}
    const word& type() const override {static const word name("ugkpSnapshot");return name;}
    static void install(){
        if(Pstream::parRun()||fileHandler().type()!=uncollatedFileOperation::typeName)return;
        autoPtr<fileOperation> operation(new SnapshotFileOperation);
        fileHandler(operation);
    }
    class Capture {
        FrozenFoamFiles* previous_;
    public:
        explicit Capture(FrozenFoamFiles& files):previous_(active_){active_=&files;}
        ~Capture(){active_=previous_;}
        Capture(const Capture&)=delete;
    };
    bool writeObject(const regIOobject& io,IOstream::streamFormat format,IOstream::versionNumber version,
        IOstream::compressionType compression,const bool write) const override {
        if(!active_)return uncollatedFileOperation::writeObject(io,format,version,compression,write);
        if(write)active_->push_back(freezeFoamFile(io,format,version,compression));
        return true;
    }
};
}}
