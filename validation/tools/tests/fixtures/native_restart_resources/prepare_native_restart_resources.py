"""Link an actual CUDA resident API resource regression to existing native objects."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


PREFIX = 'ugkwpGpuResidentStrict'
WIDTHS = dict(pColdNodeSpecificEnthalpy=8, pColdRingSolidMass=8,
              pCold2DNodeSpecificEnthalpy=64, pCold2DRingContactAge=8)
CREATE = dict(
    nCells='1', nFaces='6', nInternalFaces='0', nCellPlanes='6',
    particleCapacity='2', maxFaceWalkHops='16', injectionParcelMass='1e-6',
    rngSeed='12345', gammaGas='1.4', Rgas='287', rhoSolid='1000',
    solveParticleTemperature='0', particleGasHeatTransferModelId='1',
    particleThermalRho='1000', particleCp='1000', gasMu='1e-5', gasPr='.7',
    dragModelId='0', dragModel='0', dragResidualRe='1e-12',
    dragParameter0='0', dragParameter1='0', dragParameter2='0', dragParameter3='0',
    gravityX='0', gravityY='0', gravityZ='0', particleDiameterFallback='.01',
    particleDiameterMin='.01', particleDiameterMax='.01', particleDiameterSigma='0',
    injectionTheta='0', rhoMin='1e-12', TgasMin='1', epsSMin='1e-12',
    thetaMin='0', TpMin='1', TpMax='1e5', collisionalPressureEnabled='0',
    collisionalRestitution='.8', pressureKickFraction='1',
    jammingPressureEnabled='0', packingFraction='.6', packingProjectionIterations='1',
    gasFluxScheme='2', gasReconstruction='0', gasLimiter='0', gasTimeIntegrator='1',
    turbulenceModel='0', lesDeltaCoeff='1',
    turbulentPrandtl='.9', waleCw='.5', smagorinskyCs='.17',
    csrCellLocalPathEnabled='0', csrHeavyReductionMode='0', csrHeavyAutoInterval='100',
    particleBlockThreads='128', reductionBlockThreads='128', csrWarpAggregatedBinning='0',
    csrSplitPreDirectoryEnabled='0', handle='&handle',
)


def parameters(header, name):
    match = re.search(r'extern "C" int\s+' + PREFIX + name + r'\s*\((.*?)\);', header, re.S)
    if not match:
        raise ValueError('native API declaration unavailable: ' + name)
    return [tuple(part.strip().rsplit(' ', 1)) for part in match[1].split(',')]


def generate(root, app):
    api = root / 'applications' / app / 'gpu/GpuBackendApi.H'
    header = api.read_text()
    create = parameters(header, 'Create')
    unknown = {name for _, name in create} - CREATE.keys()
    if unknown:
        raise ValueError('unconfigured native creation arguments: ' + str(sorted(unknown)))
    fields = [(typ.removeprefix('const ').replace('*', '').strip(), name)
              for typ, name in parameters(header, 'UploadParticleRestartMirror')[2:]]
    download = parameters(header, 'DownloadParticleRestartMirror')
    if [name for _, name in download[3:]] != [name for _, name in fields]:
        raise ValueError('native upload/download field inventories differ')
    declarations = []
    for typ, name in fields:
        width = WIDTHS.get(name, 1)
        declarations.append(f'std::vector<{typ}> in_{name}({3*width},0),out_{name}({3*width},0);')
    uploads = []
    for _, name in fields:
        uploads.append('nullptr' if name.startswith('pCold') else
                       f'(missing=="{name}"?nullptr:in_{name}.data())')
    download_args = ','.join('out_' + name + '.data()' for _, name in fields)
    active_fields = [(typ, name) for typ, name in fields if not name.startswith('pCold')]
    physical_checks = ''.join(
        f'require(std::memcmp(in_{name}.data(),out_{name}.data(),2*sizeof({typ}))==0,"roundtrip {name}");'
        for typ, name in active_fields)
    snapshot = ''.join(
        f'append(bytes,out_{name}.data(),2*{WIDTHS.get(name,1)});' for _, name in fields)
    mesh = parameters(header, 'UploadMesh')[1:]
    mesh_data = ''.join(f'std::vector<{typ.removeprefix("const ").replace("*", "").strip()}> mesh_{name}(6,0);'
                        for typ, name in mesh)
    mesh_args = ','.join('mesh_' + name + '.data()' for _, name in mesh)
    cold = 'pStuck' in dict((name, typ) for typ, name in fields)
    extras = ('in_pStuckFaceId[i]=-1;' if cold else '')
    if any(name == 'pThermalEnthalpyResidual' for _, name in fields):
        extras += 'in_pThermalEnthalpyResidual[i]=0x1p-30*(i+1);'
    saved_velocity = r"""
        double sx[3]={-0.0,0x1.0000000000001p0,19},sy[3]={2,-3,29},sz[3]={4,-5,39};
        auto savedSnapshot=[&](){
            double x[2],y[2],z[2];
            require(PREFIXDownloadParticleSavedVelocity(handle,2,x,y,z)==0,"saved velocity snapshot");sync();
            std::vector<unsigned char> bytes;append(bytes,x,2);append(bytes,y,2);append(bytes,z,2);return bytes;
        };
        require(PREFIXUploadParticleSavedVelocity(handle,2,sx,sy,sz)==0,"valid saved velocity seed");sync();
        const auto savedOriginal=savedSnapshot(),physicalOriginal=snapshot();
        for(bool sending:{true,false})for(int mode=0;mode<5;++mode){
            const int n=mode==0?-1:mode==1?3:2;
            double x[3]={11,12,13},y[3]={21,22,23},z[3]={31,32,33};
            std::vector<unsigned char> hostBefore;append(hostBefore,x,3);append(hostBefore,y,3);append(hostBefore,z,3);
            double* px=mode==2?nullptr:x;double* py=mode==3?nullptr:y;double* pz=mode==4?nullptr:z;
            const int rc=sending?PREFIXUploadParticleSavedVelocity(handle,n,px,py,pz):PREFIXDownloadParticleSavedVelocity(handle,n,px,py,pz);
            require(rc!=0,"unavailable saved velocity resource accepted");sync();
            std::vector<unsigned char> hostAfter;append(hostAfter,x,3);append(hostAfter,y,3);append(hostAfter,z,3);
            require(hostAfter==hostBefore,"saved velocity rejection modified host fields");
            require(savedSnapshot()==savedOriginal,"saved velocity rejection modified device fields");
            require(snapshot()==physicalOriginal,"saved velocity rejection modified restart fields");
            const char* labels[]={"negative-count","over-capacity","null-x","null-y","null-z"};
            std::cout<<"PASS saved-velocity case="<<(sending?"upload-":"download-")<<labels[mode]<<" unchanged=1\n";
        }
        require(PREFIXUploadParticleSavedVelocity(handle,0,nullptr,nullptr,nullptr)==0,"empty saved upload");
        require(PREFIXDownloadParticleSavedVelocity(handle,0,nullptr,nullptr,nullptr)==0,"empty saved download");sync();
        require(savedSnapshot()==savedOriginal,"empty saved operation modified device fields");
        sx[0]=7.5;sy[0]=-8.5;sz[0]=9.5;
        require(PREFIXUploadParticleSavedVelocity(handle,1,sx,sy,sz)==0,"valid saved prefix upload");sync();
        std::vector<unsigned char> expected;append(expected,sx,2);append(expected,sy,2);append(expected,sz,2);
        require(savedSnapshot()==expected,"saved prefix did not preserve suffix");
        require(upload(1)==0,"one live particle for saved capacity test");sync();
        const std::uint64_t special[3]={0x7ff8000000000123ULL,0x7ff0000000000000ULL,0x8000000000000000ULL};
        std::memcpy(sx,&special[0],8);std::memcpy(sy,&special[1],8);std::memcpy(sz,&special[2],8);
        require(PREFIXUploadParticleSavedVelocity(handle,2,sx,sy,sz)==0,"valid capacity saved upload");sync();
        expected.clear();append(expected,sx,2);append(expected,sy,2);append(expected,sz,2);
        require(savedSnapshot()==expected,"saved nonfinite bits or capacity range changed");
        require(upload(2)==0,"restart restored after saved velocity controls");sync();snapshot();PHYSICAL_CHECKS
        std::cout<<"PASS saved-velocity prefix=1 empty=1 capacity=1 nonfinite=1 retry=1 actualCuda=1\n";
""" if app in ("CHT", "FSH") else ""
    code = r"""
#include <cuda_runtime_api.h>
#include "API"
#include <cstring>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
void require(bool ok,const char* label){if(!ok)throw std::runtime_error(label);}
void sync(){require(cudaDeviceSynchronize()==cudaSuccess,"CUDA synchronize");}
template<class T>void append(std::vector<unsigned char>&b,const T*p,size_t n){auto*q=reinterpret_cast<const unsigned char*>(p);b.insert(b.end(),q,q+n*sizeof(T));}
int main(){
    void*handle=nullptr;
    try{
        require(PREFIXCreate(CREATE_ARGS)==0&&handle,"native create");sync();
        MESH_DATA
        for(int f=0;f<6;++f){
            mesh_faceOwner[f]=0;mesh_faceNeighbour[f]=-1;mesh_facePeriodicPair[f]=-1;
            mesh_V[f]=1;mesh_Cx[f]=mesh_Cy[f]=mesh_Cz[f]=.5;
            mesh_faceCx[f]=mesh_faceCy[f]=mesh_faceCz[f]=.5;
            mesh_magSf[f]=1;mesh_deltaCoeffs[f]=2;mesh_faceWeight[f]=1;
            mesh_cellLength[f]=1;mesh_cellFaceId[f]=f;mesh_cellFaceNeighbor[f]=-1;
            mesh_cellFaceKind[f]=2;mesh_cellFaceRestitution[f]=1;mesh_cellFaceTangential[f]=1;
        }
        for(int axis=0;axis<3;++axis){
            const int low=2*axis,high=low+1;
            auto&normal=axis==0?mesh_Sfx:(axis==1?mesh_Sfy:mesh_Sfz);
            auto&center=axis==0?mesh_faceCx:(axis==1?mesh_faceCy:mesh_faceCz);
            auto&plane=axis==0?mesh_planeNx:(axis==1?mesh_planeNy:mesh_planeNz);
            normal[low]=plane[low]=-1;normal[high]=plane[high]=1;
            center[low]=0;center[high]=1;mesh_planeD[high]=1;
        }
        mesh_cellPlaneStart[0]=0;mesh_cellPlaneCount[0]=6;
        require(PREFIXUploadMesh(handle,MESH_ARGS)==0,"native mesh upload");
        const double rho=1,zero=0,temperature=300,pressure=287*temperature;
        const double energy=pressure/(1.4-1);
        require(PREFIXUploadFields(handle,&rho,&zero,&zero,&zero,&energy,&zero,&zero,&zero,&pressure,&temperature)==0,"native gas upload");sync();
        DECLARATIONS
        for(int i=0;i<3;++i){
            in_px[i]=.25+.25*i;in_py[i]=in_pz[i]=.5;
            in_pux[i]=.1*(i+1);in_puy[i]=-.05*(i+1);in_puz[i]=.025*(i+1);
            in_pT[i]=300+i;in_pTheta[i]=.25;in_pd[i]=.01;in_pm[i]=1e-6*(i+1);
            in_pCellId[i]=0;in_pStatus[i]=1;in_pRng[i]=12345+i;in_pOrigId[i]=67890+i;
            EXTRAS
        }
        auto upload=[&](int n,const std::string&missing=std::string()){
            return PREFIXUploadParticleRestartMirror(handle,n,UPLOAD_ARGS);
        };
        auto snapshot=[&](){
            int count=-1;
            require(PREFIXDownloadParticleRestartMirror(handle,&count,2,DOWNLOAD_ARGS)==0,"native download");sync();
            require(count==2,"resident particle count");
            std::vector<unsigned char>bytes;append(bytes,&count,1);SNAPSHOT
            return bytes;
        };
        require(upload(2)==0,"initial valid restart upload");sync();
        const auto original=snapshot();PHYSICAL_CHECKS
        for(int n:{-1,3}){
            require(upload(n)!=0,"unavailable particle count accepted");sync();
            require(snapshot()==original,"count rejection modified resident fields");
            std::cout<<"PASS native-restart case="<<(n<0?"negative-count":"over-capacity")<<" unchanged=1\n";
        }
        for(const char*missing:{REQUIRED_NAMES}){
            require(upload(2,missing)!=0,"required null input accepted");sync();
            require(snapshot()==original,"null rejection modified resident fields");
            std::cout<<"PASS native-restart case=null-"<<missing<<" unchanged=1\n";
        }
        in_pT[1]=333+0x1p-30;in_px[0]=.125;
        require(upload(2)==0,"valid upload after rejected input");sync();snapshot();PHYSICAL_CHECKS
        require(upload(0)==0,"empty restart upload");sync();
        int count=-1;
        require(PREFIXDownloadParticleRestartMirror(handle,&count,0,DOWNLOAD_ARGS)==0&&count==0,"empty resident count");
        require(upload(2)==0,"valid upload after clear");sync();snapshot();PHYSICAL_CHECKS
        SAVED_VELOCITY
        PREFIXRelease(handle);handle=nullptr;sync();
        std::cout<<"PASS native-restart app=APP fields=FIELD_COUNT retry=1 coldOptionalNull=1 actualCuda=1\n";
        return 0;
    }catch(const std::exception&e){
        std::cerr<<"FAIL native-restart app=APP "<<e.what()<<"\n";
        if(handle)PREFIXRelease(handle);
        return 1;
    }
}
"""
    replacements = dict(API=str(api.resolve()), PREFIX=PREFIX, APP=app,
        SAVED_VELOCITY=saved_velocity.replace('PHYSICAL_CHECKS',physical_checks), CREATE_ARGS=','.join(CREATE[name] for _, name in create),
        MESH_DATA=mesh_data, MESH_ARGS=mesh_args, DECLARATIONS=''.join(declarations),
        UPLOAD_ARGS=','.join(uploads), DOWNLOAD_ARGS=download_args,
        EXTRAS=extras, SNAPSHOT=snapshot, PHYSICAL_CHECKS=physical_checks,
        REQUIRED_NAMES=','.join('"'+name+'"' for _, name in active_fields),
        FIELD_COUNT=str(len(fields)))
    for key in sorted(replacements, key=len, reverse=True):
        code = code.replace(key, replacements[key])
    return code, fields


def validate_run_output(output, app, fields):
    required = [name for _, name in fields if not name.startswith('pCold')]
    expected = ['negative-count', 'over-capacity'] + ['null-' + name for name in required]
    rows = re.findall(r'^PASS native-restart case=(\S+) unchanged=1$', output, re.M)
    if sorted(rows) != sorted(expected):
        raise ValueError('incomplete actual native restart resource matrix')
    final = (f'PASS native-restart app={app} fields={len(fields)} '
             'retry=1 coldOptionalNull=1 actualCuda=1')
    if output.splitlines().count(final) != 1:
        raise ValueError('native resource driver did not complete valid retry and cleanup')
    summary = dict(app=app, invalid_inputs=len(expected), fields=len(fields),
                   valid_roundtrips=3, optional_cold_null=True)
    if app in ('CHT', 'FSH'):
        expected_saved = [direction+'-'+label for direction in ('upload','download')
                          for label in ('negative-count','over-capacity','null-x','null-y','null-z')]
        saved_rows = re.findall(r'^PASS saved-velocity case=(\S+) unchanged=1$', output, re.M)
        if sorted(saved_rows) != sorted(expected_saved):
            raise ValueError('incomplete actual saved velocity resource matrix')
        if output.splitlines().count('PASS saved-velocity prefix=1 empty=1 capacity=1 nonfinite=1 retry=1 actualCuda=1') != 1:
            raise ValueError('saved velocity controls did not complete')
        summary.update(saved_velocity_invalid_inputs=len(expected_saved), saved_velocity_valid_controls=5)
    return summary


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--app', required=True, choices=['CHT', 'FSH', 'gasUGKP'])
    parser.add_argument('--native-library', action='append', type=Path, default=[])
    parser.add_argument('--native-provenance', type=Path)
    parser.add_argument('--generate-only', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--memcheck', action='store_true')
    parser.add_argument('--compiler', default='g++')
    parser.add_argument('--cuda-root', type=Path, default=Path('/usr/local/cuda'))
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    code, fields = generate(root, args.app)
    source = output / 'native_restart.cpp'
    source.write_text(code)
    provenance = dict(app=args.app, driver_sha256=sha256(source),
                      api_sha256=sha256(root/'applications'/args.app/'gpu/GpuBackendApi.H'),
                      fields=fields, linked_native_files={str(p.resolve()):sha256(p) for p in args.native_library},
                      native_source_identity_inferred=False)
    if args.native_provenance:
        provenance['native_build_provenance'] = json.loads(args.native_provenance.read_text())
    (output/'source-and-link.json').write_text(json.dumps(provenance, indent=2)+'\n')
    if args.generate_only:
        print(source)
        return
    if not args.native_library:
        parser.error('--native-library must name existing final native objects or archives')
    binary = output/'native_restart'
    command = [args.compiler, '-std=c++17', '-O2', '-I'+str(args.cuda_root/'include'),
               str(source), *(str(p.resolve()) for p in args.native_library),
               '-L'+str(args.cuda_root/'lib64'), '-Wl,-rpath,'+str(args.cuda_root/'lib64'),
               '-lcudart', '-o', str(binary)]
    (output/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    with (output/'link.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(result.returncode)
    results = []
    for label, command in [('gpu', [str(binary)])] if args.run else []:
        results.append((label, command))
    if args.memcheck:
        results.append(('memcheck', [str(args.cuda_root/'bin/compute-sanitizer'), '--tool', 'memcheck',
                                    '--error-exitcode', '99', str(binary)]))
    for label, command in results:
        start = time.monotonic()
        with (output/(label+'.log')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        record = dict(command=command, exit_code=result.returncode, seconds=time.monotonic()-start,
                      binary_sha256=sha256(binary), log_sha256=sha256(output/(label+'.log')))
        (output/(label+'-result.json')).write_text(json.dumps(record, indent=2)+'\n')
        if result.returncode:
            raise SystemExit(result.returncode)
        summary = validate_run_output((output/(label+'.log')).read_text(), args.app, fields)
        (output/(label+'-summary.json')).write_text(json.dumps(summary, indent=2)+'\n')
    print(binary)


if __name__ == '__main__':
    main()
