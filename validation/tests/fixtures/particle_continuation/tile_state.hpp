
#define GPU_DIRECTORY_OWNS_TILE_POLICY 0
#define GPU_DIRECTORY_HAS_BASE_ONLY HAS_BASE
#define GPU_DIRECTORY_SELECTOR kind
#define GPU_DIRECTORY_FULL 0
enum class HeavyDirectoryKind {full=0,splitBaseAndInjection=1,baseOnly=2};
enum class CsrReductionTaskSource {fullIndexed=0,splitBaseDirect=1,splitInjectionIndexed=2,splitLogical=3};
struct CsrReductionTask {int cell,begin,end,source;};
struct DeviceState {
 int nCells=0,reductionBlockThreads=256,multiprocessorCount=1,lightResidentBlocksPerSm=1;
 int dynamicHeavyThreshold=0,csrHeavyCellThreshold=0,csrHeavyTileParticles=0,csrHeavyTaskCapacity=0,particleCapacity=INT_MAX;
 int *preBaseCellOffset=nullptr,*cellParticleOffset=nullptr,*csrCellTaskCount=nullptr,*csrCellTaskOffset=nullptr;
 int *csrHeavyCellCount=nullptr,*csrMultiTaskCellList=nullptr,*csrHeavyTaskCount=nullptr,*preBaseParticleCountDevice=nullptr;
 CsrReductionTask* csrReductionTasks=nullptr;
};
