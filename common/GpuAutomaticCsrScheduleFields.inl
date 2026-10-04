// Included in each application's DeviceState. These roles never alias:
// occupancy statistics, published task counts, and worker queue position.
int csrHeavyReductionMode = 0;
int csrHeavyReductionEnabled = 0;
int csrHeavyReductionActive = 0;
int csrHeavyAutoInterval = 100;
unsigned long long schedulingAdvanceCount = 0;
int* csrMaximumOccupancy = nullptr;
int* csrHeavyTaskCount = nullptr;
int* csrHeavyCellCount = nullptr;
int* csrHeavyTaskCursor = nullptr;
// Host-side publication readiness, invalidated by every directory producer.
// Device consumers still use the existing counts and descriptors.
int csrTasksReady = 0;
int csrPreparedDirectoryKind = -1;
