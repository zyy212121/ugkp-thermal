# Final CHMT native compile receipt

Production freeze confirmed by parent at 2026-10-14 17:25:20 UTC. Ns2 and Ns10 Backend.cu compiled sequentially with CUDA 13 nvcc, sm_80, --Ofast-compile=max, -Xptxas=-O0, -O0, C++17. Both exit 0. Exact commands and start/end times: receipt.json. Object sizes and hashes: objects.json. All 348 selected common/CHMT source hashes are identical before and after both builds (source-stability.json).

Prior two attempts exited 137/Killed and remain in sibling first137/second137 directories. Available memory was observed separately; the cause of those exits is not established. Those attempts are not successful final evidence.

Immediately before native compilation, six tests in test_shared_backend_compile.py and test_shared_device_storage.py passed in 27.27s. Log: ../pr14_redundant_runtime_audit/ultimate-host-fixed.log. The first host run caught one removed trialBudget test reference; replaced with actual packet energy sum versus accepted energy flux times dt, keeping prior 1.5J, donor-factor and rollback checks. Its original failed log is ultimate-host.log.

Scope: real CUDA compilation only, host emulated behavior checks. No GPU execution, native GPU numerical equivalence, or production-optimization/O3 compilation claim. Earlier exact physical parity comparisons predate final common/budget cleanup and remain stage evidence, not newly rerun final evidence.
