# Budget API test migration

Scope: only applications/CHMT/tests/test_transaction.cpp and tests/test_chmt_support_accounting_removal.py. No production edits.

Before migration, old support tests passed (2 passed in 33.62s). New transaction API test failed compilation against old Budget-reference signature (new-api-red.log). With new production API, g++ -std=c++17 -O1 -ffp-contract=off transaction compiled and passed (transaction-green.log). Migrated pytest tests: 2 passed in 33.36s (final-pytest.log). Owned diff check clean.

Physical oracles use actual gas mass/species/energy/momentum and solid donor mass/energy, explicit film enthalpy minus pV, phase transfer before/after energy, and actual consumer/storage markers. Duplicate packet/storage and donor overdraw atomicity remain covered. Resource transaction failure/publish coverage retained.

Legacy NaN/Inf Budget payload no longer enters packet API. Test compares actual clean/legacy HostState transaction inventories; retained legacy payload assertions merely verify compatibility data is untouched. Old exchangeBudget equality and consumedPackets statistics assertions were retired, not relabeled physical conservation. Phase/film pure reporting fields are excluded from BASE/current FP64 parity output; actual inventories, profiles, phase packet values and geometry remain compared byte-for-byte as hex floating text. Baseline still demonstrates old unnecessary support momentum computation in red, current demonstrates removal in green.

No production audit helper was restored. No claim of full solid/reduced-film momentum closure. These are host C++ tests; no native GPU or OpenFOAM runtime claim.
