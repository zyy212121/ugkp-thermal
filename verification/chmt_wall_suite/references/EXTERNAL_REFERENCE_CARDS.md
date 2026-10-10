# Independent sources, data availability, and scope

Every downloaded file identifier, URL, byte count and SHA-256 is in `sources.json`. The checkout contains metadata and adapters only, not noncommercial datasets. URLs and checksums were verified by direct downloads on 2026-10-09. `fetch_references.py` refuses changed bytes and repository-internal destinations.

## Gas transport and chemistry

- The periodic binary waveform is a closed-form solution of constant-density advection/Fick diffusion with identical molecular weights and calorics. The reactive companion multiplies its complete A solution by exp(-2t), with B=1-A. These are synthetic analytic verification cases. Exact cell averages, both species, conservation and actual native fields are checked.
- `common/chemistry/mechanisms/h2o2.cantera-reactor-reference.json` is a pinned independent Cantera 3.1.0 closed constant-volume reactor history. Its thermo/source SHA-256 and controls are retained in the existing manifest. Mechanism source: https://raw.githubusercontent.com/Cantera/cantera/v3.1.0/data/h2o2.yaml . Species: H2,H,O,O2,OH,H2O,HO2,H2O2,AR,N2. The existing native reactor generation/checker is reused; reactor agreement alone is not spatial species-transport validation.
- Optional future flame comparison: https://cantera.org/stable/examples/python/onedim/adiabatic_flame.html . This is a reproducible numerical example, not raw experiment. Default multicomponent/Soret transport cannot be compared as like-for-like with this solver's constant species D. No default flame-speed value is used as a false native target.

## NASA TMR flat plate

https://tmbwg.github.io/turbmodels/flatplate_val.html and https://tmbwg.github.io/turbmodels/flatplate_val_sst.html

Actual downloaded files include `FlatPlate_validation/sst-cf_cfl3d.dat`, `sst-upyp_cfl3d.dat`, and `cf_K-S.dat`. SSTm/CFL3D data are independent numerical results; Kármán–Schoenherr values are correlations. Neither is raw measured experimental data. The benchmark compares Cf(Re_theta) and u+(y+); the suite's editable plate geometry is a paired wall-family study until its inflow, Reynolds number, SST variant and sampling definition are matched to TMR. Do not treat an arbitrary seed grid as a y+ compliance result.

## Corrected reacting wall equations

Di Renzo & Urzay, 2019 annual brief, author PDF: https://web.stanford.edu/~jurzay/06_DiRenzo.pdf . Its first page explicitly records the 2020-12-23 correction of equations (3.1)–(3.3). This is equation provenance and a stationary-wall dissociating-air channel comparison. No machine-readable channel dataset was retrieved here. It does not validate newly added blowing, wet film, moving interface, or solid recession.

## TACOT/PATO numerical reference, not a real material experiment

Source: https://github.com/nasa/pato/tree/main/data/Materials/Composites/TACOT . Downloaded `constantProperties`, `virgin`, `char`, `gasProperties` bytes are separately hashed. The PATO tutorial definition is at https://pato-doc-94d501.gitlab.io/tutorials.html . PATO supplies numerical material-response examples; these are not native CHMT inputs.

The inspected source uses fiber/resin intrinsic densities 1600/1200 kg/m³ and initial volume fractions 0.1/0.1; bulk virgin density 280 kg/m³, porosity 0.8. Its three source-specific pyrolysis fractions are 0.25/0.19/0.06; A=12000/4.97777e8/4.97777e8 s^-1; Ea=71130.89/169975/169975 J/mol; order 3, temperature power 0, onset 333.3/555.6/555.6 K. Product mass splits are H2O/A1/A1OH=0.69/0.02/0.29; CO2/CO/CH4=0.09/0.33/0.58; H2=1. A1/A1OH require a mechanism's explicit benzene/phenol aliases, not guessed species mappings.

The raw tables define thermophysical ranges; their source says no extrapolation. PATO's source includes a pyrolysis-enthalpy correction to the spreadsheet. Other implementations use different two-lump models, so combining MIRGE-Com kinetics with PATO three-product data would create a new unsupported material. Current CHMT lacks a direct representation of full tabulated solid properties and these independent resin fractions; no native TACOT agreement is claimed.

## AblaNTIS v1.0.2: experimental and numerical data are genuinely available

Version DOI: https://zenodo.org/records/15938724 ; author repository: https://github.com/Fratorhe/AblaNTIS-test-cases/tree/v1.0.2 . Dataset license: CC BY-NC 4.0. Commercial-use permission is not supplied by this suite. Do not vendor the source archive or adapt it for commercial use without resolving licensing. The archive is 40,354,794 bytes and has publisher MD5 165d6d23d850e0ba1ab318c1fe108a46. Separate source-file SHA-256 entries are in the manifest.

The downloaded booklet and `Tacot_Zuram_Calcarb_database_v4.3.1.ods` provide full material definitions. Source folders contain numeric TC, surface temperature/recession, pressure/film and pyrolysis-property tables. `Num-*` cases are numerical comparisons; `Exp-*` are experiments. Prefer the low-heat-flux Exp-2-x-Z candidate (nominal 0.3 MW/m², air) for further compatibility work. It is not automatically a complete external-flow finite-rate benchmark.

Important publisher-listed v1.0.2 limitations: air Exp-x-T-x recession files omit an initial transient but are incorrectly aligned to t=0; some missing intervals approach 7 s. SEB film coefficients have acknowledged bias. Preserve raw time and report any justified correction separately. Prescribed-temperature/recession cases validate in-depth response, not predicted surface recession.

ZURAM includes cp=a+bT+c/T, orthotropic temperature-dependent conductivity, changing porosity/permeability, four resin Arrhenius components, formation enthalpies and equilibrium pyrolysis properties. These exceed the current CHMT condensed property schema. The suite intentionally provides an external-data importer and mapping-gap report, not an invented faithful ZURAM executable. The booklet's printed elemental mole fractions need checking against the workbook/XML because the printed values do not sum exactly to one; do not silently normalize a source inconsistency.

## MSS7 evidence boundary

The existing `examples/thermal/MSS7_laminar` geometry is available locally. Its temperature-history CSV is an unresolved Git-LFS pointer in the inspected checkout; no numeric experimental curve was recovered. Original CHT fluid uses a pure mixture, whereas the suite uses declared controlled equal-thermo species. The imported wedge coordinates are retained as a 3D slip-sided sector because CHMT has no original wedge/axisymmetric boundary equation path. These cases are geometry-adapted code comparisons, not reproduction of the original axisymmetric experiment or validation of actual graphite ablation.
