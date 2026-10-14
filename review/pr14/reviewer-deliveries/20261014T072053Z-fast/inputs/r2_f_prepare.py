import re,shutil,difflib,hashlib
from r2_runner import *
with (D/'TEST_PLAN.md').open('a') as f:f.write('\nF物理范围新增实例（此前NOT_RUN的原gap IDs）：保持原thermal的8+8cells、原终点0.001/2macro×50micro和原阈值，新增wet-film（rho1000/cp1000/初温350/厚1e-5）、wet-evaporation（HertzKnudsen单species accommodation1e-4/saturation200000，CoupledRecession）、dry-ablation（表面C0→S0，A1e-5/order1，CoupledRecession），分别实际Native GPU气体+原CPU材料/膜owner，不运行CPUshim。用S207实际原checkpointcodec扩展输出film/particle/完整packet及speciesbudget以独立80位库存核算，未改codec和阈值。\nCHT enabledradiation/mixedcoupler原缺口：以既有cold2D输入的真实blockMeshDict重新生成原.04×.04×.01十cell流体mesh，不继承历史thinmesh；配置匹配下方solidregion十cell/.001m、真实oneToOneConformal impactWall↔top，其他fluidfaces uncoupled，实际solid rho1000/Cp800/k10。一个disabledradiation对照、一个enabled Mie DO实例（真实table backing，particleCount/源消费和场写出检查）。gas/particle原物性、原终点1e-5和原cold阈值不变。该新增物理mesh不是原F004输入修复或原结果替代。所有nativecases外部300s。empty/多selectedface mapper未据singleface推定覆盖。\n')
for name in ['wet-film','wet-evaporation','dry-ablation']:
 case=ROOT/'work/cases/extra'/name;shutil.copytree(OLD/'inputs/native/chmt/thermal-full',case)
 for rel in ['constant/polyMesh','constant/solid/polyMesh']:shutil.copytree(ROOT/'work/cases/chmt/thermal-full'/rel,case/rel)
 q=case/'constant/chmtProperties';s=q.read_text()
 if name!='dry-ablation':
  s=s.replace('surface { patch right; gasPatch left; }','surface { patch right; gasPatch left; Y (1 0); thickness 1e-5; pressure 171600; }')
  s+='\nenableFilm true; filmThermalMode ThicknessAveraged; filmInitial { kind UniformFilmTemperature; mean 350; }\nliquid { rho 1000; cp0 1000; cp1 0; e0 100000; conductivity 1; Tmin 100; Tmax 3000; element (); }\nliquidViscosity .01; liquidReferencePressure 171600;\n'
  if name=='wet-evaporation':s+='\nmaterial { evaporation HertzKnudsen; evaporationAccommodation (1e-4 0); saturationPressureScale (200000 0); saturationTemperature (0 0); }\nmeshMotion { policy CoupledRecession; }\n'
 else:s+='\nmaterial { enableSurfaceReactions true; surfaceReactions { pyrolysis { A 1e-5; temperaturePower 0; activationEnergy 0; condensedNu (-1 0); gasNu (1 0); order (1 0); gasOrder (0 0); } } }\nmeshMotion { policy CoupledRecession; }\n'
 q.write_text(s);shutil.copytree(case,D/'inputs/native/extra'/name,ignore=shutil.ignore_patterns('polyMesh'))
old=D/'inputs/checkpoint_actual_reader.cpp';s=old.read_text();anchor=' std::printf(",\\\"chemistryAudit\\\":[");'
# Match the real existing C++ source text, preserving its complete codec call.
anchor=' std::printf(",\\\"chemistryAudit\\\":[");'
assert anchor in s
insert=r''' std::printf(",\"film\":[");first=true;for(std::size_t i=0;i<s.film.size();++i){if(!first)std::printf(",");first=false;const auto&q=s.film[i];const auto&a=s.filmAux[i];std::printf("{\"mass\":");num(q.mass);std::printf(",\"enthalpy\":");num(q.enthalpy);std::printf(",\"species\":");arr(q.species);std::printf(",\"pressure\":");num(a.pressure);std::printf(",\"kineticEnergy\":");num(a.kineticEnergy);std::printf(",\"area\":");num(a.area);std::printf(",\"thickness\":");num(a.thickness);std::printf(",\"solidFront\":");num(a.solidFront);std::printf(",\"gasFront\":");num(a.gasFront);std::printf("}");}std::printf("]");
 std::printf(",\"ledger\":[");first=true;for(const auto&q:s.ledger){if(!first)std::printf(",");first=false;std::printf("{\"kind\":%d,\"stage\":%d,\"consumerMask\":%u,\"step\":%llu,\"face\":%llu,\"mass\":",int(q.kind),q.stage,q.consumerMask,static_cast<unsigned long long>(q.step),static_cast<unsigned long long>(q.face));num(q.mass);std::printf(",\"energy\":");num(q.energy);std::printf(",\"momentum\":");vec(q.momentum);std::printf(",\"species\":");arr(q.species);std::printf(",\"condensed\":");arr(q.condensed);std::printf(",\"pore\":");arr(q.pore);std::printf(",\"conductive\":");num(q.conductive);std::printf(",\"advective\":");num(q.advective);std::printf(",\"pressureWork\":");num(q.pressureWork);std::printf(",\"viscousWork\":");num(q.viscousWork);std::printf(",\"radiation\":");num(q.radiation);std::printf(",\"liquidKineticAdvection\":");num(q.liquidKineticAdvection);std::printf("}");}std::printf("]");
'''
s=s.replace(anchor,insert+anchor);anchor=' std::printf("\\\"exchangeMass\\\":");';assert anchor in s
s=s.replace(anchor,' std::printf("\\\"exchangeSpecies\\\":[");for(int i=0;i<NParticipants;++i){if(i)std::printf(",");arr(s.budget.exchangeSpecies[i]);}std::printf("],\\\"boundarySpecies\\\":");arr(s.budget.boundarySpecies);std::printf(",");\n'+anchor)
new=D/'inputs/checkpoint_actual_reader_S207.cpp';new.write_text(s);(D/'patches/S207-real-codec-observed-fields.patch').write_text(''.join(difflib.unified_diff(old.read_text().splitlines(True),s.splitlines(True),fromfile='a/checkpoint_actual_reader.cpp',tofile='b/checkpoint_actual_reader_S207.cpp')))
print('prepared CHMT input cases and S207 reader',flush=True)
for name,rad in [('CHT-actual-coupler',False),('CHT-actual-radiation',True)]:
 case=ROOT/'work/cases/resident'/name;shutil.copytree(OLD/'inputs/native/resident/CHT-coldWall2D-INPUT2',case)
 q=case/'constant/particleProperties';q.write_text(q.read_text()+'\ngpuResidentSolidThermalCoupling true;\n')
 props='rho {type constant; value 1000;} Cp {type constant; value 800;} kappa {type constant; value 10;} emissivity {type constant; value .8;}'
 (case/'constant/solidRegionProperties').write_text('FoamFile {version 2.0;format ascii;class dictionary;object solidRegionProperties;}\nsolidThermalCoupling { formatVersion 1; solidRegion solid; mapping oneToOneConformal; coupledInterfaces ({fluidPatch impactWall;solidPatch top;}); properties {'+props+'} couplingInterval 1e-6; nonlinearMaxIterations 200; nonlinearRelativeTolerance 1e-10; energyRelativeTolerance 1e-8; energyAbsoluteTolerance 1e-8; }\n')
 model='mieTable "constant/mieTable"; do { IBoundaryCondition diffuseGrey; nTheta 2; nPhi 4; maxSourceIterations 200; sourceRelativeTolerance 1e-12; phaseBalanceTolerance 1e-12; maxPhaseBalanceIterations 200; radiationActiveEpsilonMin 0; conservationRelativeTolerance 1e-10; maxPhaseScalingCorrection 100; }'
 (case/'constant/radiationProperties').write_text('FoamFile {version 2.0;format ascii;class dictionary;object radiationProperties;}\nenabled '+str(rad).lower()+'; couplingFrequency 1; model {'+(model if rad else '')+'}\n')
 if rad:
  payload='Tvals 300 4000\nDvals .001 .01\nMuvals -1 1\nPlanckBandFractions 1 1\nDATA\n'+''.join(f'NODE {d} {t} .3 .7 1\nPHI 1 1\n' for d in range(2) for t in range(2))+'END_MIE_TABLE\n';header='MIE_TABLE\nformatVersion 1\nunits diameter_m temperature_K wavelength_m\ninterpolation logDiameter_linearTemperature_linearMu\ngeneratorSha256 fixture\ngenerationArgs {}\ndiameter .001 .01 2\ntemperature 300 4000 2\nmu -1 1 2\nwavelength 1e-6 2e-6 2\n';(case/'constant/mieTable').write_text(header+payload+'payloadSha1 '+hashlib.sha1(payload.encode()).hexdigest()+'\n')
 for base in ['0','system','constant']:(case/base/'solid').mkdir()
 for field in ['fvSchemes','fvSolution']:shutil.copy2(case/'system'/field,case/'system/solid'/field)
 q=case/'system/solid/blockMeshDict';s=(case/'system/blockMeshDict').read_text();s=s.replace('(0 0 0)','(0 0 -0.001)').replace('(0.04 0 0)','(0.04 0 -0.001)').replace('(0.04 0.04 0)','(0.04 0.04 -0.001)').replace('(0 0.04 0)','(0 0.04 -0.001)');s=s.replace('0.01','0');q.write_text(s)
 s=(case/'0/T').read_text();s=re.sub(r'internalField\s+uniform\s+[^;]+;','internalField uniform 1000;',s);s=re.sub(r'top\s*\{[^}]*\}','top { type fixedGradient; gradient uniform 0; value uniform 1000; }',s);s=re.sub(r'impactWall\s*\{[^}]*\}','impactWall { type zeroGradient; }',s);(case/'0/solid/T').write_text(s)
 inp=D/'inputs/native/resident'/name;shutil.copytree(case,inp)
 print('prepared',name,flush=True)
