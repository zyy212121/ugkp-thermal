from r3_runner import *
import importlib.util
old=(Path(__file__).parent/'r3_wet_audit.py').read_text()
new=old.replace("-(dec(b['supportImpulse'][j])-dec(bi['supportImpulse'][j]))", "+(dec(b['supportImpulse'][j])-dec(bi['supportImpulse'][j]))")
anchor="sf=s['surface']['solidFace'][f];gf=s['surface']['gasFace'][f];"
assert anchor in new
new=new.replace(anchor,"address=json.loads((D/'artifacts/WET-FILM-FORMAL'/f'checkpoint-{i}-address-decoded.json').read_text())['surface'];sf=address['gasFaceIds'][address['gasFace'][f]];gf=address['persistentId'][f];")
new=new.replace('wet-complete-physical-audit.json','wet-complete-physical-audit-S307.json').replace('AUDIT-WET-FULL-ME-P','AUDIT-WET-FULL-ME-P-S307')
new=new.split('# Wet full/restart actual first difference')[0]
source=D/'inputs/wet-audit-S307-real-addressing.py';source.write_text(new)
spec=importlib.util.spec_from_file_location('r3_wet_corrected_audit',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
amend('AUDIT-WET-FULL-ME-P',observations='SUPPORT_ANALYSIS_ERROR retained: faceindex/persistentpacketnamespace mixed; globalP support sign reversed. Not a productionphysicalfailure. Correctactualmapping/nativecodec andsign in AUDIT-WET-FULL-ME-P-S307, originalrawdata/threshold unchanged.')
amend('NS10-REACTING-FULL-INITCHECK-1800',observations='OriginalNs10complete reactingfullinput/controlDict end2e-5; actual700gas microsteps over2windows; finalinitcheckERRORSUMMARY0 +originalcheckerPASS. Previous observation .001 was prose error, actualoriginalphysicalend never changed.')
