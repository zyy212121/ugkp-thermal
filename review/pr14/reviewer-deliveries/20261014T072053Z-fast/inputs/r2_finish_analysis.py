import struct,re,json,decimal,csv
from r2_runner import *
with (D/'TEST_PLAN.md').open('a') as f:f.write('\nD checkpoint补充只读核验：实际baseline trace版本与同case原binary的3个checkpoint全bytes比较失败已保存。依据真实checkpoint格式逐段检查header/config/payload，量化仅编译sourceFingerprint及其checksum的变化；禁止重写元数据/原始输出伪造bitwise通过。新限定结果与原全bytes FAIL并列。\n')
rows=[]
header=ROOT/'work/clock-build/baseline/generated/CHMTBuildIdentity.H'
fingerprint=re.search(r'#define CHMT_SOURCE_FINGERPRINT "([^"]+)"',header.read_text()).group(1).encode()
def fnv(data):
 x=14695981039346656037
 for b in data:x=((x^b)*1099511628211)&((1<<64)-1)
 return x
for i in range(3):
 original=ROOT/'work/clock-cases/baseline/chmtOutput'/f'checkpoint-{i}/state.chmt';trace=ROOT/'work/clock-cases/baseline/saved-clock-output'/f'checkpoint-{i}/state.chmt';a=original.read_bytes();b=trace.read_bytes();cs,ss=struct.unpack_from('<QQ',a,48);csb,ssb=struct.unpack_from('<QQ',b,48);at=b.find(fingerprint,72,72+csb);assert at!=-1
 diffs=[j for j in range(len(a)) if a[j]!=b[j]];permitted=set(range(64,72))|set(range(at,at+len(fingerprint)));state_a=a[72+cs:];state_b=b[72+csb:]
 row={'checkpoint':i,'original_sha256':sha(original),'trace_sha256':sha(trace),'header_equal_except_checksum':a[:64]==b[:64],'config_size':cs,'state_size':ss,'same_sizes':(cs,ss)==(csb,ssb),'only_source_fingerprint_and_checksum_differ':set(diffs)<=permitted,'differing_byte_count':len(diffs),'differing_offsets':diffs,'source_fingerprint_offset':at,'original_source_fingerprint':a[at:at+len(fingerprint)].decode(),'trace_source_fingerprint':fingerprint.decode(),'matches_actual_generated_identity':b[at:at+len(fingerprint)]==fingerprint,'state_payload_bit_identical':state_a==state_b,'original_state_payload_sha256':hashlib.sha256(state_a).hexdigest(),'trace_state_payload_sha256':hashlib.sha256(state_b).hexdigest(),'original_checksum_valid':struct.unpack_from('<Q',a,64)[0]==fnv(a[72:]),'trace_checksum_valid':struct.unpack_from('<Q',b,64)[0]==fnv(b[72:])};rows.append(row)
p=D/'artifacts/clock-trace/baseline-checkpoint-identity-proof.json';p.write_text(json.dumps({'rows':rows,'limitation':'Whole checkpoint bytes remain unequal because the build legitimately embeds changed observer source identity. All serialized physical state payload bytes and all other config/header bytes are identical. Original raw-byte FAIL retained.'},indent=2)+'\n')
t=run('TRACE-BASELINE-CHECKPOINT-PAYLOAD-PROOF',['python3','-c','print("Read-only segmented checkpoint proof retained; assertions follow in driver")'],[5,14,15,27],expected='Statepayload/config arithmetic unchanged; honest buildidentity/checksum metadata exception, no repaired oldoutput bytes.',inputs=[p,header],kind='ANALYSIS',variant='diagnostic-clock-baseline',source=PATHS['baseline'])
ok=all(x['same_sizes'] and x['state_payload_bit_identical'] and x['only_source_fingerprint_and_checksum_differ'] and x['matches_actual_generated_identity'] and x['original_checksum_valid'] and x['trace_checksum_valid'] for x in rows);assertion(t,ok,'All3 actualserialized physicalpayloads bitidentical; onlyreal compilefingerprint64ASCII/checksum changed. Wholebyte FAIL remains.',artifacts=[str(p.relative_to(D))]);print(json.dumps(rows,indent=2))
