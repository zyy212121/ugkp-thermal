import re,json
from r2_runner import *
for app,sub in [('CHT','gpu'),('FSH','private_backend')]:
 native=(ROOT/f'applications/{app}/{sub}/GpuResidentStrict.cu').read_text();start=native.index('struct DeviceState');end=native.index('\n};',start)+4
 prefix=native[:end]
 # Close the actual anonymous namespace; all canonical fields are exactly the source definition.
 p=D/'inputs'/('actual-'+app+'-state-prefix.cuh');p.write_text(prefix+'\n}\n')
 print(app,'prefixbytes',len(prefix),'state fields',native[start:end].count('sst'),'NAMESPACE?',prefix[-160:])
