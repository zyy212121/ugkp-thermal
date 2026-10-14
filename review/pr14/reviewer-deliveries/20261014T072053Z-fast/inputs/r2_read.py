import sys,json
from r2_runner import *
for path in sys.argv[1:]:
 p=ROOT/path
 print('\nFILE '+path+'\n'+p.read_text())
