import sys
from r2_runner import *
for token in sys.argv[1:]:
 path,begin,end=token.rsplit(':',2);lines=(ROOT/path).read_text().splitlines();print('FILE '+path)
 for i in range(int(begin)-1,min(int(end),len(lines))):print(str(i+1)+':'+lines[i])
