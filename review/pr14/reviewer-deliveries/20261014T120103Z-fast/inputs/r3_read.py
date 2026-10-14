from r3_runner import *
import sys
for arg in sys.argv[1:]:
 pieces=arg.rsplit(':',2);p=ROOT/pieces[0]
 lines=p.read_text().splitlines()
 if len(pieces)==3:lines=lines[int(pieces[1])-1:int(pieces[2])]
 print('\nFILE',arg,'\n'+'\n'.join(lines))
