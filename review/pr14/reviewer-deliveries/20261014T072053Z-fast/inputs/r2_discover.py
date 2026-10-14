import subprocess,pathlib,json,hashlib
R=pathlib.Path('/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261013T182722Z-fast')
H='75cfd885c028fe80d011a8a081f8bed1f8aeb55e'
def git(*a): return subprocess.check_output(['git','-C',str(R),*a],text=True)
files=git('ls-tree','-r','--name-only',H).splitlines()
print('AGENTS', [x for x in files if x.endswith('AGENTS.md')])
for p in R.parents:
 if (p/'AGENTS.md').is_file():print(str(p/'AGENTS.md'),(p/'AGENTS.md').read_text())
prefix='review/pr14/developer-deliveries/20261014T055427Z-fast/'
for x in ['REPORT.md','SOURCE.json','VALIDATION.json','CLOCK_ANALYSIS.txt','clock-trace/RUN.txt']:
 print('\nFILE '+prefix+x+'\n'+git('show',H+':'+prefix+x))
print('\nPROTOCOL\n'+git('show',H+':review/pr14/PROTOCOL.md'))
print('\nCHANGES\n'+git('diff','--stat','802a11e7',H,'--',':!review',':!README.md'))
print('\nTEST FILES',*[x for x in files if x.startswith('tests/')],sep='\n')
