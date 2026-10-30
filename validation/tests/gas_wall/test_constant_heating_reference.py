"""Independent Decimal Green-function reference, including signed large flux."""
from decimal import Decimal as D, localcontext
from pathlib import Path
import subprocess
from validation.tools.paths import SOURCE_ROOT

def test_constant_heating_closed_form(tmp_path):
    exe=tmp_path/'heating'
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(SOURCE_ROOT/'common'),str(SOURCE_ROOT/'validation/pr33/constant_heating.cpp'),'-o',str(exe)],check=True)
    result=subprocess.check_output([str(exe)],text=True)
    with localcontext() as ctx:
        ctx.prec=100
        def f(x):return (x.exp()-1)/x if x else D(1)
        for line in result.splitlines():
            aa,bb,ok,hh=line.split();a,b,h=D(aa),D(bb),D(hh)
            ref=D('.5') if a==0 else (a/(a.exp()-1))**2*(f(2*a-b)-(-b).exp()*f(2*a))/(1-(-b).exp())
            assert ok=='1',line
            assert abs(h-ref)<=abs(ref)*D('2e-11')+D('5e-324'),(line,ref)
