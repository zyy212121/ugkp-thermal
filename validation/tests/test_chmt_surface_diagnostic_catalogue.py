"""New surface diagnostic sites must resolve to their actual producer call."""
import re
import pytest
from validation.tools.paths import SOURCE_ROOT

@pytest.mark.parametrize("site,function", [(3860,"findFeasibleSurfaceRoot"),(3861,"makeInterfacePrimitive"), *[(site,"surfaceTemperatureConfiguration") for site in range(3862,3870)], (3870,"evaluateSurfaceTemperature")])
def test_new_surface_diagnostic_producer_location(site,function):
    text=(SOURCE_ROOT/"common/DiagnosticsCatalogue.H").read_text()
    matches=re.findall(r'\{'+str(site)+r',"([^"]+)","([^"]+)",(\d+),"([^"]+)"\}',text)
    assert len(matches)==1, f"site {site} must have one descriptor"
    path, producer, line, message=matches[0]
    assert path=="applications/CHMT/ablation/InterfaceMath.H" and producer==function
    lines=(SOURCE_ROOT/path).read_text().splitlines()
    line=int(line);assert 0<line<=len(lines)
    assert re.search(r"\b"+str(site)+r"\b",lines[line-1]), (site,line,lines[line-1])
    local="\n".join(lines[max(0,line-4):line+1])
    assert re.search(r"(?:interfaceFailure|makeRecord)\s*\([\s\S]{0,260}\b"+str(site)+r"\b",local),local
    assert message
