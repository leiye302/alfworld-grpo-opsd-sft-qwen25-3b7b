"""CPU checks of storage decisions and the preserved native launch recipe."""
import ast
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
sys.path.insert(0,str(ROOT/'scripts'))
from hardware_profile import GIB,PROFILE,storage_flags
from handoff import arguments


def main():
    cases=[
        (100,100,False,1*GIB,(False,False)),
        (100,100,True,20*GIB,(False,False)),
        (16384,100,True,20*GIB,(True,False)),
        (100,512*1024**2,True,20*GIB,(False,True)),
        (100,100,True,7*GIB,(True,True)),
        (100,100,True,8*GIB,(False,False)),
    ]
    for *inputs,expected in cases:
        assert storage_flags(*inputs)==expected,(inputs,expected)
    assert 40*GIB>=PROFILE['minimum_gpu_memory_bytes']>32*GIB
    work=Path('/data/alfwork')
    for size in ('3b','7b'):
        for method in ('sft','baseline'):
            run=work/'runs'/(size+'_'+method)
            expected=json.loads((ROOT/'configs'/(size+'_'+method+'.json')).read_text())
            expected=[x.replace('/data/alfrepo',ROOT.as_posix()) for x in expected]
            actual=arguments(work,run,size,method)
            # Linux path spelling can also be checked on the Windows maintainer.
            actual=[x.replace('\\','/') for x in actual]
            assert actual==expected,[(a,b) for a,b in zip(actual,expected) if a!=b]
    for relative in ('scripts/handoff.py','scripts/preflight.py','runtime/hardware_profile.py',
                     'runtime/response_head_memory.py','runtime/sdar_observe.py'):
        ast.parse((ROOT/relative).read_text(encoding='utf-8'),filename=relative)
    print(json.dumps({'passed':True,'storage_cases':len(cases),'native_configurations_unchanged':4,
                      'hardware_profile':PROFILE,'GPU_FSDP_tested':False},indent=2))


if __name__=='__main__':main()
