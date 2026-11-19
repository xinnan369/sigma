"""R4 closure: rerun R3 independent checks with newly measured CPU results."""
import argparse
import json
from pathlib import Path
import shutil

from . import independent_audit as r3

FIX='04c12f1fb4ca9fb403a0a3ec72e3ab0fb35a8d8f'
R3='b532d580a5dc431bc4a4048833d94db433dddd4f'
TESTED='aa6f081126fc0da517b5e4a7dc111c5cdbe0a62c'
R3_EVIDENCE='reports/evidence/r3-d1b-r3/'
FIX_PATHS={'experiments/r2k_q1v2_llvip/history.py','experiments/r2k_q1v2_llvip/scan.py',
           'tests/test_r2k_q1v2_llvip.py'}
CARRY=('numerics.json','numerics.log','resume.json','resume.log','data-integrity.json',
       'data.log','gt-coverage.json','gt-per-image.jsonl')


def close(previous,rerun):
    for path in (previous,rerun):
        r3.must(not path.resolve().is_relative_to(r3.ROOT),'Audit runs must stay outside repo')
    r3.must(r3.git('merge-base',TESTED,'HEAD').decode().strip()==TESTED,'Wrong audit ancestry')
    r3.must(r3.git('show','-s','--format=%P',TESTED).decode().split()==[R3,FIX],
            'Unexpected integration parents')
    for left,right in ((r3.HEAD,FIX),(R3,TESTED)):
        files=set(r3.git('diff','--name-only',left,right).decode().splitlines())
        r3.must(files==FIX_PATHS,'Fix changed files outside the compatibility patch')
    r3.must(not r3.git('diff',TESTED,'--','tests','experiments/r3_d1',
                       'experiments/r2k_q1v2_llvip','docs/R3_D1_DETECTION_PROTOCOL.md'),
            'Tested code no longer matches integrated tree')
    r3.must(json.loads((rerun/'tested-head.json').read_bytes())=={'head':TESTED},
            'CPU evidence does not identify integrated tree')
    for log in ('cpu.log','compileall.log'):
        r3.must(r3.exit_code(rerun/log)==0,f'CPU gate failed: {log}')
    archived=lambda p:r3.git('show',f'{R3}:{R3_EVIDENCE}{p}')
    old=json.loads(archived('independent.json'))
    verification=json.loads(archived('verification.json'))
    r3.must(old['status']=='HOLD' and old['exit_codes']['cpu']==1 and not old['authorized'],
            'Expected an intact historical HOLD')
    for f,h in verification['files_sha256'].items():
        runtime_path=r3.ROOT/R3_EVIDENCE/f
        raw=runtime_path.read_bytes()
        r3.must(r3.sha(archived(f))==h and (raw==archived(f) or
                raw.replace(b'\r\n',b'\n')==archived(f)),f'Historical R3 artifact drift: {f}')
    target=rerun/'recombined'
    target.mkdir()  # Never overwrite either run or an earlier closure.
    for name in CARRY:
        obj=(previous/name).read_bytes()
        r3.must(r3.sha(obj)==old['fresh_file_sha256'][name],f'Original rerun artifact differs: {name}')
        (target/name).write_bytes(obj)
    for name in ('cpu.log','compileall.log'):
        shutil.copyfile(rerun/name,target/name)
    result=r3.audit(target)
    r3.must(result['status']=='PASS' and not result['authorized'],'Independent gate did not close')
    result.update(tested_integration_head=TESTED,compatibility_commit=FIX,
                  historical_r3_head=R3,historical_r3_status=old['status'],
                  historical_r3_independent_sha256=r3.sha(archived('independent.json')),
                  retained_runtime_artifacts=list(CARRY),
                  fresh_cpu_metadata_sha256={n:r3.sha((rerun/n).read_bytes())
                                             for n in ('tested-head.json','environment.log')})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--rerun',type=Path,required=True)
    args=p.parse_args()
    result=close(args.previous,args.rerun)
    with (args.rerun/'closure.json').open('x',encoding='utf-8',newline='\n') as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write('\n')
    print('PASS: all technical gates; no authorization issued')
