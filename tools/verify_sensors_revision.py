"""Verify addendum completeness separately from scientific submission readiness."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'src'))
import sensors_train as S
from sensors_evaluate import jobs


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--draft',action='store_true')
    args=ap.parse_args()
    checks=[]
    def check(label,condition):
        checks.append(dict(check=label,passed=bool(condition)))
    for name in ['models.py','pkg_train.py','pkg_eval.py']:
        check('original source preserved: '+name,(ROOT/'src'/name).read_bytes()==(ROOT/'logs/sensors_v4/backup'/name).read_bytes())
    check('original manuscript preserved',(ROOT/'paper/manuscript_zh_evidence.md').read_bytes()==(ROOT/'logs/sensors_v4/backup/manuscript_zh_evidence.md').read_bytes())
    protocol=json.loads(S.PROTOCOL.read_text())
    specs=S.expanded_runs(protocol)
    check('12 primary runs',sum(s['phase']=='main' for s in specs)==12)
    check('6 budget runs',sum(s['phase']=='budget' for s in specs)==6)
    for lang in ['en','zh']:
        path=ROOT/'paper/sensors_submission'/f'manuscript_{lang}.md'
        text=path.read_text()
        check(lang+' balanced displayed equations',text.count('$$')%2==0)
        check(lang+' rendered template',not re.search(r'\{\{[A-Z_]+\}\}',text))
        if not args.draft:
            check(lang+' no missing result placeholders','MEASURED RESULT PENDING' not in text and '待实测结果' not in text)
        for image in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text):
            check('image exists: '+image,(path.parent/image).exists())
    if not args.draft:
        for spec in specs:
            tag=spec['tag'];path=S.OUT/f'{tag}.json'
            check('training complete: '+tag,path.exists())
            if path.exists():
                meta=json.loads(path.read_text())
                check('training specification: '+tag,meta['spec']==spec and meta['protocol_sha256']==S.digest(S.PROTOCOL))
                check('checkpoint intact: '+tag,S.digest(S.OUT/f'{tag}.pt')==meta['checkpoint_sha256'])
        for tag,datasets in jobs():
            p=ROOT/'results/sensors_v4/evaluation'/f'{tag}.json'
            check('evaluation exists: '+tag,p.exists())
            if p.exists():
                a=json.loads(p.read_text())
                expected=158 if datasets=={'ronin','ridi'} else 130
                check('evaluation complete: '+tag,a['complete'] and len(a['rows'])==expected)
                check('no duplicated sequences: '+tag,len({(r['dataset'],r['split'],r['seq']) for r in a['rows']})==expected)
        p=ROOT/'results/sensors_v4/analysis.json'
        check('analysis exists',p.exists())
        if p.exists():
            a=json.loads(p.read_text())
            check('four primary comparisons',sum(c['primary'] for c in a['contrasts'])==4)
            check('all primary participant counts',all(c['n_subjects']==(15 if c['dataset']=='ronin' else 11) for c in a['contrasts'] if c['primary']))
    failed=[r['check'] for r in checks if not r['passed']]
    report=dict(mode='draft' if args.draft else 'completed_batch',checks=checks,failed=failed,
                submission_ready=False,reason='Numerical/file validation does not replace author declarations, scientific interpretation or final human approval.')
    out=ROOT/'verification/generated/sensors_v4'/('draft_audit.json' if args.draft else 'batch_audit.json')
    S.atomic_json(out,report)
    print(f'{len(checks)-len(failed)}/{len(checks)} checks passed')
    if failed:raise SystemExit('\n'.join(failed))


if __name__=='__main__':main()
