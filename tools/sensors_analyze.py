"""Subject-level effects, convergence checks and protocol sensitivity tables."""
from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'src'))
import sensors_train as S

OUT=ROOT/'results/sensors_v4'
FAMILIES={
    'GN only':'Sensors_v4_resnet_gn_s{}',
    'GN + yaw augmentation':'Sensors_v4_resnet_yaw_s{}',
    'GN + mixed-sensor PCA':'Sensors_v4_resnet_mixed_pca_s{}',
    'GN + HeadingNorm':'ResNet18_v3_hn_gn_s{}',
    'Transformer + HN + GN (no PreFilter)':'A6_v3_nopre_s{}',
    'IMUNet + HN + GN':'IMUNet2024_v3_hn_gn_s{}',
    'Historical full HN-Transformer':'A6_v2_f100_s{}',
    'Historical ResNet18':'ResNet18_v2_matched_s{}',
    'Historical IMUNet':'IMUNet2024_v21_matched_s{}'}


def subject_values(rows, template, dataset, split, metric):
    wanted={template.format(seed):seed for seed in range(4)}
    sequence=defaultdict(dict)
    for row in rows:
        if row['model'] in wanted and row['dataset']==dataset and row['split']==split:
            if metric in row:
                sequence[row['seq']][wanted[row['model']]]=float(row[metric])
    if not sequence:
        return None
    if not all(set(values)=={0,1,2,3} for values in sequence.values()):
        raise RuntimeError(f'Incomplete seeds: {template}/{dataset}/{metric}')
    grouped=defaultdict(list)
    for seq,values in sequence.items():
        grouped[seq.split('_')[0]].append([values[s] for s in range(4)])
    subjects={s:np.mean(v,axis=0) for s,v in grouped.items()}
    seq_seed=np.array([[v[s] for s in range(4)] for v in sequence.values()])
    return subjects,seq_seed,len(sequence)


def effect(a,b):
    if a.keys()!=b.keys():
        raise RuntimeError('Unpaired subject sets')
    keys=sorted(a)
    av,bv=np.array([a[k] for k in keys]),np.array([b[k] for k in keys])
    per_subject=(av-bv).mean(axis=1)
    rng=np.random.default_rng(20260906)
    bs=per_subject[rng.integers(0,len(keys),(10000,len(keys)))].mean(axis=1)
    return dict(n_subjects=len(keys),treatment_mean=float(av.mean()),control_mean=float(bv.mean()),
                difference=float(per_subject.mean()),conditional_subject_bootstrap95=np.quantile(bs,[.025,.975]).tolist(),
                improved_subjects=int((per_subject<0).sum()),paired_seed_differences=(av-bv).mean(axis=0).tolist(),
                subject_differences={s:float(d) for s,d in zip(keys,per_subject)})


def budget_analysis(protocol):
    rows=[]
    main_ref={'gn':'Sensors_v4_resnet_gn_s0','yaw':'Sensors_v4_resnet_yaw_s0',
              'mixed_pca':'Sensors_v4_resnet_mixed_pca_s0','hn':'ResNet18_v3_hn_gn_s0',
              'imunet_raw':'IMUNet2024_v21_matched_s0','transformer_full':'A6_v2_f100_s0'}
    for spec in S.expanded_runs(protocol):
        if spec['phase']!='budget': continue
        name=spec['family'][len('budget_'):]
        paths=[S.OUT/f'{main_ref[name]}.json',ROOT/'models/submission_frozen'/f'{main_ref[name]}.json']
        base=json.loads(next(p for p in paths if p.exists()).read_text())
        extended=json.loads((S.OUT/f'{spec["tag"]}.json').read_text())
        old=float(base['best_val_loss']); new=float(extended['best_val_loss'])
        rows.append(dict(family=name,reference=main_ref[name],extended=spec['tag'],seed=0,
                         val_loss_20k=old,val_loss_40k=new,relative_val_loss_reduction=(old-new)/old,
                         best_step_20k=min(base['history'],key=lambda h:h['val_loss'])['step'],
                         best_step_40k=extended['best_step']))
    front=[r for r in rows if r['family'] in {'gn','yaw','mixed_pca','hn'}]
    old_best=min(front,key=lambda r:r['val_loss_20k'])['family']
    new_best=min(front,key=lambda r:r['val_loss_40k'])['family']
    material=any(r['relative_val_loss_reduction']>.05 for r in rows) or old_best!=new_best
    return rows,material,dict(best_frontend_20k=old_best,best_frontend_40k=new_best)


def main():
    from sensors_evaluate import jobs
    for tag,_ in jobs():
        if not (OUT/'evaluation'/f'{tag}.json').exists():
            raise RuntimeError(f'Missing required evaluation: {tag}')
    protocol=json.loads(S.PROTOCOL.read_text())
    frozen=json.loads((ROOT/'results/submission_frozen/eval_rq1_rq4_v3_registered.json').read_text())
    lookup={(r['model'],r['dataset'],r['split'],r['seq']):dict(r) for r in frozen}
    sources={}
    for p in sorted((OUT/'evaluation').glob('*.json')):
        artifact=json.loads(p.read_text())
        if not artifact['complete']: raise RuntimeError(f'Incomplete evaluation {p}')
        sources[str(p.relative_to(ROOT))]=S.digest(p)
        for r in artifact['rows']:
            key=(r['model'],r['dataset'],r['split'],r['seq'])
            lookup[key]={**lookup.get(key,{}),**r}
    rows=list(lookup.values())
    groups=[('ronin','seen'),('ronin','unseen'),('ridi','')]
    summary=[]
    for name,template in FAMILIES.items():
        for ds,split in groups:
            for metric in ['ate','rte','suffix_ate','suffix_rte']:
                found=subject_values(rows,template,ds,split,metric)
                if found is None: continue
                subjects,sequence,nseq=found
                subject_seed=np.array(list(subjects.values())).mean(axis=0)
                seq_seed=sequence.mean(axis=0)
                summary.append(dict(family=name,dataset=ds,split=split,metric=metric,
                                    subjects=len(subjects),sequences=nseq,
                                    subject_macro_mean=float(subject_seed.mean()),subject_macro_seed_sd=float(subject_seed.std(ddof=1)),
                                    sequence_macro_mean=float(seq_seed.mean()),sequence_macro_seed_sd=float(seq_seed.std(ddof=1))))
    contrasts=[]
    for control in ['GN only','GN + yaw augmentation','GN + mixed-sensor PCA']:
        for ds,split in groups:
            for metric in ['ate','rte','suffix_ate']:
                a=subject_values(rows,FAMILIES['GN + HeadingNorm'],ds,split,metric)
                b=subject_values(rows,FAMILIES[control],ds,split,metric)
                if a is None or b is None: raise RuntimeError('Missing primary comparison')
                contrasts.append(dict(treatment='GN + HeadingNorm',control=control,dataset=ds,split=split,metric=metric,
                                      primary=(ds=='ridi' or split=='unseen') and metric=='ate' and control!='GN only',
                                      **effect(a[0],b[0])))
    budget,material,ranking=budget_analysis(protocol)
    # All rows and bins are retained; no examples selected by favorable outcomes.
    geometry=[]
    for r in rows:
        for b in r.get('exploratory_geometry_bins',[]):
            geometry.append(dict(model=r['model'],dataset=r['dataset'],split=r['split'],seq=r['seq'],
                                 subject=r['seq'].split('_')[0],**b))
    suffix=[]
    for ds,split in [('ridi',''),('tlio','official_test')]:
        names=(['GN only','GN + yaw augmentation','GN + mixed-sensor PCA','GN + HeadingNorm'] if ds=='ridi' else
               ['Historical full HN-Transformer','Historical ResNet18','Historical IMUNet'])
        for name in names:
            values=subject_values(rows,FAMILIES[name],ds,split,'suffix_ate')
            if values:
                subjects,seq,nseq=values
                full=subject_values(rows,FAMILIES[name],ds,split,'ate')
                suffix.append(dict(family=name,dataset=ds,split=split,sequences=nseq,
                                   full_subject_ate=float(np.array(list(full[0].values())).mean()),
                                   suffix_subject_ate=float(np.array(list(subjects.values())).mean()),
                                   suffix_subject_seed_sd=float(np.array(list(subjects.values())).mean(axis=0).std(ddof=1)),
                                   suffix_sequence_ate=float(seq.mean()),suffix_sequence_seed_sd=float(seq.mean(axis=0).std(ddof=1))))
    S.atomic_json(OUT/'analysis.json',dict(protocol_sha256=S.digest(S.PROTOCOL),sources=sources,
                  summaries=summary,contrasts=contrasts,budget=budget,budget_material_change=material,budget_ranking=ranking,
                  exploratory_geometry_bins=geometry,suffix_sensitivity=suffix,
                  uncertainty_note='Subject percentile bootstrap conditions on four saved training runs. Prefix-derived subject IDs. No p-values; no claim of untouched confirmation.'))
    print(f'[analysis] {len(summary)} summaries, {len(contrasts)} paired contrasts; material budget change={material}')


if __name__=='__main__': main()
