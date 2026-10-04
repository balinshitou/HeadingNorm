"""Evaluate revision checkpoints without changing the frozen benchmark files."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import pkg_common as P
import pkg_eval as E
import sensors_frontend as F
import sensors_train as S

OUT = ROOT / 'results/sensors_v4/evaluation'


def load(tag, device):
    path = S.OUT / f'{tag}.pt'
    if path.exists():
        metadata_path = path.with_suffix('.json')
        if not metadata_path.exists():
            raise RuntimeError(f'Incomplete training: {tag}')
        metadata = json.loads(metadata_path.read_text())
        if S.digest(path) != metadata['checkpoint_sha256']:
            raise RuntimeError(f'Checkpoint digest mismatch: {tag}')
        o = torch.load(path, map_location='cpu', weights_only=False)
        spec = o['spec']
        net = F.build(spec['frontend'], o['gnorm_mu'], o['gnorm_sd'],
                      spec['arch'], spec['pre'], spec['width'])
        net.load_state_dict(o['model_state_dict'], strict=True)
        return net.eval().to(device), 1., 200., path
    path = ROOT / 'models/submission_frozen' / f'{tag}.pt'
    if not path.exists():
        raise FileNotFoundError(path)
    o = torch.load(path, map_location='cpu', weights_only=False)
    import models
    net = models.build(o['arch'], pre=o['pre'], width=o['width'], hnorm=o['hnorm'],
                       gnorm=bool(o['gnorm']), gnorm_mu=o['gnorm_mu'], gnorm_sd=o['gnorm_sd'], rate=o['rate'])
    net.load_state_dict(o['model_state_dict'], strict=True)
    return net.eval().to(device), float(o['win_sec']), float(o['rate']), path


def score(aligned, gt, ts, rate, prefix_seconds):
    ate,rte=P.ate_rte(aligned,gt,rate)
    suffix=np.asarray(ts) - ts[0] > prefix_seconds
    if prefix_seconds == 0:
        suffix=np.ones(len(ts),dtype=bool)
    if suffix.sum() < 2:
        raise RuntimeError('Sequence too short for suffix evaluation')
    sa,sr=P.ate_rte(aligned[suffix],gt[suffix],rate)
    return dict(ate=ate,rte=rte,suffix_ate=sa,suffix_rte=sr,
                suffix_samples=int(suffix.sum()),prefix_seconds=prefix_seconds)


def geometry_bins(feat, ids, velocity, tm, te, gt):
    """Exploratory error-by-conditioning bins; no GT-fitted yaw and no tuning."""
    x=np.stack([feat[j:j+200,3:5] for j in ids])
    x=x-x.mean(axis=1,keepdims=True)
    cov=np.einsum('nti,ntj->nij',x,x)/200
    a,b,c=cov[:,0,0],cov[:,1,1],cov[:,0,1]
    delta=np.sqrt((a-b)**2+4*c*c)
    gap=delta/np.maximum(a+b,1e-12)
    theta=.5*np.arctan2(2*c,a-b)
    projection=x[:,:,0]*np.cos(theta)[:,None]+x[:,:,1]*np.sin(theta)[:,None]
    skew=np.abs((projection**3).mean(axis=1))/np.maximum((projection**2).mean(axis=1)**1.5,1e-12)
    start=tm[ids]
    target=np.stack([(np.interp(start+1,te,gt[:,i])-np.interp(start,te,gt[:,i])) for i in range(2)],axis=1)
    error=np.linalg.norm(velocity-target,axis=1)
    speed=np.linalg.norm(target,axis=1)
    bins=[]
    for name,value,edges in [('relative_eigenvalue_gap',gap,[0,.1,.3,.6,float('inf')]),
                             ('absolute_standardized_third_moment',skew,[0,.05,.2,1,float('inf')]),
                             ('target_speed_mps',speed,[0,.1,.3,1,2,float('inf')])]:
        for lo,hi in zip(edges[:-1],edges[1:]):
            mask=(value>=lo)&(value<hi)
            bins.append(dict(variable=name,lower=lo,upper=None if np.isinf(hi) else hi,
                             windows=int(mask.sum()),sum_velocity_error=float(error[mask].sum())))
    return bins


def jobs():
    protocol=json.loads(S.PROTOCOL.read_text())
    runs=[(r['tag'], {'ronin','ridi'}) for r in S.expanded_runs(protocol)]
    runs.extend((f'ResNet18_v3_hn_gn_s{s}', {'ronin','ridi'}) for s in range(4))
    for template in ['A6_v2_f100_s{}','ResNet18_v2_matched_s{}','IMUNet2024_v21_matched_s{}']:
        runs.extend((template.format(s),{'ridi','tlio'}) for s in range(4))
    return runs


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tag')
    ap.add_argument('--primary-only',action='store_true',help='Evaluate completed primary runs while the independent budget checks finish')
    ap.add_argument('--limit-per-group',type=int)
    args=ap.parse_args()
    device=P.pick_device()
    torch.set_num_threads(4)
    output=OUT if args.limit_per_group is None else ROOT/'verification/generated/sensors_v4/eval_smoke'
    output.mkdir(parents=True,exist_ok=True)
    files=[]
    for path in sorted(P.CACHE_EVAL.glob('*.npz'))+sorted(P.CACHE_TLIO.glob('*.npz')):
        with np.load(path,allow_pickle=False) as z:
            ds,split=str(z['dataset']),str(z['split'])
        if ds in {'ronin','ridi','tlio'}:
            files.append((path,ds,split))
    evaluator_hash=S.digest(Path(__file__))
    primary_seed0={'Sensors_v4_resnet_gn_s0','Sensors_v4_resnet_yaw_s0',
                   'Sensors_v4_resnet_mixed_pca_s0','ResNet18_v3_hn_gn_s0'}
    for tag,datasets in jobs():
        if args.tag and tag!=args.tag:
            continue
        if args.primary_only and not (tag.startswith('Sensors_v4_resnet_') or tag.startswith('ResNet18_v3_hn_gn_')):
            continue
        started=time.monotonic()
        net,win_sec,rate,checkpoint=load(tag,device)
        wanted=[]
        counts={}
        for path,ds,split in files:
            key=(ds,split)
            if ds not in datasets or args.limit_per_group and counts.get(key,0)>=args.limit_per_group:
                continue
            wanted.append((path,ds,split)); counts[key]=counts.get(key,0)+1
        out=output/f'{tag}.json'
        provenance=dict(model=tag,checkpoint_sha256=S.digest(checkpoint),evaluator_sha256=evaluator_hash,
                        expected_sequences=len(wanted),device=str(device))
        rows=[]
        if out.exists():
            previous=json.loads(out.read_text())
            if all(previous.get(k)==v for k,v in provenance.items()):
                if previous['complete']:
                    print(f'[verified existing evaluation] {tag}',flush=True)
                    continue
                rows=previous['rows']
        finished={r['cache_path'] for r in rows}
        for path,ds,split in wanted:
            rel=str(path.relative_to(ROOT))
            if rel in finished:
                continue
            with np.load(path,allow_pickle=False) as z:
                feat=np.asarray(z['feat'],np.float32)
                tm,te,gt=(np.asarray(z[k],float) for k in ['tm','te','gt'])
                seq=str(z['seq'])
                score_rate=float(z['rate']) if 'rate' in z else 200.
            ids,velocity=E.predict(net,feat,device,win_sec,rate,chunk=512)
            if ids is None or not np.isfinite(velocity).all():
                raise RuntimeError(f'Invalid predictions {tag}/{seq}')
            trajectory=P.trajectory_from_velocity(velocity,ids,tm,te)
            if ds=='tlio':
                aligned,n_prefix,angle=P.align_prefix_rigid_2d(trajectory,gt,te,10.)
            else:
                aligned,n_prefix,angle=P.align_registered(ds,trajectory,gt,te)
            row=dict(model=tag,dataset=ds,split=split,seq=seq,subject=seq.split('_')[0],
                     cache_path=rel,alignment_prefix_samples=n_prefix,theta0=angle,
                     **score(aligned,gt,te,score_rate,0. if ds=='ronin' else 10.))
            if not np.isfinite([row[k] for k in ['ate','rte','suffix_ate','suffix_rte']]).all():
                raise RuntimeError(f'Invalid metrics {tag}/{seq}')
            if tag in primary_seed0:
                row['exploratory_geometry_bins']=geometry_bins(feat,ids,velocity,tm,te,gt)
            rows.append(row)
            S.atomic_json(out,{**provenance,'complete':False,'rows':rows})
        if len(rows)!=len(wanted):
            raise RuntimeError('Evaluation row count mismatch')
        S.atomic_json(out,{**provenance,'complete':True,'rows':rows,
                          'elapsed_seconds':time.monotonic()-started})
        print(f'[evaluated] {tag}: {len(rows)} sequences; {time.monotonic()-started:.1f}s',flush=True)


if __name__=='__main__': main()
