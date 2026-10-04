"""Post-review evaluation of frozen models. No fitting or checkpoint selection.

Run with --mode official, temporal, or loss. Reports retain source hashes,
sequence results, and arrays for independent reanalysis. Existing results are read-only.
"""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, sys, time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import hn_paths   # 外部资料路径表
import pkg_common as P, pkg_eval as E, pkg_train as T
from sensors_evaluate import load
from e1_heading_repeatability import sequences
OUT = ROOT/'verification/priority_revision_20260909'

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def csvwrite(name, rows):
    with (OUT/name).open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def wrap(x): return np.arctan2(np.sin(x),np.cos(x))
def inputs(): return [r[0] for r in sequences() if r[1:]==('ronin','unseen')]
def cache(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:np.asarray(z[k]) for k in ['feat','tm','te','gt','seq','rate']}
def module(path, name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def official(args):
    m=module(args.official_source,'review_official_resnet')
    net=m.ResNet1D(6,2,m.BasicBlock1D,[2,2,2,2],base_plane=64,output_block=m.FCOutputModule,
                  kernel_size=3,fc_dim=512,in_dim=7,dropout=.5,trans_planes=128)
    obj=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    net.load_state_dict(obj['model_state_dict'],strict=True);net.eval().to(args.device)
    old=json.loads((ROOT/'results/published/eval_a5.json').read_text())
    rows=[]
    for i,p in enumerate(inputs()):
        z=cache(p);seq=str(z['seq']);ids,v=E.predict(net,z['feat'],args.device,1.,200.,chunk=512)
        xy=P.trajectory_from_velocity(v,ids,z['tm'],z['te']);xy,_,_=P.align_registered('ronin',xy,z['gt'],z['te'])
        a,r=P.ate_rte(xy,z['gt'],float(z['rate']))
        ref=next(q for q in old if q['model']=='official' and q['seq']==seq and q['dataset']=='ronin' and q['align']=='registered')
        rows.append(dict(sequence=seq,subject=seq.split('_')[0],ate=a,rte=r,archived_ate=ref['ate'],archived_rte=ref['rte'],ate_difference=a-ref['ate'],rte_difference=r-ref['rte'],cache_sha256=digest(p)))
        np.savez_compressed(OUT/'official_arrays'/f'{seq}.npz',ids=ids,velocity=v,position=xy)
        print(f'official {i+1}/32 {seq} ATE {a:.6f}',flush=True)
    csvwrite('official_replay.csv',rows)
    summary={}
    for k in ['ate','rte']:
        subjects=sorted({r['subject'] for r in rows})
        summary[k]=dict(sequence_macro=float(np.mean([r[k] for r in rows])),subject_macro=float(np.mean([np.mean([r[k] for r in rows if r['subject']==s]) for s in subjects])),max_abs_archive_difference=max(abs(r[k+'_difference']) for r in rows))
    write('official_report.json',dict(device=str(args.device),sequences=len(rows),subjects=len(subjects),checkpoint=str(args.checkpoint),checkpoint_sha256=digest(args.checkpoint),official_source=str(args.official_source),source_sha256=digest(args.official_source),script_sha256=digest(__file__),summary=summary,passed=all(summary[k]['max_abs_archive_difference']<1e-4 for k in summary)))

def geometry(feat, ids):
    import models
    hn=models.HeadingNorm()
    theta=[];gap=[];skew=[];raw=[]
    for j in range(0,len(ids),512):
        x=torch.from_numpy(np.stack([feat[i:i+200].T for i in ids[j:j+512]]).astype('float32'))
        with torch.no_grad():
            _,th=hn(x)
            ac=x[:,3:5]-x[:,3:5].mean(-1,keepdim=True)
            a=(ac[:,0]**2).mean(-1);b=(ac[:,1]**2).mean(-1);c=(ac[:,0]*ac[:,1]).mean(-1)
            th0=.5*torch.atan2(2*c,a-b);q=ac[:,0]*th0.cos()[:,None]+ac[:,1]*th0.sin()[:,None]
            m=(q**3).mean(-1)
            theta.extend(th.numpy());raw.extend(th0.numpy())
            gap.extend((torch.sqrt((a-b)**2+4*c*c)/torch.clamp(a+b,min=1e-12)).numpy())
            skew.extend((m.abs()/torch.clamp((q**2).mean(-1)**1.5,min=1e-12)).numpy())
    return {k:np.asarray(v) for k,v in [('theta',theta),('theta0',raw),('gap',gap),('kappa',skew)]}

def temporal(args):
    protocol=dict(scope='All 32 RoNIN unseen sequences and four saved ResNet18 seeds, HN and yaw augmentation',
      jump_threshold_degrees=90,sensitivity_thresholds_degrees=[60,120],event_neighbourhood_seconds=.5,
      frame_change='atan2(sin(theta[t]-theta[t-1]),cos(theta[t]-theta[t-1]))',
      target='One-second displacement from interpolated evaluation positions, same as original S5',
      low_gap=.1,low_kappa=.05,speed_edges=[0,.3,1,2,None],
      horizons_seconds=[1,5,10,30,60,120],acf_lags_seconds=[.05,.5,1,5,10,30,60],
      status='Exploratory post-review protocol fixed before reading new diagnostic results',
      causal_scope='Associations only. No frame smoothing intervention and no retraining.')
    if (OUT/'temporal_protocol.json').exists():assert json.loads((OUT/'temporal_protocol.json').read_text())==protocol
    else:write('temporal_protocol.json',protocol)
    models_by_seed={}
    for seed in range(4):
        models_by_seed[seed]={a:load(t.format(seed),args.device) for a,t in [('yaw','Sensors_v4_resnet_yaw_s{}'),('hn','ResNet18_v3_hn_gn_s{}')]}
    records=[];events=[];strata=[];horizons=[];acfs=[];geometry_rows=[]
    for ni,p in enumerate(inputs()):
        z=cache(p);seq=str(z['seq']);subject=seq.split('_')[0];feat=z['feat'];ids=np.arange(0,len(feat)-200,10)
        geopath=OUT/'temporal_arrays'/f'{seq}_geometry.npz'
        if geopath.exists():
            with np.load(geopath) as q:g={k:q[k] for k in ['theta','theta0','gap','kappa']}
        else:
            g=geometry(feat,ids);np.savez_compressed(geopath,ids=ids,**g)
        d=np.r_[0,np.abs(wrap(np.diff(g['theta'])))]
        jump=d>np.pi/2
        near=np.convolve(jump.astype(int),np.ones(21,dtype=int),mode='same')>0
        lowk=g['kappa']<.05;lowg=g['gap']<.1
        endpointlow=np.r_[False,lowk[1:]|lowk[:-1]]
        theta0branch=np.r_[False,np.abs(np.diff(g['theta0']))>np.pi/2]
        geometry_rows.append(dict(sequence=seq,subject=subject,windows=len(ids),transitions=len(ids)-1,jumps=int(jump.sum()),near_windows=int(near.sum()),low_kappa_windows=int(lowk.sum()),low_gap_windows=int(lowg.sum()),low_endpoint_transitions=int(endpointlow.sum()),jumps_with_low_endpoint=int((jump&endpointlow).sum()),raw_axis_branch_crossings=int(theta0branch.sum()),raw_branch_without_frame_jump=int((theta0branch&~jump).sum()),jumps60=int((d>np.pi/3).sum()),jumps120=int((d>2*np.pi/3).sum())))
        start=z['tm'][ids];target=np.stack([np.interp(start+1,z['te'],z['gt'][:,i])-np.interp(start,z['te'],z['gt'][:,i]) for i in range(2)],1)
        speed=np.linalg.norm(target,axis=1);dt=float(np.diff(start).mean())
        for seed, nets in models_by_seed.items():
            for arm,(net,w,rate,cp) in nets.items():
                dest=OUT/'temporal_arrays'/f'{seq}_{arm}_s{seed}.npz'
                if dest.exists():
                    with np.load(dest) as q:v=q['velocity']
                else:
                    got,v=E.predict(net,feat,args.device,w,rate,chunk=512);assert np.array_equal(got,ids)
                    np.savez_compressed(dest,ids=ids,velocity=v,target=target)
                xy=P.trajectory_from_velocity(v,ids,z['tm'],z['te']);xy,_,_=P.align_registered('ronin',xy,z['gt'],z['te']);a,r=P.ate_rte(xy,z['gt'],float(z['rate']))
                saved=json.loads((ROOT/'results/e1_heading_repeatability'/f'{cp.stem}.json').read_text())
                old=next(q for q in saved['rows'] if q['seq']==seq and q['dataset']=='ronin' and q['angle_index']==0)
                assert np.allclose([a,r],[old['ate'],old['rte']],atol=1e-5,rtol=1e-6),(seq,arm,seed,a,r,old)
                error=v-target;err=np.linalg.norm(error,axis=1)
                base=dict(sequence=seq,subject=subject,seed=seed,arm=arm)
                records.append(dict(**base,windows=len(ids),mean_error=float(err.mean()),p95_error=float(np.quantile(err,.95)),fraction_error_over_1=float((err>1).mean()),ate=a,rte=r,ate_replay_difference=a-old['ate'],rte_replay_difference=r-old['rte']))
                for label,mask in [('all',np.ones(len(ids),bool)),('jump',jump),('near_jump',near),('away',~near),('low_gap',lowg),('other_gap',~lowg),('low_kappa',lowk),('other_kappa',~lowk)]:
                    if mask.any():events.append(dict(**base,condition=label,windows=int(mask.sum()),mean_error=float(err[mask].mean()),p95_error=float(np.quantile(err[mask],.95)),fraction_error_over_1=float((err[mask]>1).mean())))
                for lo,hi in zip([0,.3,1,2],[.3,1,2,np.inf]):
                    for label,mask in [('near_jump',near),('away',~near)]:
                        take=mask&(speed>=lo)&(speed<hi)
                        if take.any():strata.append(dict(**base,speed_lower=lo,speed_upper=None if np.isinf(hi) else hi,condition=label,windows=int(take.sum()),mean_error=float(err[take].mean())))
                mu=error.mean(0);center=error-mu
                for lag in protocol['acf_lags_seconds']:
                    k=int(round(lag/.05));l=center[:-k];rr=center[k:]
                    denom=np.sqrt(np.mean(np.sum(l*l,1))*np.mean(np.sum(rr*rr,1)))
                    acfs.append(dict(**base,lag_seconds=lag,correlation=float(np.mean(np.sum(l*rr,1))/denom)))
                # Exact finite-sample decomposition of integrated diagnostic residuals.
                # Does not replace the official RTE evaluator or its reconstruction endpoints.
                csum=np.vstack([np.zeros(2),np.cumsum(error*dt,axis=0)])
                for h in protocol['horizons_seconds']:
                    k=int(round(h/.05));sums=csum[k:]-csum[:-k];bias=mu*(k*dt);fluct=sums-bias
                    total=float(np.mean(np.sum(sums*sums,1)));b=float(bias@bias);f=float(np.mean(np.sum(fluct*fluct,1)));cross=float(2*np.mean(fluct@bias))
                    assert np.isclose(total,b+f+cross,atol=1e-9,rtol=1e-10)
                    horizons.append(dict(**base,horizon_seconds=h,total_squared_m=total,sequence_mean_squared_m=b,centered_squared_m=f,cross_squared_m=cross,identity_error=total-b-f-cross))
        print(f'temporal {ni+1}/32 {seq} windows={len(ids)} jumps={jump.sum()}',flush=True)
    for name,rows in [('temporal_scores',records),('temporal_events',events),('temporal_speed_strata',strata),('temporal_horizons',horizons),('temporal_acf',acfs),('temporal_geometry',geometry_rows)]:csvwrite(name+'.csv',rows)
    write('temporal_report.json',dict(status='pass',sequences=32,seeds=4,checkpoints=8,sequence_model_evaluations=len(records),replayed_metrics=len(records)*2,maximum_replay_difference=max(abs(r[k]) for r in records for k in ['ate_replay_difference','rte_replay_difference']),checkpoint_sha256={cp.stem:digest(cp) for nets in models_by_seed.values() for _,_,_,cp in nets.values()},cache_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs()},script_sha256=digest(__file__),protocol=protocol))

def loss(args):
    # Analytic counterexample, followed by paired real-window parameter gradients.
    e=torch.tensor([[.5,0.]],dtype=torch.float64);c=2**-.5;r=torch.tensor([[c,-c],[c,c]],dtype=torch.float64)
    fn=T.make_loss('huber',.27);example=[float(fn(e,torch.zeros_like(e))),float(fn(e@r.T,torch.zeros_like(e)))]
    assert abs(example[0]-example[1])>.009
    path=inputs()[0];z=cache(path);starts=np.linspace(0,len(z['feat'])-201,32,dtype=int)
    x=torch.from_numpy(np.stack([z['feat'][j:j+200].T for j in starts]).astype('float32'))
    ts=z['tm'][starts];target=np.stack([np.interp(ts+1,z['te'],z['gt'][:,i])-np.interp(ts,z['te'],z['gt'][:,i]) for i in range(2)],1)
    y=torch.tensor(target,dtype=torch.float32);net,_,_,cp=load('ResNet18_v3_hn_gn_s0',torch.device('cpu'))
    from sensors_frontend import rotate_batch
    values=[];original={}
    for kind in ['huber','mse']:
        for angle in [0,45,90]:
            xx,yy=rotate_batch(x,y,torch.full((len(x),),angle*np.pi/180))
            net.zero_grad(set_to_none=True);pred=net(xx);l=T.make_loss(kind,.27)(pred,yy);l.backward()
            grad=torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None]).detach()
            if angle==0:original[kind]=grad.clone()
            rel=float((grad-original[kind]).norm()/original[kind].norm())
            values.append(dict(loss=kind,angle_degrees=angle,value=float(l.detach()),relative_parameter_gradient_change=rel))
    write('loss_report.json',dict(status='pass',coordinate_huber_example=example,sequence=str(z['seq']),window_starts=starts.tolist(),window_selection='32 equally spaced windows from alphabetically first unseen sequence',mode='eval with gradients enabled; no parameter updates; MSE diagnostic only',rows=values,checkpoint_sha256=digest(cp),script_sha256=digest(__file__)))
    print(json.dumps(values,indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['official','temporal','loss'],required=True)
    ap.add_argument('--device',default='mps');ap.add_argument('--checkpoint',type=Path,default=hn_paths.RONIN_WEIGHTS/'ronin_resnet/checkpoint_gsn_latest.pt')
    ap.add_argument('--official-source',type=Path,default=hn_paths.RONIN_SRC/'model_resnet1d.py')
    args=ap.parse_args();args.device=torch.device(args.device);torch.set_num_threads(4)
    for p in [OUT,OUT/'official_arrays',OUT/'temporal_arrays']:p.mkdir(parents=True,exist_ok=True)
    globals()[args.mode](args)
if __name__=='__main__':main()
