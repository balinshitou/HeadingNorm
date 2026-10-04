"""Replay a declared, score-independent illustration from frozen checkpoints."""
from pathlib import Path
import sys,json,hashlib,csv
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import models,sensors_frontend,e6_frontend,pkg_common as P,pkg_eval as E
from e1_heading_repeatability import sequences,rotate_horizontal,rotate_2d
OUT=ROOT/'verification/mst_revision_20260909/illustration'

def main():
 OUT.mkdir(parents=True,exist_ok=True);device=torch.device('mps');torch.set_num_threads(4)
 path,ds,split=min((r for r in sequences() if r[1:]==('ronin','unseen')),key=lambda r:str(r[0]))
 with np.load(path,allow_pickle=False) as z:
  feat=np.asarray(z['feat'],np.float32);tm,te,gt=(np.asarray(z[k],float) for k in ['tm','te','gt']);seq=str(z['seq']);rate=float(z['rate'])
 arrays={'time_s':te-te[0],'ground_truth_m':gt};rows=[];sources={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()}
 for arm,folder,tag in [('yaw','sensors_v4','Sensors_v4_resnet_yaw_s0'),('ns','e6_sign_ablation','E6_resnet_heading_nosign_s0'),('hn','submission_frozen','ResNet18_v3_hn_gn_s0')]:
  cp=ROOT/'models'/folder/(tag+'.pt');ck=torch.load(cp,map_location='cpu',weights_only=False)
  if 'spec' in ck:
   s=ck['spec'];factory=e6_frontend if arm=='ns' else sensors_frontend
   net=factory.build(s['frontend'],ck['gnorm_mu'],ck['gnorm_sd'],s['arch'],s['pre'],s['width'])
  else:net=models.build(ck['arch'],pre=ck['pre'],width=ck['width'],hnorm=ck['hnorm'],nout=ck['nout'],gnorm=bool(ck['gnorm']),gnorm_mu=ck['gnorm_mu'],gnorm_sd=ck['gnorm_sd'],rate=ck['rate'])
  net.load_state_dict(ck['model_state_dict'],strict=True);net=net.eval().to(device)
  source=ROOT/'results'/('e6_sign_ablation' if arm=='ns' else 'e1_heading_repeatability')/(tag+'.json')
  saved=json.loads(source.read_text());sha=hashlib.sha256(cp.read_bytes()).hexdigest();assert saved['checkpoint_sha256']==sha
  sources[str(cp.relative_to(ROOT))]=sha;sources[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
  for k in [0,2,4,6]:
   psi=2*np.pi*k/8;x=feat if k==0 else rotate_horizontal(feat,psi)
   ids,v=E.predict(net,x,device,1.,200.,chunk=512);v=v if k==0 else rotate_2d(v,-psi)
   traj=P.trajectory_from_velocity(v,ids,tm,te);aligned,_,_=P.align_registered(ds,traj,gt,te)
   ate,rte=P.ate_rte(aligned,gt,rate);old=next(r for r in saved['rows'] if r['seq']==seq and r['angle_index']==k)
   assert np.allclose([ate,rte],[old['ate'],old['rte']],rtol=1e-6,atol=1e-5)
   arrays[f'{arm}_{k}_position_m']=aligned
   if k==0:base=aligned.copy()
   rows.append(dict(arm=arm,model=tag,sequence=seq,angle_degrees=45*k,ate_m=ate,rte_m=rte,ate_replay_difference_m=ate-old['ate'],rte_replay_difference_m=rte-old['rte'],maximum_trajectory_difference_from_zero_m=float(np.linalg.norm(aligned-base,axis=1).max())))
  del net,ck;torch.mps.empty_cache()
 np.savez_compressed(OUT/'trajectories.npz',**arrays)
 (OUT/'report.json').write_text(json.dumps(dict(sequence=seq,selection='Alphabetically first RoNIN unseen sequence, seed 0, angles 0, 90, 180, 270 degrees; chosen without reading scores',source_sha256=sources,rows=rows),indent=2)+'\n')
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
