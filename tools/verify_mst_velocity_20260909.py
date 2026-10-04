"""Replay archived seed-0 velocity-coordinate probes without changing frozen artifacts."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import sensors_frontend as F
from sensors_evaluate import load

def main():
 source=ROOT/'results/sensors_v4/diagnostics.json';d=json.loads(source.read_text());torch.set_num_threads(4)
 windows=[]
 for row in d['rotation_window_sources']:
  with np.load(ROOT/row['file'],allow_pickle=False) as z:windows.append(np.asarray(z['feat'][row['start']:row['start']+200].T,np.float32))
 x=torch.from_numpy(np.stack(windows));rows=[]
 for tag in ['Sensors_v4_resnet_gn_s0','Sensors_v4_resnet_yaw_s0','Sensors_v4_resnet_mixed_pca_s0','ResNet18_v3_hn_gn_s0']:
  net,_,_,cp=load(tag,torch.device('cpu'))
  assert hashlib.sha256(cp.read_bytes()).hexdigest()==d['provenance']['checkpoint_sha256'][tag]
  with torch.inference_mode():
   pred=net(x)
   for r in [q for q in d['rotation_tests'] if q['model']==tag]:
    xr,expected=F.rotate_batch(x,pred,torch.full((len(x),),r['angle_rad']));error=(net(xr)-expected).norm(dim=1)
    vals=[float(error.mean()),float(error.max())];old=[r['mean_error_mps'],r['maximum_error_mps']]
    assert np.allclose(vals,old,atol=1e-9,rtol=1e-5),(tag,vals,old)
    rows.append(dict(model=tag,angle_rad=r['angle_rad'],mean_difference=vals[0]-old[0],maximum_difference=vals[1]-old[1]))
 report=dict(status='pass',checkpoints=4,windows=72,angle_count=4,compared_values=32,maximum_difference=max(abs(r[k]) for r in rows for k in ['mean_difference','maximum_difference']),rows=rows)
 (ROOT/'verification/mst_revision_20260909/direct_velocity_replay.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
