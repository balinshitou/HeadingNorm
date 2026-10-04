"""Prespecified-window rotation probes and measured batch-one inference time."""
from __future__ import annotations
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'src'))
import sensors_frontend as F
import sensors_train as S
from sensors_evaluate import load

TAGS=['Sensors_v4_resnet_gn_s0','Sensors_v4_resnet_yaw_s0','Sensors_v4_resnet_mixed_pca_s0',
      'ResNet18_v3_hn_gn_s0','A6_v3_nopre_s0','IMUNet2024_v3_hn_gn_s0',
      'A6_v2_f100_s0','IMUNet2024_v21_matched_s0']


def collect():
    report_path=ROOT/'results/sensors_v4/diagnostics.json'
    window_source=ROOT/'verification/generated/independent_review_20260906/diagnostics.json'
    checkpoints={tag:next(p for p in [S.OUT/f'{tag}.pt',ROOT/'models/submission_frozen'/f'{tag}.pt'] if p.exists()) for tag in TAGS}
    provenance=dict(diagnostic_source_sha256=S.digest(Path(__file__)),window_source_sha256=S.digest(window_source),
                    checkpoint_sha256={tag:S.digest(p) for tag,p in checkpoints.items()})
    if report_path.exists():
        previous=json.loads(report_path.read_text())
        if previous.get('provenance')==provenance:
            print('[diagnostics] verified existing measurements',flush=True)
            return previous
    torch.set_num_threads(4)
    source=json.loads(window_source.read_text())
    windows=[]
    for row in source['rotation_window_sources']:
        with np.load(ROOT/row['file'],allow_pickle=False) as z:
            windows.append(np.asarray(z['feat'][row['start']:row['start']+200].T,np.float32))
    x=torch.from_numpy(np.stack(windows))
    rotations=[]
    for tag in TAGS:
        net,_,_,_=load(tag,torch.device('cpu'))
        with torch.inference_mode():
            pred=net(x)
            for angle in [np.pi/6,np.pi/2,np.pi,5*np.pi/3]:
                xr,expected=F.rotate_batch(x,pred,torch.full((len(x),),angle))
                error=(net(xr)-expected).norm(dim=1)
                rotations.append(dict(model=tag,angle_rad=float(angle),windows=len(x),
                                      mean_error_mps=float(error.mean()),maximum_error_mps=float(error.max())))
    timings=[]
    devices=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])
    for device in devices:
        dev=torch.device(device)
        for tag in TAGS:
            net,_,_,checkpoint=load(tag,dev)
            sample=x[:1].to(dev)
            def sync():
                if device=='mps': torch.mps.synchronize()
            with torch.inference_mode():
                for _ in range(30): net(sample)
                sync()
                elapsed=[]
                for _ in range(100):
                    started=time.perf_counter()
                    net(sample)
                    sync()
                    elapsed.append((time.perf_counter()-started)*1000)
            timings.append(dict(model=tag,device=device,batch=1,precision='float32',warmup=30,repetitions=100,
                                median_ms=float(np.median(elapsed)),p95_ms=float(np.quantile(elapsed,.95)),
                                parameters=sum(p.numel() for p in net.parameters()),
                                parameter_bytes=sum(p.numel()*p.element_size() for p in net.parameters()),
                                checkpoint_sha256=S.digest(checkpoint)))
    hardware={}
    if sys.platform=='darwin':
        for key in ['machdep.cpu.brand_string','hw.memsize']:
            hardware[key]=subprocess.check_output(['sysctl','-n',key],text=True).strip()
    report=dict(provenance=provenance,hardware=hardware,platform=platform.platform(),torch=torch.__version__,cpu_threads=torch.get_num_threads(),
                scope='Exploratory probes on 72 fixed windows and eight seed-0 checkpoints. Rotation consistency is not accuracy. Timings include model frontends and output restoration; exclude data loading, orientation preprocessing and trajectory integration. Mixed PCA includes explicit CPU eigensolver transfer. Parameter bytes are not peak runtime memory. No smartphone performance claim.',
                rotation_window_sources=source['rotation_window_sources'],rotation_tests=rotations,timings=timings)
    S.atomic_json(report_path,report)
    return report


if __name__=='__main__': collect()
