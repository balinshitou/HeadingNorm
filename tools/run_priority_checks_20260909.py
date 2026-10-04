"""Run the frozen post-review checks in a fresh destination without cached predictions."""
from pathlib import Path
import argparse, hashlib, json
import torch
import review_priority_diagnostics_20260909 as D

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',required=True,choices=['official','temporal','loss'])
    ap.add_argument('--output-dir',required=True,type=Path)
    ap.add_argument('--device',default='mps')
    ap.add_argument('--checkpoint',type=Path,default=D.hn_paths.RONIN_WEIGHTS/'ronin_resnet/checkpoint_gsn_latest.pt')
    ap.add_argument('--official-source',type=Path,default=D.hn_paths.RONIN_SRC/'model_resnet1d.py')
    args=ap.parse_args();dest=args.output_dir.resolve()
    if dest.exists():raise SystemExit('The output directory must be new. Existing results are never overwritten.')
    dest.mkdir(parents=True);D.OUT=dest
    for p in [dest/'official_arrays',dest/'temporal_arrays']:p.mkdir()
    args.device=torch.device(args.device);torch.set_num_threads(4)
    getattr(D,args.mode)(args)
    files={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dest.rglob('*')) if p.is_file()}
    (dest/'run_manifest.json').write_text(json.dumps(dict(mode=args.mode,device=str(args.device),fresh_inference=True,files=files),indent=2)+'\n')
if __name__=='__main__':main()
