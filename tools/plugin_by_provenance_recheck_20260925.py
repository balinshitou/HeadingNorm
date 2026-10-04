"""按模型来源分组复算外挂HN结果（docs/107）。用法：.venv/bin/python tools/plugin_by_provenance_recheck_20260925.py"""
import json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SEED,NB=20260906,10000
def subject(ds,seq):
    if ds.startswith('imunet'): return re.search(r'Subje[ct]+_(\d+)',seq).group(1)
    if ds.startswith('tlio'): return seq
    return seq.split('_')[0]
cell=defaultdict(lambda: defaultdict(dict))
def load(dirs):
    for d in dirs:
        for p in sorted((ROOT/d).glob('*.json')):
            j=json.loads(p.read_text())
            if not isinstance(j,dict) or 'rows' not in j: continue
            assert j['complete'],p
            for r in j['rows']:
                cell[(r['model'],r['dataset'])][r['sequence']][r['seed']]=r
load(['results/unified_eval_20260913','results/confirm_20260913','results/plugin_arch_20260923'])
def boot(x,q):
    rng=np.random.default_rng(SEED); return np.quantile(x[rng.integers(0,len(x),(NB,len(x)))].mean(1),q)
def cmp(t,c,ds,metric='ate_u',q=(.025,.975)):
    A,B=cell[(t,ds)],cell[(c,ds)]
    assert A and set(A)==set(B),(t,c,ds,len(A),len(B))
    ns={len(v) for v in A.values()}|{len(v) for v in B.values()}
    sa={s:np.mean([r[metric] for r in A[s].values()]) for s in A}
    sb={s:np.mean([r[metric] for r in B[s].values()]) for s in B}
    g=defaultdict(list)
    for s in A: g[subject(ds,s)].append(s)
    k=sorted(g); ua=np.array([np.mean([sa[s] for s in g[i]]) for i in k]); ub=np.array([np.mean([sb[s] for s in g[i]]) for i in k])
    ds_=ua-ub; dq=np.array([sa[s]-sb[s] for s in sorted(A)])
    ci_s=boot(ds_,q); ci_q=boot(dq,q)
    ci=ci_q if ds.startswith('tlio') else ci_s
    both = ds.startswith('imunet')
    sigf = (ci_s[1]<0 and ci_q[1]<0) if both else ci[1]<0
    return dict(t=t,c=c,ds=ds,seeds=sorted(ns),nsub=len(k),nseq=len(A),tm=ua.mean(),cm=ub.mean(),d=ds_.mean(),rel=100*ds_.mean()/ub.mean(),
                ci_s=ci_s,ci_q=ci_q,sig=sigf,imp_sub=int((ds_<0).sum()),imp_seq=int((dq<0).sum()))
TEST=['ronin_seen','ronin_unseen','ridi','tlio','imunet']; CONF=['tlio_c','imunet_c']
NAME={'ronin_seen':'RoNIN训练受试者组','ronin_unseen':'RoNIN独立受试者组','ridi':'RIDI','tlio':'TLIO测试','imunet':'手机测试','tlio_c':'TLIO确认318','imunet_c':'手机确认87'}
def show(title,pairs,dss,metric='ate_u',q=(.025,.975)):
    print(f'\n## {title}  [{metric}]')
    for t,c in pairs:
        for ds in dss:
            if not cell[(t,ds)]: print(f'  {t} | {NAME[ds]}: 无数据'); continue
            r=cmp(t,c,ds,metric,q)
            ci=r['ci_q'] if ds.startswith('tlio') else r['ci_s']
            print(f"  {t:24s} vs {c:18s} | {NAME[ds]:10s} | seeds{r['seeds']} n={r['nsub']}/{r['nseq']} | {r['cm']:.3f}→{r['tm']:.3f} | {r['rel']:+.1f}% | CI[{ci[0]:+.3f},{ci[1]:+.3f}]{' seqCI[%+.3f,%+.3f]'%tuple(r['ci_q']) if ds.startswith('imunet') else ''} | {'显著降' if r['sig'] else ('显著升' if (ci[0]>0 and (r['ci_q'][0]>0 or not ds.startswith('imunet'))) else '含零')} | 改善 {r['imp_sub']}/{r['nsub']}")
print('模型×数据集可用单元：')
for k in sorted({m for m,_ in cell}): print(' ',k, sorted(ds for m,ds in cell if m==k))
# 组1 本文训练的增强网络，外挂
show('组1 ResNet18增强网络：外挂HN',[('ours_yaw_hn','ours_yaw')],TEST+CONF)
show('组1 ResNet18增强网络：两方向帧平均',[('ours_yaw_fa2','ours_yaw')],TEST+CONF)
show('组1 ResNet18增强网络：C8',[('ours_yaw_tta8','ours_yaw')],TEST+CONF)
show('组1 Transformer增强网络：外挂HN',[('int_transformer_yaw_hn','int_transformer_yaw')],TEST+CONF)
show('组1 IMUNet增强网络：外挂HN',[('int_imunet_yaw_hn','int_imunet_yaw')],TEST+CONF)
# 组2 已发表模型
show('组2 RoNIN ResNet(原作者)：外挂HN',[('ext_resnet_hn','ext_resnet')],TEST+CONF)
show('组2 RoNIN LSTM(原作者)：外挂HN',[('ext_lstm_hnseq','ext_lstm')],TEST)
show('组2 RoNIN TCN(原作者)：外挂HN',[('ext_tcn_hnseq','ext_tcn')],TEST)
# 形状误差（TLIO确认）
show('形状误差 TLIO确认',[('ours_yaw_hn','ours_yaw'),('int_transformer_yaw_hn','int_transformer_yaw'),('int_imunet_yaw_hn','int_imunet_yaw'),('ext_resnet_hn','ext_resnet')],['tlio_c'],metric='ate_shape')
# 确认族的区间水平
Q8=(0.05/16,1-0.05/16); Q2=(0.05/4,1-0.05/4)
show('确认族 99.375%',[('ours_yaw_hn','ours_yaw'),('ours_yaw_fa2','ours_yaw'),('ours_hn','ours_pca'),('ext_resnet_hn','ext_resnet')],CONF,q=Q8)
show('跨架构族 97.5%',[('int_transformer_yaw_hn','int_transformer_yaw'),('int_imunet_yaw_hn','int_imunet_yaw')],['tlio_c'],q=Q2)
# 训练时用HN（组1b）统一口径
show('组1b 训练时用HN vs 增强/混合PCA/GN（统一口径）',[('ours_hn','ours_yaw'),('ours_hn','ours_pca'),('ours_hn','ours_gn')],TEST+CONF)
