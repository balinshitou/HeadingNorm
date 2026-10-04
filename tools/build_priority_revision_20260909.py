"""Build the MST revision from the same frozen evidence and a declared replay illustration."""
from pathlib import Path
import json,re,csv,hashlib
import numpy as np
import matplotlib.pyplot as plt
import build_hn_revision_20260908 as B
import check_hn_style_20260908 as STYLE
ROOT=B.ROOT
B.OUT=ROOT/'paper/priority_revision_20260909';B.DATA=B.OUT/'source_data';B.FIG=B.OUT/'figures'
for p in [B.OUT,B.DATA,B.FIG]:p.mkdir(parents=True,exist_ok=True)

def order_references():
 p=B.OUT/'manuscript_en.md';s=p.read_text();body,refs=s.split('## References\n',1)
 records={int(m[1]):m[2].strip() for m in re.finditer(r'(?ms)^\[(\d+)\] (.*?)(?=^\[\d+\] |\Z)',refs)}
 mapping={}
 def replace(m):
  nums=[int(n) for n in m[1].split(',')]
  for n in nums:
   assert n in records,n
   if n not in mapping:mapping[n]=len(mapping)+1
  return '['+','.join(str(mapping[n]) for n in nums)+']'
 body=re.sub(r'\[(\d+(?:,\d+)*)\]',replace,body)
 assert set(mapping)==set(records),(set(records)-set(mapping))
 p.write_text(body+'## References\n\n'+'\n\n'.join(f'[{new}] {records[old]}' for old,new in mapping.items())+'\n')
 (B.DATA/'reference_number_map.json').write_text(json.dumps(mapping,indent=2)+'\n')

def main():
 from build_priority_additions_20260909 import prepare
 prepare(B)
 report_path=ROOT/'verification/mst_revision_20260909/illustration/report.json'
 replay=json.loads(report_path.read_text());B.TOKENS['EXAMPLE_SEQUENCE']=replay['sequence'].replace('_',r'\_')
 d=B.read('results/sensors_v4/diagnostics.json');selected=[]
 for arm in ['gn','yaw','pca','hn']:
  sub=[r for r in d['rotation_tests'] if r['model']==B.ARMS[arm][0]+'0'];assert len(sub)==4
  for r in sub:selected.append(dict(frontend=B.ARMS[arm][2],angle_degrees=round(np.degrees(r['angle_rad'])),windows=r['windows'],mean_discrepancy_mps=r['mean_error_mps'],maximum_discrepancy_mps=r['maximum_error_mps']))
  if arm in ['yaw','hn']:
   for agg,func in [('MIN',min),('MAX',max)]:
    value=func(r['mean_error_mps'] for r in sub)
    B.TOKENS[f'VEL_{arm.upper()}_{agg}']=f'{value:.2e}' if arm=='hn' else f'{value:.3f}'
 B.csvout('direct_velocity_discrepancies',selected)
 B.table('TABLE_S9',['ResNet18 frontend','Angle (degrees)','Mean discrepancy (m/s)','Maximum discrepancy (m/s)'],[[r['frontend'],r['angle_degrees'],B.sci(r['mean_discrepancy_mps']),B.sci(r['maximum_discrepancy_mps'])] for r in selected])
 B.SOURCES[str(report_path.relative_to(ROOT))]=hashlib.sha256(report_path.read_bytes()).hexdigest()
 B.main()
 # The trajectory figure uses replayed positions, never invented or rescaled paths.
 with np.load(report_path.parent/'trajectories.npz') as z:
  fig,axes=plt.subplots(1,3,figsize=(10.5,3.65),sharex=True,sharey=True)
  colors=['#0072B2','#D55E00','#009E73','#CC79A7'];styles=['-','--',':','-.']
  for ax,arm,title in zip(axes,['yaw','ns','hn'],['Yaw augmentation','HN without sign rule','HN']):
   gt=z['ground_truth_m'];ax.plot(gt[:,0],gt[:,1],color='black',lw=1.35,label='Reference',zorder=1)
   for i,k in enumerate([0,2,4,6]):
    xy=z[f'{arm}_{k}_position_m'];ax.plot(xy[:,0],xy[:,1],color=colors[i],ls=styles[i],lw=1.1,label=f'{45*k}°',alpha=.85)
   ax.set_title(title,fontsize=11);ax.set_xlabel('x (m)');ax.set_aspect('equal',adjustable='box')
  axes[0].set_ylabel('y (m)');handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5,bbox_to_anchor=(.5,-.01),frameon=False)
  fig.tight_layout(rect=(0,.08,1,1));B.savefig(fig,'fig_trajectory_example')
 B.csvout('trajectory_illustration_scores',replay['rows'])
 order_references()
 # Submission names follow displayed figure numbers rather than historical names.
 names={'fig1_pipeline':'figure_01_pipeline','fig_trajectory_example':'figure_02_trajectories',
        'fig2_rotation_range':'figure_03_rotation_range','fig3_accuracy_contrasts':'figure_04_accuracy',
        'fig5_temporal_events':'figure_05_frame_events','fig6_residual_structure':'figure_06_residual_structure',
        'fig4_elapsed_error':'figure_S01_elapsed_error'}
 for old,new in names.items():
  for ext in ['png','pdf','svg']:
   (B.FIG/(old+'.'+ext)).replace(B.FIG/(new+'.'+ext))
 for name in ['manuscript_en','supplementary_materials_en']:
  p=B.OUT/(name+'.md');s=p.read_text()
  for old,new in names.items():s=s.replace('figures/'+old+'.png','figures/'+new+'.png')
  p.write_text(s)
 (B.DATA/'figure_filename_map.json').write_text(json.dumps(names,indent=2)+'\n')
 reports=[STYLE.audit(B.OUT/(n+'.md')) for n in ['manuscript_en','supplementary_materials_en']]
 dest=ROOT/'verification/priority_revision_20260909/style_audit.json';dest.write_text(json.dumps(dict(reports=reports,manual_checks=['one assertion per sentence','at most one subordinate clause','no nested subordinate clauses']),indent=2)+'\n')
 print(json.dumps(reports,indent=2))
 if any(r['errors'] for r in reports):raise SystemExit(1)
if __name__=='__main__':main()
