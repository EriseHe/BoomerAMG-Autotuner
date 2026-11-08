from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parent
data=json.loads((root/'results.json').read_text())
rows=data['rows']
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(10.5,4.7))
methods=['default_setup_lstdq','bandit_default','bandit_lstdq']
labels=['LSTDQ','LinUCB','LinUCB–LSTDQ']
colors=['#3A9AB2','#A5C2A3','#F11B00']
hatches=['','///','xx']
x=np.arange(6);width=.25
for j,(method,label,color,hatch) in enumerate(zip(methods,labels,colors,hatches)):
    values=[next(r['total_reduction_pct'] for r in rows if r['seed']==seed and r['method']==method) for seed in range(1,7)]
    bars=ax.bar(x+(j-1)*width,values,width=.23,color=color,edgecolor='#444444',linewidth=.45,label=label,hatch=hatch,zorder=3)
    ax.bar_label(bars,labels=[f'{v:.1f}' for v in values],padding=3,fontsize=9)
ax.set_xticks(x,['1','2','3','4','5','6'])
ax.set_xlabel('Seed')
ax.set_ylabel('Total runtime reduction vs Default (%)')
ax.set_ylim(-5,66)
ax.set_yticks(np.arange(0,61,10))
ax.grid(axis='y',color='#dddddd',linewidth=.6,zorder=0)
ax.axhline(0,color='#333333',linewidth=.8)
ax.axvline(2.5,color='#999999',linestyle='--',linewidth=.7)
ax.text(1,64,'Older results',ha='center',va='top',fontsize=11)
ax.text(4,64,'Rerun',ha='center',va='top',fontsize=11)
fig.suptitle('Module 2: variable-flow online learning',fontsize=15,y=.98)
fig.text(.5,.895,'5,000 problems per seed; all attempts and measured learner overhead included',ha='center',fontsize=10)
fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.005),ncol=3,frameon=False)
fig.subplots_adjust(left=.10,right=.985,top=.85,bottom=.19)
for suffix in ('png','svg'):
 fig.savefig(root/f'six_seed_runtime_reduction.{suffix}',dpi=180,facecolor='white')
plt.close(fig)
