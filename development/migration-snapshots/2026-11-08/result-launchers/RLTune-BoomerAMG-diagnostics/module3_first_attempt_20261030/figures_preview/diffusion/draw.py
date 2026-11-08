"""Diffusion-only Figure 6 preview in the accepted SVG layout; no measured data change."""
from pathlib import Path
import copy,hashlib,json,re,subprocess,xml.etree.ElementTree as ET
import numpy as np
from matplotlib import _path
from matplotlib.font_manager import get_font
from matplotlib.ft2font import LoadFlags
from matplotlib.path import Path as MPath
from matplotlib.transforms import IdentityTransform

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
PAPER=Path('/Users/erisehe/Documents/GitHub/SISC-Overleaf')
SVG='http://www.w3.org/2000/svg';XL='http://www.w3.org/1999/xlink'
T=lambda n:f'{{{SVG}}}{n}'
ET.register_namespace('',SVG);ET.register_namespace('xlink',XL)
source=PAPER/'figures/matched_policy_savings.svg'
tree=ET.parse(source,parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)));root=tree.getroot()
ids=lambda:{e.get('id'):e for e in root.iter() if e.get('id')}
# Move definitions out of panels that will be removed, retaining all glyphs/markers.
defs=root.find(T('defs'))
for nested in list(root.iter(T('defs'))):
 if nested is defs:continue
 for element in list(nested):nested.remove(element);defs.append(element)
byid=ids();axes1=byid['axes_1'];axes2=byid['axes_2']
char_ids={}
for text in root.iter():
 label=text.get('aria-label','')
 if re.fullmatch(r'\d+\.\d ± \d+\.\d',label):
  uses=list(text.iter(T('use')));chars=label.replace(' ','');assert len(uses)==len(chars)
  for char,use in zip(chars,uses):char_ids[char]=use.get(f'{{{XL}}}href')
# Eight is the only new annotation glyph required by the updated diffusion SDs.
if '8' not in char_ids:
 font=get_font('/Library/TeX/texbin/../texmf-dist/fonts/type1/public/mlmodern/mlmr10.pfb') if False else get_font(subprocess.check_output(['/Library/TeX/texbin/kpsewhich','mlmr10.pfb'],text=True).strip())
 font.set_size(100,72);font.load_glyph(font.get_name_index('eight'),flags=LoadFlags.NO_HINTING)
 vertices,codes=font.get_path();d=_path.convert_to_string(MPath(vertices,codes),IdentityTransform(),None,False,None,6,[b'M',b'L',b'Q',b'C',b'z'],False).decode()
 ET.SubElement(defs,T('path'),id='MLM-diffusion-eight',d=d);char_ids['8']='#MLM-diffusion-eight'
with np.load(ROOT/'diffusion/run/paper_figures/plot_data.npz') as data:
 costs=data['costs_sec'][...,0];seeds=data['seeds'].tolist()
 assert seeds==[1,2,3,4,5,6]
 gains=100*(1-costs[:,1:].sum(axis=2)/costs[:,:1].sum(axis=2))
 np.testing.assert_allclose(gains,data['native_reductions_pct'][:,1:],rtol=1e-12,atol=1e-10)
left,right=96.003885,495.002
x=lambda value:left+(float(value)+25)/80*(right-left)
for j in range(4):
 values=gains[:,j];mu=values.mean();sd=values.std(ddof=1)
 collection=byid[f'PathCollection_{j+1}'];uses=list(collection.iter(T('use')));assert len(uses)==6
 for use,value in zip(uses,values):use.set('x',f'{x(value):.6f}')
 path=next(byid[f'LineCollection_{j+1}'].iter(T('path')));y=float(uses[0].get('y'))
 path.set('d',f'M {x(mu-sd):.6f} {y:.6f} L {x(mu+sd):.6f} {y:.6f}')
 for id_,value in ((14+2*j,mu-sd),(15+2*j,mu+sd),(22+j,mu)):
  marker=next(byid[f'line2d_{id_}'].iter(T('use')));marker.set('x',f'{x(value):.6f}')
 text=next(e for e in byid[f'text_{5+j}'].iter() if e.get('aria-label'))
 label=f'{mu:.1f} ± {sd:.1f}';chars=label.replace(' ','');glyphs=list(text.iter(T('use')));assert len(chars)==len(glyphs)
 for use,char in zip(glyphs,chars):use.set(f'{{{XL}}}href',char_ids[char])
 text.set('aria-label',label)
 text.set('transform',re.sub(r'^translate\([^)]*\)',f'translate({x(max(values.max(),mu+sd))+7:.6f} {y+2.2075:.6f})',text.get('transform')))
# Reuse the bottom panel's numbered x axis at the top panel's baseline.
delta=64.239356-137.058877
old_axis=byid['matplotlib.axis_1'];new_axis=copy.deepcopy(byid['matplotlib.axis_3'])
new_axis.set('transform',f'translate(0 {delta:.6f})')
index=list(axes1).index(old_axis);axes1.remove(old_axis);axes1.insert(index,new_axis)
parent={c:p for p in root.iter() for c in p}
parent[axes2].remove(axes2)
for element in list(root.iter()):
 label=element.get('aria-label','')
 if label.startswith('(b) Variable-flow'):
  group=element
  while parent[group].get('id')!='figure_1':group=parent[group]
  parent[group].remove(group)
 elif label.startswith('First-attempt native solve-time'):
  group=element
  while parent[group].get('id')!='figure_1':group=parent[group]
  group.set('transform',f'translate(0 {delta:.6f})')
view=list(map(float,root.get('viewBox').split()));view[3]+=delta
root.set('height',f'{view[3]:.8f}pt');root.set('viewBox',' '.join(f'{v:.8f}' for v in view))
svg=OUT/'figure6_diffusion.svg';tree.write(svg,encoding='utf-8',xml_declaration=True)
subprocess.run(['/opt/homebrew/bin/inkscape',str(svg),'--export-area-page','--export-width=1800','--export-background=white','--export-background-opacity=1',f'--export-filename={OUT/"figure6_diffusion.png"}'],check=True,capture_output=True)
report=dict(source_run=str(ROOT/'diffusion/run'),template=str(source),template_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),mean=gains.mean(axis=0).tolist(),sample_sd=gains.std(axis=0,ddof=1).tolist(),seeds=seeds,x_limits=[-25,55],font='MLModern; existing outlined glyphs retained',paper_files_modified=False)
(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
