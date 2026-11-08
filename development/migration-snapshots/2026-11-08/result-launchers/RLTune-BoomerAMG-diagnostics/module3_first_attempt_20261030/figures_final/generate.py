"""Figures 6 and 7 from completed first-attempt scans/tests, preserving approved SVG layouts."""
from pathlib import Path
import base64,copy,hashlib,io,json,re,shutil,subprocess,sys,xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from matplotlib import _path
from matplotlib.font_manager import get_font
from matplotlib.ft2font import LoadFlags
from matplotlib.path import Path as MPath
from matplotlib.transforms import IdentityTransform

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
PAPER=Path('/Users/erisehe/Documents/GitHub/SISC-Overleaf')
sys.path.insert(0,str(ROOT/'source'))
from experiments.paper_final.common.figure_style import WEIGHT_CMAP
NS='http://www.w3.org/2000/svg';XL='http://www.w3.org/1999/xlink'
T=lambda name:f'{{{NS}}}{name}'
ET.register_namespace('',NS);ET.register_namespace('xlink',XL)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parse=lambda p:ET.parse(p,parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
report={'metric':'first-attempt native solve time','scan_repetitions':3,'independent_test_repetitions':3,'recovery_executed':False,'paper_files_modified':False,'families':{}}
data={}
for family in ('diffusion','variable_flow'):
 run=ROOT/family/'run'
 assert json.loads((run/'status.json').read_text())['status']=='complete'
 summary=json.loads((run/'analysis/summary.json').read_text())
 protocol=summary['protocol'];assert protocol['timing_protocol']=='first_attempt_native_solve_v1'
 assert protocol['scan_repetitions']==protocol['repetitions_per_case']==3
 assert protocol['test_cases']==100 and protocol['training_seeds']==[1,2,3,4,5,6]
 assert not protocol['recovery_enabled'] and protocol['failed_attempt_times_included']
 assert summary['audit']['exact_coverage'] and summary['audit']['recovered_evaluations']==0
 manifest=json.loads((run/'paper_figures/manifest.json').read_text());assert manifest['checks']['passed']
 array=run/'paper_figures/plot_data.npz';assert sha(array)==manifest['output_sha256']['plot_data.npz']
 with np.load(array,allow_pickle=False) as z:
  assert z['cost_fields'][0]=='native_solve_sec'
  costs=z['costs_sec'][...,0].copy();weights=z['first_repetition_primary_weights'].copy()
  case_ids=z['case_ids'].copy();success=z['all_repetitions_success'].copy();methods=z['methods'].tolist()
  assert costs.shape==(6,5,100) and weights.shape==(6,5,100,50)
  totals=costs.sum(axis=2);gains=100*(1-totals[:,1:]/totals[:,:1])
  np.testing.assert_allclose(gains,z['native_reductions_pct'][:,1:],rtol=1e-12,atol=1e-10)
 for i,seed in enumerate(protocol['training_seeds']):
  for j,method in enumerate(methods):
   row=next(r for r in summary['per_seed'] if r['seed']==seed and r['method']==method)
   np.testing.assert_allclose(totals[i,j],row['native_solve_sec'],rtol=1e-10,atol=1e-10)
   assert int((~success[i,j]).sum())==row['failed_cases']
 data[family]=dict(gains=gains,weights=weights,cases=case_ids,methods=methods)
 report['families'][family]=dict(run=str(run),plot_data_sha256=sha(array),means=gains.mean(0).tolist(),sample_sd=gains.std(0,ddof=1).tolist(),first_attempt_failures={method:int((~success[:,j]).sum()) for j,method in enumerate(methods)})
for name in ('matched_policy_savings','matched_policy_weights'):
 for suffix in ('svg','png'):
  template=OUT/'templates'/f'{name}.{suffix}'
  if not template.exists():shutil.copy2(PAPER/'figures'/f'{name}.{suffix}',template)
 report.setdefault('template_sha256',{})[name]=sha(OUT/'templates'/f'{name}.svg')

# Figure 6: update data marks and numeric annotations in the original vector template.
tree=parse(OUT/'templates/matched_policy_savings.svg');root=tree.getroot()
byid={e.get('id'):e for e in root.iter() if e.get('id')};defs=root.find(T('defs'))
char_ids={}
for text in root.iter():
 label=text.get('aria-label','')
 if re.fullmatch(r'\d+\.\d ± \d+\.\d',label):
  uses=list(text.iter(T('use')));chars=label.replace(' ','');assert len(uses)==len(chars)
  for char,use in zip(chars,uses):char_ids[char]=use.get(f'{{{XL}}}href')

def add_glyph(char,fontname,glyphname):
 font=get_font(subprocess.check_output(['/Library/TeX/texbin/kpsewhich',fontname+'.pfb'],text=True).strip())
 font.set_size(100,72);index=font.get_name_index(glyphname);assert index
 font.load_glyph(index,flags=LoadFlags.NO_HINTING);vertices,codes=font.get_path()
 outline=_path.convert_to_string(MPath(vertices,codes),IdentityTransform(),None,False,None,6,[b'M',b'L',b'Q',b'C',b'z'],False).decode()
 id_='MLM-new-results-'+glyphname;ET.SubElement(defs,T('path'),id=id_,d=outline);char_ids[char]='#'+id_
if '8' not in char_ids:add_glyph('8','mlmr10','eight')
add_glyph('-','mlmsy10','minus')
# Retain the template's exact nominal glyph size, tabular digit advance and spacing.
example=next(e for e in byid['text_5'].iter() if e.get('aria-label'))
uses=list(example.iter(T('use')))
xy=[list(map(float,re.search(r'translate\(([-\d.]+) ([-\d.]+)\)',u.get('transform')).groups())) for u in uses]
scale=float(re.search(r'scale\(([-\d.]+)\)',uses[0].get('transform'))[1])
digit=xy[1][0]-xy[0][0];dot=xy[3][0]-xy[2][0]
space=xy[4][0]-xy[3][0]-digit;plusminus=xy[5][0]-xy[4][0]-space
advance={**{str(i):digit for i in range(10)},'.':dot,' ':space,'±':plusminus,'-':plusminus}
left,right=96.003885,495.002;x=lambda v:left+(float(v)+25)/80*(right-left)
annotations=[]
for a,family in enumerate(('diffusion','variable_flow')):
 gains=data[family]['gains'];means=gains.mean(0);sds=gains.std(0,ddof=1)
 assert gains.min()>-25 and gains.max()<55 and min(means-sds)>-25 and max(means+sds)<55
 for j in range(4):
  n=a*4+j+1;values=gains[:,j];mu,sd=means[j],sds[j]
  collection=byid[f'PathCollection_{n}'];points=list(collection.iter(T('use')));assert len(points)==6
  for marker,value in zip(points,values):marker.set('x',f'{x(value):.6f}')
  y=float(points[0].get('y'))
  path=next(byid[f'LineCollection_{n}'].iter(T('path')))
  path.set('d',f'M {x(mu-sd):.6f} {y:.6f} L {x(mu+sd):.6f} {y:.6f}')
  base_cap=14 if a==0 else 39;base_mean=22 if a==0 else 47
  for id_,value in ((base_cap+2*j,mu-sd),(base_cap+2*j+1,mu+sd),(base_mean+j,mu)):
   next(byid[f'line2d_{id_}'].iter(T('use'))).set('x',f'{x(value):.6f}')
  tid=(5 if a==0 else 21)+j;text=next(e for e in byid[f'text_{tid}'].iter() if e.get('aria-label'))
  label=f'{mu:.1f} ± {sd:.1f}';container=next(iter(text));assert container.tag==T('g')
  for child in list(container):container.remove(child)
  pos=0.
  for char in label:
   if char!=' ':
    ET.SubElement(container,T('use'),{f'{{{XL}}}href':char_ids[char],'transform':f'translate({pos:.8f} 0) scale({scale:.10f})'})
   pos+=advance[char]
  text.set('aria-label',label)
  text.set('transform',re.sub(r'^translate\([^)]*\)',f'translate({x(max(values.max(),mu+sd))+7:.6f} {y+2.2075:.6f})',text.get('transform')))
  np.testing.assert_allclose([(float(p.get('x'))-left)/(right-left)*80-25 for p in points],values,atol=2e-7,rtol=0)
  annotations.append(dict(family=family,row=j,text=label))
svg6=OUT/'figure6.svg';tree.write(svg6,encoding='utf-8',xml_declaration=True)
report['figure6']=dict(common_x_limits=[-25,55],upper_axis_line_and_ticks=True,upper_tick_numbers=False,point_marker='open circle',mean_marker='diamond',dispersion='sample SD across six seeds',annotations=annotations,fonts_and_layout='Original MLModern glyphs and SVG geometry retained')

# Figure 7: only replace the four raster heatmaps inside the approved vector layout.
tree=parse(OUT/'templates/matched_policy_weights.svg');root=tree.getroot();before=copy.deepcopy(root)
z=data['variable_flow'];seed_index=0
order=np.lexsort((z['cases'],-np.isfinite(z['weights'][seed_index,0]).sum(axis=1)))
indices=[z['methods'].index(m) for m in ('reference','oracle','periodic','rl')]
shown=z['weights'][seed_index,indices][:,order]
assert shown.shape==(4,100,50)
original_hrefs={};image_ids=[]
for j in range(4):
 ax=next(e for e in root.iter() if e.get('id')==f'axes_{j+1}')
 image=next(ax.iter(T('image')));key=f'{{{XL}}}href';original_hrefs[image.get('id')]=image.get(key);image_ids.append(image.get('id'))
 old=Image.open(io.BytesIO(base64.b64decode(image.get(key).split(',')[1])))
 rgba=WEIGHT_CMAP(np.ma.masked_invalid((shown[j].T-1)/2),bytes=True)
 scaled=Image.fromarray(rgba).resize(old.size,Image.Resampling.NEAREST)
 # SVG's y-flip requires the PNG itself to be stored bottom-to-top.
 buffer=io.BytesIO();scaled.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(buffer,format='PNG')
 image.set(key,'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode())
 # Every cell center reproduces the exact recorded weight/color or white termination mask.
 sx=((np.arange(100)+.5)*old.width/100).astype(int);sy=((np.arange(50)+.5)*old.height/50).astype(int)
 np.testing.assert_array_equal(np.asarray(scaled)[sy[:,None],sx[None,:]],rgba)
recovered=copy.deepcopy(root)
for element in recovered.iter():
 if element.get('id') in original_hrefs:element.set(f'{{{XL}}}href',original_hrefs[element.get('id')])
assert ET.tostring(recovered)==ET.tostring(before),'Unrelated Figure 7 layout/text changed'
svg7=OUT/'figure7.svg';tree.write(svg7,encoding='utf-8',xml_declaration=True)
np.savez_compressed(OUT/'figure7_displayed_data.npz',weights=shown,case_ids=z['cases'][order])
report['figure7']=dict(family='variable_flow',seed=1,repetition=0,ordering='Decreasing fixed-w=1 first-attempt cycle count; ties by case ID',images_updated=image_ids,all_other_svg_elements_unchanged=True,all_cell_center_colors_verified=True,displayed_cycles=[0,49],per_problem_label=True)
for target,template in ((svg6,'matched_policy_savings'),(svg7,'matched_policy_weights')):
 width=Image.open(OUT/'templates'/f'{template}.png').width
 subprocess.run(['/opt/homebrew/bin/inkscape',str(target),'--export-area-page',f'--export-width={width}','--export-background=white','--export-background-opacity=1',f'--export-filename={target.with_suffix(".png")}'],check=True,capture_output=True)
 report.setdefault('outputs',{})[target.stem]=dict(svg_sha256=sha(target),png_sha256=sha(target.with_suffix('.png')),png_dimensions=list(Image.open(target.with_suffix('.png')).size))
(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
