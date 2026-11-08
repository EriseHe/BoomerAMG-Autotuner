"""Deterministic irregular tetrahedral mesh for the Example 16 heat pilot."""
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay

root=Path(__file__).resolve().parents[2]
rng=np.random.default_rng(20261104)
shape=(19,6,5);extent=np.array([8.,1.,1.]);spacing=extent/(np.array(shape)-1)
points=np.stack(np.meshgrid(*[np.linspace(0,extent[d],shape[d]) for d in range(3)],
                           indexing='ij'),axis=-1).reshape(-1,3)
for d in range(3):
    free=(points[:,d]>0)&(points[:,d]<extent[d])
    points[free,d]+=rng.uniform(-.17,.17,free.sum())*spacing[d]
tri=Delaunay(points);tets=tri.simplices
det=np.linalg.det(points[tets[:,1:]]-points[tets[:,0,None]])
assert np.all(abs(det)>1e-13)
assert np.isclose(abs(det).sum()/6,np.prod(extent))
boundary=[np.delete(t,o) for t,neigh in zip(tets,tri.neighbors)
          for o in np.flatnonzero(neigh==-1)]
path=root/'results/application_default_100/heat.mesh'
with path.open('w') as f:
    f.write(f'MFEM mesh v1.0\n\ndimension\n3\n\nelements\n{len(tets)}\n')
    for t in tets:f.write('1 4 '+' '.join(map(str,t))+'\n')
    f.write(f'\nboundary\n{len(boundary)}\n')
    for t in boundary:f.write('1 2 '+' '.join(map(str,t))+'\n')
    f.write(f'\nvertices\n{len(points)}\n3\n')
    for v in points:f.write(' '.join(map(str,v))+'\n')
print(path)
