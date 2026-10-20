from common import *
import inspect,subprocess
import geometry_core as ref
assert (R/'results/C4_result_freeze.json').exists()
assert sha(R/'code/geometry_core.py')==json.load(open(R/'method_lock.json'))['geometry_core_sha256']
cpp=r'''#include <cmath>
#include <cstdint>
extern "C" int64_t masks(const double* p,int64_t n,double w,double h,double d,int64_t* counts,int64_t* ids){
 int64_t ni=0;for(int64_t i=0;i<n;++i){double x=p[3*i],y=p[3*i+1],z=p[3*i+2];bool ht=std::abs(z)<h/2.,f=(x>d-.06)&&(x<d),side=std::abs(y)<w/2.+.01;
 counts[0]+=ht&&f&&(y>-w/2.-.01)&&(y<-w/2.);
 counts[1]+=ht&&f&&(y>w/2.)&&(y<w/2.+.01);
 counts[2]+=ht&&side&&(x<=d-.06)&&(x>d-.07);
 counts[3]+=ht&&side&&(x<=d-.07)&&(x>d-.12);
 if(ht&&f&&(std::abs(y)<=w/2.)){counts[4]++;ids[ni++]=i;}
 }return ni;
}
'''
(R/'code/masks.cpp').write_text(cpp);t=time.time();subprocess.run(['g++','-O3','-std=c++11','-shared','-fPIC','-fno-fast-math','-ffp-contract=off',str(R/'code/masks.cpp'),'-o',str(R/'code/masks.so')],check=True)
prefix='''from geometry_core import *
import ctypes
_lib=ctypes.CDLL(str(R/'code/masks.so'))
_lib.masks.argtypes=[ctypes.c_void_p,ctypes.c_int64,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_void_p,ctypes.c_void_p]
_lib.masks.restype=ctypes.c_int64
def cov_exact(points):
 X=np.array(points.T,ndmin=2,dtype=np.float64)
 avg=np.average(X,axis=1)
 X-=avg[:,None]
 c=np.dot(X,X.T.conj())
 c*=np.true_divide(1,len(points)-1)
 return c.squeeze()

class CachedTree:
 def __init__(self,tree,poses):
  centers={p[13:16].tobytes():p[13:16] for p in poses}
  values=list(centers.values());queries=tree.query_ball_point(np.asarray(values),.16,return_sorted=False,workers=1)
  self.lookup={k:np.asarray(q,dtype=np.int64) for k,q in zip(centers,queries)}
 def query_ball_point(self,c,r):
  assert r==.16
  return self.lookup[c.tobytes()]

'''
v=inspect.getsource(ref.visible).replace('def visible(', 'def visible_fast(')
v=v.replace('u,inv=np.unique(grid,axis=0,return_inverse=True)','''lo_grid=grid.min(0);span=grid.max(0)-lo_grid+1
 assert int(span[0])*int(span[1])*int(span[2])<np.iinfo(np.int64).max
 shifted=grid-lo_grid;key=(shifted[:,0]*span[1]+shifted[:,1])*span[2]+shifted[:,2]
 u,inv=np.unique(key,return_inverse=True)''')
g=inspect.getsource(ref.geom);start=g.index(' height=np.abs(z)');end=g.index(' out=list(',start)
mask=''' counts=np.zeros(5,np.int64);indices=np.empty(len(pts),np.int64)
 ni=_lib.masks(pts.ctypes.data,len(pts),float(w),float(h),float(d),counts.ctypes.data,indices.ctypes.data)
 counts=counts.astype(float);inner=pts[indices[:ni]]
'''
g2=(g[:start]+mask+g[end:]).replace('def geom(','def geom_mask(')
g3=g2.replace('def geom_mask(','def geom_cov(').replace('np.cov(band.T)','cov_exact(band)').replace('np.cov(inner.T)','cov_exact(inner)')
suffix='''
def extract(scene,frame,pred,variant):
 cloud,identity=visible(scene,frame,pred) if variant=='baseline' else visible_fast(scene,frame,pred)
 tree=cKDTree(cloud)
 if variant!='baseline':tree=CachedTree(tree,pred)
 fn={'baseline':geom,'index_query':geom,'mask':geom_mask,'cov':geom_cov}[variant]
 raw=np.array([fn(cloud,tree,p) for p in pred],np.float32)
 return raw,cloud,identity
'''
(R/'code/geometry_fast.py').write_text(prefix+v+'\n'+g2+'\n'+g3+suffix)
write(R/'results/fast_build.json',dict(status='PASS',compile_seconds=time.time()-t,compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],flags=['-O3','-std=c++11','-shared','-fPIC','-fno-fast-math','-ffp-contract=off'],source_sha256=sha(R/'code/masks.cpp'),binary_sha256=sha(R/'code/masks.so'),geometry_fast_sha256=sha(R/'code/geometry_fast.py'),no_environment_installation=True,C4_result_freeze_sha256=sha(R/'results/C4_result_freeze.json')))
print('FAST_BUILD_COMPLETE',flush=True)
