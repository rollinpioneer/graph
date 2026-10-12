from common import *
source=(R/'geometry.py').read_text();exec(source[source.index('def rot('):source.index('rows=[]')])
a=np.zeros((3,13));b=np.zeros((3,13));a[:,6]=1;b[:,6]=1;b[:,0]=[.05,.01,.03];b[2,1]=.03
out=sat(a,np.full(3,.035),b,np.full(3,.035));expected=np.array([.015,-.025,-.005]);err=float(np.abs(out-expected).max());assert err<1e-12
(R/'receipts/geometry_selfcheck.json').write_text(json.dumps({'known_axis_aligned_cases':['15mm separated','25mm overlap','diagonal centers>40mm but5mm geometry overlap'],'SAT_gap':out.tolist(),'expected':expected.tolist(),'maxabs_error':err,'interpretation':'SAT overlap/separation, not physical contact'},indent=2));print('GEOMETRY_SYNTHETIC_CHECK_PASS',err)
