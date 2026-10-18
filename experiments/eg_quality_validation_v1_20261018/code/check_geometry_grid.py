from validate import Geometry,E,N,np,json,time
c=Geometry(8);c.frame(0)
z=np.load(E/'results/subset/dev/scene_0008/evaluated/0000.npz');owners=c.owners(z['grasps']);errs=[]
for i in [0,10,25,49]:
 g=z['grasps'][i];grid,w,ex,m=c.variants(g,int(owners[i]));r=grid[:,4:13].reshape(48,3,3)
 err=float(np.max(np.abs(np.einsum('nji,njk->nik',r,r)-np.eye(3))));det=float(np.max(np.abs(np.linalg.det(r)-1)))
 assert err<2e-6 and det<2e-6
 assert np.array_equal(grid[:,13:16],np.tile(g[13:16],(48,1)))
 assert np.allclose(r[:,:,0],g[4:13].reshape(3,3)[:,0],rtol=0,atol=1e-12)
 assert len(np.unique(grid[:,3]))==4 and np.all((grid[:,1]>=0)&(grid[:,1]<=.1))
 assert np.allclose(np.unique(grid[:,3]),[.01,.02,.03,.04],atol=1e-8)
 errs.append(dict(rank=i,max_orthogonality_error=err,max_determinant_error=det))
(N/'geometry_grid_check.json').write_text(json.dumps(dict(status='PASS',rows=errs,checks=['fixed center','fixed decoded approach','four original depths','12 angle cells','official width limits','orthonormal right-handed rotations']),indent=2));print('GRID_PASS',errs)
