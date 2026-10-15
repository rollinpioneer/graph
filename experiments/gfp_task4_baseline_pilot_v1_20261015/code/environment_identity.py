import importlib.metadata as m,json,sys
print(json.dumps(dict(python=sys.version,packages={k:m.version(k) for k in ['jax','jaxlib','flax','optax','numpy','mujoco','gymnasium','ogbench']})))
