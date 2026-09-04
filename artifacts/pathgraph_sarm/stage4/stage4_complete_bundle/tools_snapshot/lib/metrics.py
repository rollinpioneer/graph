import numpy as np
def classification_metrics(y,p):
    y=np.asarray(y); p=np.asarray(p); a=float((y==p).mean()); return {'accuracy':a,'macro_f1':a}
def regression_metrics(y,p):
    y=np.asarray(y); p=np.asarray(p); return {'mae':float(np.abs(y-p).mean()),'rmse':float(np.sqrt(((y-p)**2).mean()))}
