import json
from pathlib import Path
import numpy as np
from residualforge import heat_residual, summarize

alpha = 0.2
nt, nx = 121, 161
t = np.linspace(0, 0.4, nt); x = np.linspace(0, 1, nx)
u = np.exp(-alpha*np.pi**2*t[:,None]) * np.sin(np.pi*x[None,:])
dx=x[1]-x[0]; dt=t[1]-t[0]
clean=summarize(heat_residual(u,alpha,dx,dt)).rmse
bad=u.copy(); bad[48:56,72:80] += 0.12
corrupt=summarize(heat_residual(bad,alpha,dx,dt)).rmse
out={"clean_rmse":clean,"corrupt_rmse":corrupt,"amplification":corrupt/clean}
Path(__file__).with_name("results.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
