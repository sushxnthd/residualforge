import argparse, json
import numpy as np
from .core import heat_residual, burgers_residual, summarize

def main():
    p=argparse.ArgumentParser(); p.add_argument("file"); p.add_argument("--equation",choices=["heat","burgers"],default="heat"); p.add_argument("--coef",type=float,required=True); p.add_argument("--dx",type=float,required=True); p.add_argument("--dt",type=float,required=True)
    a=p.parse_args(); u=np.load(a.file)
    r=heat_residual(u,a.coef,a.dx,a.dt) if a.equation=="heat" else burgers_residual(u,a.coef,a.dx,a.dt)
    print(json.dumps(summarize(r).__dict__,indent=2))

if __name__ == "__main__":
    main()
