# residualforge

[Live demo](https://sushxnthd.github.io/residualforge/) | [Architecture](docs/architecture.md) | [Benchmark](benchmarks/results.json)

`residualforge` audits whether a predicted trajectory actually satisfies the PDE it claims to approximate. It uses finite differences on a time-by-space grid and reports residual magnitude plus the worst residual location.

```bash
pip install -e .
residualforge solution.npy --equation heat --coef 0.2 --dx 0.01 --dt 0.002
```

## Included equations

- 1D heat equation: `u_t = alpha u_xx`
- 1D viscous Burgers equation: `u_t + u u_x = nu u_xx`

The package deliberately starts with equations whose residuals can be checked without an automatic differentiation framework. That makes it useful for exported predictions from any model family.

## Benchmark

`python benchmarks/run.py` evaluates the analytic heat-equation solution and then injects a local trajectory defect. The benchmark records how strongly the residual RMSE increases after the defect.

## Scope

Finite-difference residuals depend on grid resolution and boundary handling. Residualforge evaluates the interior grid and should be treated as a diagnostic, not as a proof of numerical convergence.

## Roadmap

- boundary-condition checks
- 2D diffusion and wave equations
- irregular-grid adapters
- residual heatmap export

MIT licensed.
