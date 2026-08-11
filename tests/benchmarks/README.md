# CUDA-Graph-Capture Physics Benchmark

`physics_graph_capture.py` verifies the captured Simplicits step against the
normal path before reporting timings. It requires a CUDA-capable GPU.

Run it from the repository root:

```
python tests/benchmarks/physics_graph_capture.py
```

Useful options:

```
python tests/benchmarks/physics_graph_capture.py --only-correctness
python tests/benchmarks/physics_graph_capture.py --stacking-cubes --steps 200
```

The benchmark reports timings only when the host and captured trajectories
agree within its correctness tolerance.
