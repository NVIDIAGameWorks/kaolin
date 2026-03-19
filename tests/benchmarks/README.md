# Physics Benchmarking Instructions

The script benchmarks Kaolin's physics simulator on two different objects, a unit cube and a complex fox geometry.

## Benchmark without Nsight
Running without nsight will enable the polyscope visualizer. To run the `physics_benchmarks.py` file without Nsight Systems, use the following command:

```
python physics_benchmarks.py --<optional args>
```

## Benchmark with Nsight

Running with Nsight will disable the polyscope visualizer. To run the `physics_benchmarks.py` file with Nsight Systems, use the following command:

```
nsys profile -o {output_filename} --stats=true physics_benchmarks.py --<optional args>
```

### Optional args 

Run benchmarking script with optional args using the following template:

```
python physics_benchmarks.py --num_frames 100 --num_objects 2
```

Here are the optional args. 

Note: Using nsys to profile the code will enforce `headless=True` regardless of the value passed in.

| Argument                        | Type     | Default     | Description                        |
|----------------------------------|----------|-------------|------------------------------------|
| `--object`                       | string   | "fox"       | Either "fox" or "box"              |
| `--screenshot`                   | bool     | False       | Save a screenshot after simulation |
| `--write_mesh`                   | bool     | False       | Write mesh output to file          |
| `--timestep`                     | float    | 0.01        | Simulation timestep                |
| `--max_newton_steps`             | int      | 5           | Max Newton solver steps            |
| `--max_ls_steps`                 | int      | 20          | Max line search steps              |
| `--newton_hessian_regularizer`   | float    | 1e-4        | Regularization for Hessian         |
| `--direct_solve`                 | bool     | True        | Use direct solver                  |
| `--device`                       | str      | "cuda"      | Device to use ("cuda" or "cpu")    |
| `--dtype`                        | str      | "float32"   | Data type ("float32", etc.)        |
| `--headless`                     | bool     | False       | Run without visualization          |
| `--num_frames`                   | int      | 50          | Number of simulation frames        |
| `--num_objects`                  | int      | 1           | Number of objects in the scene     |
| `--num_samples`                  | int      | 1000        | Number of samples per object in the scene     |
| `--num_handles`                  | int      | 0           | Number of handles per object in the scene, writes model to file if it doesn't exist|