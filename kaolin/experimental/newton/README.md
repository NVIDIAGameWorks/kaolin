# Newton Integration (Experimental)

> **Warning: This is an experimental module. The API is constantly in flux and subject to breaking changes without notice.**

Bridges Kaolin's [Simplicits](https://kaolin.readthedocs.io/en/latest/notes/simplicits.html) soft-body solver with the [NVIDIA Newton](https://newton-physics.github.io/newton/stable/guide/overview.html) physics engine, enabling soft deformable objects to coexist and interact with rigid bodies, articulated robots, and MPM granular materials inside a single Newton simulation loop.

---

## Requirements

| Package | Version | Notes |
|---|---|---|
| `newton` | Release 1.0 | [NVIDIA Newton](https://newton-physics.github.io/newton/stable/guide/overview.html) physics engine — see [Installing Newton](https://newton-physics.github.io/newton/stable/guide/installation.html) below |
| `warp-lang` | ≥ 1.10 | Already installed as a Kaolin dependency |
| Python | ≥ 3.10 | Required by Newton |
---

## Installing Newton

**Supported version: Newton release tag `1.0` only.**


### From source

See the [Newton installation guide](https://newton-physics.github.io/newton/1.0.0/guide/installation.html) for full details. Local dev installs work the same way — point `pip install -e ".[examples]"` at your local checkout.

> `warp-lang` is already installed as a Kaolin dependency and does not need to be installed separately.

### From PyPI

```bash
pip install "newton[examples]"
```

The `[examples]` extra installs optional dependencies (MuJoCo, renderers, etc.) used by the coupling example scripts.

---

## Coupling Solvers Overview

Newton supports **multi-solver setups** where multiple solvers step in sequence each frame. This is how Simplicits integrates: the `SimplicitsSolver` runs as one solver in a chain alongside Newton's built-in solvers. See the [Newton documentation](https://newton-physics.github.io/newton/stable/guide/overview.html) for the native solver and model APIs.

### Shared model (most cases)

`SimplicitsModel` inherits from [newton.Model](https://newton-physics.github.io/newton/stable/api/_generated/newton.Model.html#newton.Model), so a single model instance can be passed to two separate solvers. For example, soft↔rigid coupling uses:

```
SimplicitsSolver  ──┐
                    ├── share one SimplicitsModel
SolverSemiImplicit ─┘
```

Both solvers read and write into the same particle/body state arrays each frame — `SimplicitsSolver` handles the Simplicits DOF range, [SolverSemiImplicit](https://newton-physics.github.io/newton/stable/api/_generated/newton.solvers.SolverSemiImplicit.html) handles the rigid bodies.

### Multiple models (particle-particle coupling)

MPM coupling examples use **separate models** because each particle solver (MPM, Simplicits) owns its own particle array. The solvers exchange forces through the contact energy rather than shared state.

### Key classes

| Class | Role |
|---|---|
| `SimplicitsModelBuilder` | Extends [newton.ModelBuilder](https://newton-physics.github.io/newton/stable/api/_generated/newton.ModelBuilder.html#newton.ModelBuilder); registers soft-body objects and collision shapes |
| `SimplicitsModel` | Extends [newton.Model](https://newton-physics.github.io/newton/stable/api/_generated/newton.Model.html#newton.Model); holds `SimplicitsScene` + particle index range |
| `SimplicitsState` | Extends [newton.State](https://newton-physics.github.io/newton/stable/api/_generated/newton.State.html); carries `sim_z`, `sim_z_dot`, `sim_z_prev` |
| `SimplicitsSolver` | Extends [newton.SolverBase](https://newton-physics.github.io/newton/stable/api/_generated/newton.solvers.SolverBase.html); runs one Simplicits Newton-Raphson step per frame |

### How it works

1. **Builder phase** (`SimplicitsModelBuilder.finalize()`): registers Simplicits quadrature points as Newton particles with so broadphase contact detection still sees them and can return contacts for each particle.
2. **Per-frame step** (`SimplicitsSolver.step()`): copies `sim_z`/`sim_z_dot` into the live scene, runs `SimplicitsScene.run_sim_step()`, writes results back, and reconstructs full particle positions.
3. **Contact energy** (`SimplicitsParticleNewtonShapeSoftContact`): IPC-style penalty energy with friction and optional restitution, called from inside the Simplicits Newton-Raphson inner loop.

---

## Examples

End-to-end newton coupling example scripts in `examples/tutorial/physics/`:

| Script | Description |
|---|---|
| `newton_rigidbody_coupling.ipynb` | Soft body colliding with a falling rigid box |
| `newton_mpm_coupling_oneway.ipynb` | One-way MPM coupling: soft body as static obstacle for MPM sand |
| `newton_franka_coupling.ipynb` | Soft body manipulated by a Franka robot arm |

---

## Quick start

```python
import torch
import warp as wp
from kaolin.experimental.newton.builder import SimplicitsModelBuilder
from kaolin.experimental.newton.solver import SimplicitsSolver
from kaolin.physics.simplicits import SimplicitsObject

# 1. Create a Simplicits soft-body object (see kaolin.physics.simplicits docs)
sim_obj = SimplicitsObject.create_rigid(pts, yms, prs, rhos, approx_volume)

# 2. Build the coupled model
builder = SimplicitsModelBuilder(up_axis="y", gravity=-9.81)
builder.add_object(sim_obj, num_qp=1024)
builder.add_shape_plane(plane=(*builder.up_vector, -1.0), width=0.0, length=0.0)

model = builder.finalize(device="cuda")

# 3. Create solver and states
solver = SimplicitsSolver(model)
state_in  = model.state()
state_out = model.state()

# 4. Simulation loop
dt = 0.033
for frame in range(300):
    contacts = model.collide(state_in)
    solver.step(state_in, state_out, control=None, contacts=contacts, dt=dt)
    state_in, state_out = state_out, state_in
```
