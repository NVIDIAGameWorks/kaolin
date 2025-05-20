# Physics Sim Overview

Let's say we have an object denoted by $n$ vertices flattened to $x \in R^{3n}$ and velocities $v \in R^{3n}$. 
Following newtons law $f(x) = f_{int}(x) + f_{ext} = Ma$, we must apply forces to the object to transform/deform it at each timestep $i$. The forces on an object are a sum of the internal forces $f_{int}$ and external forces $f_{ext}$

## Time Integrators

The way we apply forces and compute the deformation of the object through time is called a time-integration scheme. 
An explicit time integration scheme is one where the next timestep ($i+1$) is computed directly using the quantities from the previous timestep ($i$).
One example is explicit euler:

**Explicit Euler**

$x_{i+1} = x_i + \Delta t v_i$, 

$v_{i+1} = v_i + \Delta t a_i \rightarrow v_{i+1} = v_i + \Delta t M^{-1}(f_{int}(x_i) + f_{ext})$

There are other explicit time integration schemes as well (like Runge-Kutta (RK2, RK4) and more). *However*, the problem with explicit integrators is that they require very small timesteps and are inherently unstable for elastodynamics.

Therefore, in Simplicits, we use an implicit time-integrator where $x_{i+1}(v_{i+1}), v_{i+1}(x_{i+1})$. There are several options (such as Implicit Euler, BDF2, and more) from which we pick implicit euler:

**Implicit Euler**

$x_{i+1} = x_i + \Delta t v_{i+1}$, 

$v_{i+1} = v_i + \Delta t M^{-1}f(x_{i+1})$

This inter-dependence of variables is solved using a Newton's Method optimizer (2nd order method for faster convergence). We first rewrite the equations solely in terms of $x_{i+1}$

$x_{i+1} = x_i + \Delta t (v_i + \Delta t M^{-1}f(x_{i+1}))$, 

Now the *optimal* $x_{i+1}$ will ensure that 

$ 0 = x_{i+1} - x_i - \Delta t (v_i + \Delta t M^{-1}f(x_{i+1}))$,

so we think of this equation as setting the "newton_gradient" (CODE: `partial_newton_G(...)`) to 0. The "newton gradient" measures the convergence of the newton's method optimizer. When the gradient is 0, the "newton_energy" is optimized. 

One further simplification of the "newton gradient" is to multiply both sides by the mass matrix $M$ to remove the inverse term:

$ G : 0 = Mx_{i+1} - Mx_i - \Delta t M v_i - \Delta t^2 f(x_{i+1})$.


## Newton's Method
The "newton energy" (CODE: `partial_newton_E(...)`), which is the antiderivative of the "newton gradient" mentioned above, is the quantity we optimize in Newton's Method:

$x_{i+1}* = argmin \;\; E = argmin\;\; 0.5  x_{i+1}^TMx_{i+1} - x_{i+1}^TMx_i - \Delta t x_{i+1}^T M v_i + \Delta t^2 e(x_{i+1})$

Notice some of the notation changes and sign-flips here. Force is the negative derivative of energy, thus the sign-flip in the antidifferentiation. The physics energy $e$ is the potential energy of the system (CODE: `potential_energy_sum`) 

$e = e_{ext} + e_{int}(x_{i+1})$. 

Next step, finding the optimal next positions $x_{i+1}$ by newton's method requires finding its hessian, the "newton hessian" (CODE: `partial_newton_H(...)`):

$H(x_{i+1}) : M - \Delta t^2 \nabla f(x_{i+1})$

where $\nabla f = \nabla f_{ext} + \nabla f_{int}(x_{i+1})$. Some of these forces (such as gravity, $f = Mg$) are constant, and the hessian w.r.t $x_{i+1}$, is 0. Some of these forces are linear in $x_{i+1}$ (like penalty springs $f = -Kx$) and the hessian is a constant $K$. Some of these forces like neohookean elasticity are nonlinear in $x_{i+1}$ and the hessian must be computed at each step of newtons method.

At each step of newtons method, we compute the descent direction $d$ 

$d = -H(x_{i+1})^{-1} g(x_{i+1})$

that brings us towards the optimal $x_{i+1}*$. Adding a line search to find the optimal step size $\alpha$ where $d\leftarrow \alpha d$ ensures we converge properly.

## In Terms of Handles

Previously we described physics simulation in terms of vertices. Now we must encode the vertex positions $x$ as a function of skinning handles $z$ such that $x = Bz + x_0$.

Swapping in this transforms, we get the updated "newton energy" optimization:

$z_{i+1}* = argmin \;\; E = argmin\;\; 0.5 z_{i+1}^T B^TMB z_{i+1} - z_{i+1}^T B^T M B z_i - \Delta t z_{i+1}^T B^T M B \dot{z}_i + \Delta t^2 e(z_{i+1})$








