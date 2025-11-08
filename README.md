# Pedestrian Flow Simulation in NGSolve

This repository contains NGSolve implementations of pedestrian flow models based on the Hughes model with regularization.

## Overview

The project implements stationary pedestrian flow simulations using a coupled system of PDEs:
- **Helmholtz equation** for path planning (derived from Eikonal equation via Cole-Hopf transformation)
- **Continuity equation** for density evolution with diffusion regularization
- **Weidmann fundamental diagram** for density-dependent walking speed

## Repository Structure

```
pedestrianFlowInNGSolve/
├── NGSolve/                           # Implementation files
│   ├── src/                               # 🆕 Python utilities
│   │   ├── __init__.py                    # Package initialization
│   │   ├── parameter_analysis.py          # Parameter analysis function
│   │   └── README.md                      # API documentation
│   ├── stationary_singlePhase_pedestrianFlow.ipynb               # Single-group Picard
│   ├── stationary_singlePhase_pedestrianFlow_monolithicNewton.ipynb  # 🆕 Single-group Newton
│   ├── stationary_singlePhase_pedestrianFlow_SUPG.ipynb          # SUPG-stabilized variant
│   ├── benchmark_Picard_vs_Newton.ipynb                          # 🆕 Solver comparison
│   └── stationary_twoGroup_pedestrianFlow.ipynb                  # Two-group counter-flow
├── references/                        # Reference papers and literature
└── TGF2024_Proceedings_MatthiasSCHMID_finalSubmission.pdf  # Original paper
```

## Implemented Models

### 1. Single-Group Pedestrian Flow
**File**: `NGSolve/stationary_singlePhase_pedestrianFlow.ipynb`

A stationary single-group model where pedestrians:
- Enter from the top boundary
- Exit at the right boundary
- Follow optimal paths determined by the potential field ψ

**Features**:
- Picard iteration solver for the coupled nonlinear system
- Order-2 H¹ finite elements
- Regularized formulation with diffusion (ε = 0.1 m²)
- Weidmann fundamental diagram for realistic walking speeds

### 2. Two-Group Counter-Flow
**File**: `NGSolve/stationary_twoGroup_pedestrianFlow.ipynb`

A stationary two-group model with pedestrians moving in opposite directions:
- **Group 1**: Enters from left, exits at top
- **Group 2**: Enters from right, exits at bottom
- Groups are coupled through **total density** ρ = ρ₁ + ρ₂

**Key Coupling**:
Both groups feel the total density in:
- Speed function: f(ρ_total)
- Helmholtz equation: κ²(ρ_total) = 1/(δ² f²(ρ_total))

This creates realistic crowd dynamics where groups influence each other's movement.

### 3. Monolithic Newton Method (Single-Group)
**File**: `NGSolve/stationary_singlePhase_pedestrianFlow_monolithicNewton.ipynb`

Fully coupled monolithic solver using NGSolve's built-in Newton method with automatic differentiation for the Jacobian.

**Features**:
- Compound finite element space for coupled (ρ, ψ) system
- Automatic Jacobian computation via symbolic differentiation
- Smooth Weidmann function (no IfPos conditionals for differentiability)
- InnerProduct for gradient norm computation
- Damping factor for robustness (dampfactor = 0.5)

**Current Status (⚠ Experimental)**:

The monolithic Newton method offers **quadratic convergence** when it works, but faces challenges converging from cold start for this highly nonlinear problem:

**Convergence Challenges**:
- Multiple coupled nonlinearities: κ²(ρ) = 1/(δ²f²(ρ)), velocity normalization u = f(ρ)∇ψ/||∇ψ||
- Sensitive to initial guess - may diverge from simple initial conditions (ρ=0.1, ψ=y/Hcol)
- Even with damping (dampfactor=0.5) and smooth formulations, cold-start convergence not guaranteed

**Key Technical Learnings**:
- IfPos conditionals create **discontinuous derivatives** that break automatic differentiation (even though function appears continuous)
- Smooth regularization required: `sqrt(ρ² + ρ_min²)` to approximate max(ρ, ρ_min) with continuous derivatives
- Gradient normalization η must be inside sqrt: `sqrt(||∇ψ||² + η)` for proper regularization at stagnation points
- Trial functions (not GridFunctions) required in residual for symbolic differentiation

**Trade-offs vs Picard**:
- **Newton**: Fast (quadratic) when converges, but requires good initial guess
- **Picard**: Robust (always converges), but slow (linear convergence, ~31 iterations)

This represents ongoing research into robust monolithic solvers for coupled pedestrian flow systems.

### 4. Picard vs Newton Benchmark
**File**: `NGSolve/benchmark_Picard_vs_Newton.ipynb`

Side-by-side comparison of Picard iteration and monolithic Newton methods starting from identical initial conditions.

**Metrics Compared**:
- Iteration count
- Computation time
- Convergence rate
- Solution agreement

**Key Findings**:
- Picard: Reliable convergence but requires ~30-40 iterations
- Newton: Quadratic convergence when successful, but sensitive to initialization
- Demonstrates importance of solver choice based on application requirements (robustness vs speed)

### 5. SUPG-Stabilized Single-Group Flow
**File**: `NGSolve/stationary_singlePhase_pedestrianFlow_SUPG.ipynb`

An experimental variant of the single-group model using SUPG (Streamline Upwind Petrov-Galerkin) stabilization for convection-dominated problems with reduced diffusion coefficient.

**Features**:
- SUPG stabilization: τ(u·∇w)(u·∇ρ) with τ = C_supg · h / (2||u||)
- Underrelaxation for Picard iteration: ρ^(k+1) = ω·ρ_new + (1-ω)·ρ^(k)
- Target: Achieve convergence with ε = 0.01 m² (10× reduction from standard model)

**Convergence Characteristics**:

The SUPG method enables stable solutions for reduced diffusion, but convergence becomes increasingly challenging as ε decreases:

| ε [m²] | C_supg | ω | Max Iter | Status |
|--------|--------|---|----------|--------|
| 0.1 | N/A | 1.0 | 100 | ✓ Converges (standard model, ~31 iterations) |
| 0.05 | 1.0 | 0.5 | 200 | ✓ Converges with SUPG |
| 0.01 | 2.0 | 0.2 | 500 | ✗ Challenging convergence |
| 0.01 | 20.0 | 0.01 | 500 | ✗ Near convergence, not achieved |

**Current Limitation**:

With ε = 0.01 m², the Péclet number Pe = ||u||h/(2ε) ≈ O(10²-10³) indicates extreme convection dominance. Despite aggressive stabilization (C_supg = 20) and strong underrelaxation (ω = 0.01), full convergence has not been achieved. The problem may require:
- Alternative stabilization methods (GLS, shock-capturing)
- Adaptive time-stepping approaches
- Mesh refinement strategies
- Different nonlinear solver approaches (Newton-Raphson instead of Picard)

This represents an active area of research for low-diffusion pedestrian flow simulations.

**Note**: Line 94 originally suggested "Newton-Raphson instead of Picard" as a potential solution. See section 3 (Monolithic Newton Method) for experimental results showing Newton also faces convergence challenges from cold start.

## Mathematical Model

### Strong Form

For each group *i*:

**Continuity Equation**:
```
∇ · (-ε∇ρᵢ + ρᵢuᵢ) = 0    in Ω
```

**Helmholtz Equation**:
```
Δψᵢ - (1/(δ²fᵢ²(ρ))) ψᵢ = 0    in Ω
```

**Velocity Field**:
```
uᵢ = fᵢ(ρ) ∇ψᵢ / ||∇ψᵢ||
```

### Parameters

| Parameter | Symbol | Value | Unit | Description |
|-----------|--------|-------|------|-------------|
| Free-flow speed | u₀ | 1.36 | m/s | Maximum walking speed |
| Critical density | ρ_c | 8.0 | ped/m² | Density at zero speed |
| Weidmann parameter | γ | 1.913 | ped/m² | Shape parameter |
| Viscosity | δ | 0.1 | m | Regularization parameter |
| Diffusion | ε | 0.1 | m² | Diffusion coefficient |

### Weidmann Fundamental Diagram

```python
f(ρ) = u₀ (1 - exp(-γ(1/ρ - 1/ρ_c)))
```

This gives realistic speed-density relationships:
- Low density (ρ < 1 ped/m²): Free-flow speed ≈ u₀
- Medium density (ρ ≈ 2-4 ped/m²): Reduced speed
- High density (ρ → ρ_c): Speed → 0 (jam conditions)

## Solution Methods

### Picard Iteration Algorithm

For each iteration k = 0, 1, 2, ...:

1. **Update total density**: ρ = ρ₁ + ρ₂ (for two-group model)
2. **For each group**:
   - Solve Helmholtz equation for ψᵢ using current ρ
   - Compute velocity uᵢ from ∇ψᵢ
   - Solve Continuity equation for ρᵢ using current uᵢ
3. **Check convergence**: ||ρ^(k+1) - ρ^(k)|| < tol

**Advantages**:
- Robust: Always converges for well-posed problems
- Simple implementation
- Each subproblem is linear

**Disadvantages**:
- Linear convergence rate (slow)
- Typically requires 30-40 iterations

### Monolithic Newton Method

Solves the fully coupled nonlinear system using Newton-Raphson:

1. **Assemble residual**: R(ρ, ψ) for coupled system
2. **Automatic differentiation**: Jacobian J = ∂R/∂(ρ,ψ) computed symbolically
3. **Newton iteration**: Solve J·Δu = -R, update (ρ,ψ) ← (ρ,ψ) + damp·Δu
4. **Check convergence**: ||R|| < tol

**Advantages**:
- Quadratic convergence (very fast when it works)
- Fewer iterations needed (typically 3-5 if converges)

**Disadvantages**:
- Sensitive to initial guess - may not converge from cold start
- Requires smooth, differentiable nonlinearities
- More complex implementation

**Implementation Notes**:
- Uses NGSolve's built-in `Newton()` solver with automatic Jacobian computation
- Requires smooth formulation of Weidmann function (no IfPos conditionals)
- Damping (dampfactor < 1.0) improves robustness
- See `benchmark_Picard_vs_Newton.ipynb` for detailed comparison

### Boundary Conditions

| Boundary | Density (ρ) | Potential (ψ) |
|----------|-------------|---------------|
| Walls | No-flux: (-ε∇ρ + ρu)·n = 0 | Neumann: ∇ψ·n = 0 |
| Exits | Free outflow: (-ε∇ρ)·n = 0 | Dirichlet: ψ = 1 |
| Entries | Prescribed flux: -(-ε∇ρ + ρu)·n = g | Robin: (u₀δ∇ψ)·n + ψ = 0 |

## Running the Code

### Prerequisites

```bash
pip install ngsolve numpy jupyter
```

### 🆕 Parameter Analysis Tool (Recommended!)

Before running simulations, use the parameter analysis tool to verify your parameters:

```python
# In your notebook (in NGSolve/ folder), after defining parameters:
from src import analyze_parameters

results = analyze_parameters(
    u0=u0, rho_c=rho_c, gamma_w=gamma_w,
    delta=delta, epsilon=epsilon,
    mesh=mesh, mesh_maxh=mesh_maxh, p_order=p_order,
    Hwid=Hwid, Hcol=Hcol, omega=omega
)
```

**Benefits**:
- ✓ Instant feedback on parameter choices
- ✓ Specific recommendations for h, p, ε, ω
- ✓ Identifies potential stability issues before running solver
- ✓ Saves hours of debugging time

See **[docs/06_QUICK_START_GUIDE.md](docs/06_QUICK_START_GUIDE.md)** for detailed usage.

### Running Notebooks

**Recommended starting point (robust Picard solver)**:
```bash
jupyter notebook NGSolve/stationary_singlePhase_pedestrianFlow.ipynb
```

**Experimental Newton solver**:
```bash
jupyter notebook NGSolve/stationary_singlePhase_pedestrianFlow_monolithicNewton.ipynb
```

**Solver comparison benchmark**:
```bash
jupyter notebook NGSolve/benchmark_Picard_vs_Newton.ipynb
```

**Two-group counter-flow**:
```bash
jupyter notebook NGSolve/stationary_twoGroup_pedestrianFlow.ipynb
```

### Expected Results

**Single-group model**:
- Converges in ~31 iterations with ε = 0.1
- Mean density: ~0.2-0.3 ped/m²
- Maximum density: ~3-4 ped/m² (at entrance)

**Two-group model**:
- Two distinct flow patterns in opposite directions
- Interaction zone where groups cross paths
- Total density shows combined crowding effects

## Numerical Details

### Finite Element Discretization

- **Space**: H¹(Ω) with order-2 elements
- **DOFs**: ~500 per variable (for mesh size h = 0.1)
- **Solver**: Direct solver with FreeDofs for linear systems

### Post-Processing

**Computing min/max values**:

Since we use order-2 elements (DOFs include edge values), we interpolate to order-1 elements for meaningful min/max:

```python
fes_p1 = H1(mesh, order=1)
gf_p1 = GridFunction(fes_p1)
gf_p1.Set(gf_rho)  # Interpolate to P1

rho_min = min(gf_p1.vec)  # Min at vertices
rho_max = max(gf_p1.vec)  # Max at vertices
```

**Computing mean density**:

```python
domain_area = Integrate(1.0 * dx, mesh)
rho_integral = Integrate(gf_rho * dx, mesh)
rho_mean = rho_integral / domain_area
```

## Documentation

### Theoretical Background

See `docs/01_WEAK_FORMULATION_DERIVATION.md` for:
- Complete derivation of weak forms
- Step-by-step integration by parts
- Boundary condition application
- NGSolve implementation details

### References

The implementation is based on:
- **Hughes (2002)**: Original crowd flow model
- **Weidmann (1993)**: Fundamental diagram for pedestrian speed
- **Bellomo & Dogbé (2011)**: Mathematical modeling of crowds
- **Schmid & Bernhardsgrütter (2024)**: TGF 2024 Proceedings - Theoretical framework for multi-group pedestrian flow
- **Schöberl (2014)**: C++11 Implementation of Finite Elements in NGSolve, ASC Report 30/2014, Institute for Analysis and Scientific Computing, Vienna University of Technology

## Author & Acknowledgments

Implementation by **Camilo Tello Fachin, MSc** for pedestrian flow research.

Based on the theoretical work by **Matthias Schmid** and **David Bernhardsgrütter** (TGF 2024 Proceedings).

Built with [NGSolve](https://ngsolve.org/) - A high-performance finite element library developed by Joachim Schöberl.

## License

This project is licensed under the GNU Lesser General Public License v2.1 (LGPL-2.1) - see the [LICENSE](LICENSE) file for details.
