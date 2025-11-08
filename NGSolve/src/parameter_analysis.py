"""
Parameter Analysis for Pedestrian Flow Simulations

This module provides functions to analyze numerical parameters and provide
recommendations for mesh size, polynomial order, and stabilization requirements.

Author: Based on analysis by Claude and Camilo Tello
Date: 2025-01-07
"""

import numpy as np


def analyze_parameters(u0, rho_c, gamma_w, delta, epsilon, mesh, mesh_maxh,
                      p_order, Hwid, Hcol, omega=None, fes_rho=None):
    """
    Analyze numerical parameters for pedestrian flow simulation.

    Computes relevant dimensionless numbers (Péclet, wavelengths, length scales)
    and provides specific recommendations for stable, efficient simulations.

    Parameters
    ----------
    u0 : float
        Free-flow walking speed [m/s]
    rho_c : float
        Critical density [ped/m²]
    gamma_w : float
        Weidmann shape parameter [ped/m²]
    delta : float
        Helmholtz viscosity parameter [m]
    epsilon : float
        Continuity diffusion coefficient [m²]
    mesh : NGSolve Mesh
        The finite element mesh
    mesh_maxh : float
        Target maximum element size [m]
    p_order : int
        Polynomial order of finite elements
    Hwid : float
        Domain width [m]
    Hcol : float
        Domain height [m]
    omega : float, optional
        Underrelaxation parameter for Picard iteration
    fes_rho : NGSolve FESpace, optional
        Finite element space for density (to get exact DOF count)

    Returns
    -------
    dict
        Dictionary containing analysis results:
        - 'Pe_h': Element Péclet number
        - 'lambda_min': Minimum Helmholtz wavelength [m]
        - 'stability_continuity': Stability assessment ('EXCELLENT', 'GOOD', 'MARGINAL', 'POOR')
        - 'stability_helmholtz': Resolution assessment
        - 'h_recommended': Recommended mesh size [m]
        - 'recommendations': List of recommendation strings

    Example
    -------
    >>> from ngsolve import *
    >>> from src import analyze_parameters
    >>>
    >>> # After defining parameters and mesh
    >>> results = analyze_parameters(
    ...     u0=1.36, rho_c=8.0, gamma_w=1.913,
    ...     delta=0.1, epsilon=0.1,
    ...     mesh=mesh, mesh_maxh=0.05, p_order=2,
    ...     Hwid=1.0, Hcol=1.0, omega=0.7
    ... )
    >>>
    >>> # Check if stable
    >>> if results['stability_continuity'] == 'GOOD':
    ...     print("✓ Stable configuration!")
    """

    print("=" * 70)
    print("PARAMETER ANALYSIS AND NUMERICAL RECOMMENDATIONS")
    print("=" * 70)

    # ========================================
    # 1. Physical Parameters Summary
    # ========================================
    print("\n" + "="*70)
    print("1. PHYSICAL PARAMETERS")
    print("="*70)

    print(f"\nFundamental Diagram:")
    print(f"  u₀ (free-flow speed)      = {u0:.3f} m/s")
    print(f"  ρc (critical density)     = {rho_c:.1f} ped/m²")
    print(f"  γ  (Weidmann parameter)   = {gamma_w:.3f} ped/m²")

    print(f"\nRegularization:")
    print(f"  δ (Helmholtz viscosity)   = {delta:.3f} m")
    print(f"  ε (continuity diffusion)  = {epsilon:.3f} m²")

    # Weidmann speed function
    def weidmann_speed_py(rho_val):
        rho_reg = max(rho_val, 1e-10)
        speed = u0 * (1 - np.exp(-gamma_w * (1/rho_reg - 1/rho_c)))
        speed = min(max(speed, 0.0), u0)
        return speed

    # Compute speeds at different densities
    rho_test = [0.5, 1.0, 2.0, 4.0, 6.0]
    print(f"\nSpeed at different densities:")
    for rho in rho_test:
        speed = weidmann_speed_py(rho)
        print(f"  f({rho:.1f}) = {speed:.3f} m/s")

    # ========================================
    # 2. Characteristic Length Scales
    # ========================================
    print("\n" + "="*70)
    print("2. CHARACTERISTIC LENGTH SCALES")
    print("="*70)

    L_domain = np.sqrt(Hwid * Hcol)
    print(f"\nDomain:")
    print(f"  Width × Height = {Hwid:.2f} × {Hcol:.2f} m")
    print(f"  L (characteristic) = {L_domain:.2f} m")

    # Diffusion length
    l_epsilon = np.sqrt(epsilon * L_domain / u0)
    print(f"\nDiffusion length (continuity):")
    print(f"  lε = √(ε·L/u₀) = {l_epsilon:.4f} m = {l_epsilon*100:.2f} cm")
    print(f"  Physical meaning: density spreads ~{l_epsilon*100:.1f} cm")

    # Screening length
    l_delta = delta * u0
    print(f"\nScreening length (Helmholtz):")
    print(f"  lδ = δ·u₀ = {l_delta:.4f} m = {l_delta*100:.2f} cm")
    print(f"  Physical meaning: path smoothing ~{l_delta*100:.1f} cm")

    # Helmholtz wavelengths
    print(f"\nHelmholtz wavelength λ = 2π·δ·f(ρ):")
    lambda_values = []
    for rho in rho_test:
        speed = weidmann_speed_py(rho)
        wavelength = 2 * np.pi * delta * speed
        lambda_values.append(wavelength)
        print(f"  λ(ρ={rho:.1f}) = {wavelength:.4f} m = {wavelength*100:.2f} cm")

    lambda_min = min(lambda_values)
    print(f"\n  λ_min ≈ {lambda_min:.4f} m (at high density)")
    print(f"  ⚠ THIS CONTROLS MESH REFINEMENT! ⚠")

    # ========================================
    # 3. Dimensionless Numbers
    # ========================================
    print("\n" + "="*70)
    print("3. DIMENSIONLESS NUMBERS")
    print("="*70)

    # Global Peclet
    Pe_global = (u0 * L_domain) / epsilon
    print(f"\nGlobal Péclet number:")
    print(f"  Pe = (u₀·L)/ε = {Pe_global:.2f}")
    if Pe_global < 1:
        print(f"  → Diffusion-dominated globally")
    elif Pe_global < 10:
        print(f"  → Balanced advection-diffusion")
    else:
        print(f"  → Convection-dominated globally")

    # Dimensionless ratios
    ratio_epsilon = (l_epsilon**2) / (L_domain**2)
    ratio_delta = (l_delta**2) / (L_domain**2)
    print(f"\nDimensionless ratios (Schmid et al.):")
    print(f"  l²ε/L² = {ratio_epsilon:.6f} (inverse Péclet)")
    print(f"  l²δ/L² = {ratio_delta:.6f} (screening parameter)")

    # Effective Peclet
    f_typical = weidmann_speed_py(2.0)
    epsilon_eff = np.sqrt(epsilon**2 + (delta * f_typical)**2)
    Pe_eff = (f_typical * L_domain) / epsilon_eff
    print(f"\nEffective Péclet (two-scale):")
    print(f"  f(ρ=2) = {f_typical:.3f} m/s")
    print(f"  ε_eff = √(ε² + (δ·f)²) = {epsilon_eff:.4f} m²")
    print(f"  Pe_eff = (f·L)/ε_eff = {Pe_eff:.2f}")

    # ========================================
    # 4. Current Mesh Analysis
    # ========================================
    print("\n" + "="*70)
    print("4. CURRENT MESH ANALYSIS")
    print("="*70)

    print(f"\nMesh properties:")
    print(f"  h_max (target)     = {mesh_maxh:.4f} m = {mesh_maxh*100:.2f} cm")
    print(f"  # elements         = {mesh.ne}")
    print(f"  # vertices         = {mesh.nv}")
    print(f"  polynomial order   = {p_order}")

    # Estimate actual h
    h_actual = np.sqrt(Hwid * Hcol / mesh.ne)
    print(f"  h_avg (estimated)  = {h_actual:.4f} m = {h_actual*100:.2f} cm")

    # Element Peclet
    Pe_h = (u0 * h_actual) / (2 * epsilon * p_order)
    print(f"\nElement Péclet number:")
    print(f"  Pe_h = (u₀·h)/(2·ε·p) = {Pe_h:.3f}")

    if Pe_h < 0.5:
        print(f"  → ✓✓ Very stable (diffusion-dominated)")
        stability_continuity = "EXCELLENT"
    elif Pe_h < 1.0:
        print(f"  → ✓ Stable (standard Galerkin works)")
        stability_continuity = "GOOD"
    elif Pe_h < 2.0:
        print(f"  → ⚠ Borderline (may need SUPG or refinement)")
        stability_continuity = "MARGINAL"
    else:
        print(f"  → ✗ Unstable (need SUPG or finer mesh)")
        stability_continuity = "POOR"

    # Helmholtz resolution
    points_per_wavelength = lambda_min / (h_actual * p_order)
    print(f"\nHelmholtz wavelength resolution:")
    print(f"  λ_min / (h·p) = {points_per_wavelength:.2f} points per effective element")

    if points_per_wavelength < 1.0:
        print(f"  → ✗ UNDER-RESOLVED! Refine mesh or increase order")
        stability_helmholtz = "POOR"
    elif points_per_wavelength < 1.5:
        print(f"  → ⚠ Marginally resolved")
        stability_helmholtz = "MARGINAL"
    elif points_per_wavelength < 3.0:
        print(f"  → ✓ Adequately resolved")
        stability_helmholtz = "GOOD"
    else:
        print(f"  → ✓✓ Well resolved")
        stability_helmholtz = "EXCELLENT"

    # DOF count
    print(f"\nDegrees of freedom:")
    if fes_rho is not None:
        dof_rho = fes_rho.ndof
        print(f"  DOFs per variable  = {dof_rho}")
        print(f"  Total DOFs (ρ,ψ,u)= ~{dof_rho * 3}")
    else:
        # Estimate
        if p_order == 1:
            dof_estimate = int(mesh.nv)
        elif p_order == 2:
            dof_estimate = int(mesh.nv + mesh.ne * 3 / 2)
        elif p_order == 3:
            dof_estimate = int(mesh.nv + mesh.ne * 2 + mesh.ne)
        else:
            dof_estimate = int(mesh.nv * (p_order + 1))
        print(f"  DOFs per variable  ≈ {dof_estimate} (estimated)")
        print(f"  Total DOFs (ρ,ψ,u)≈ ~{dof_estimate * 3} (estimated)")

    # ========================================
    # 5. Recommendations
    # ========================================
    print("\n" + "="*70)
    print("5. RECOMMENDATIONS")
    print("="*70)

    # Limiting factor
    if stability_helmholtz in ["POOR", "MARGINAL"]:
        limiting_factor = "Helmholtz wavelength resolution"
    elif stability_continuity in ["POOR", "MARGINAL"]:
        limiting_factor = "Element Péclet number"
    else:
        limiting_factor = "None (well-resolved)"

    print(f"\nLimiting factor: {limiting_factor}")
    print(f"Stability: Continuity={stability_continuity}, Helmholtz={stability_helmholtz}")

    # Mesh size recommendation
    h_peclet = 2 * epsilon * p_order / u0
    h_helmholtz = lambda_min / 6.0
    h_recommended = min(h_peclet, h_helmholtz) * 0.8

    print(f"\n{'─'*70}")
    print("MESH SIZE:")
    print(f"{'─'*70}")
    print(f"  From Pe_h < 1:         h ≤ {h_peclet:.4f} m")
    print(f"  From λ resolution:     h ≤ {h_helmholtz:.4f} m")
    print(f"  ⭐ RECOMMENDED:        h ≤ {h_recommended:.4f} m")

    if h_actual > h_recommended:
        print(f"\n  ⚠ Current h={h_actual:.4f} > recommended!")
        print(f"  ⚠ Consider refining mesh")
    elif h_actual > h_recommended * 0.5:
        print(f"\n  ✓ Current mesh acceptable")
    else:
        print(f"\n  ✓✓ Current mesh well-refined")

    # Polynomial order
    print(f"\n{'─'*70}")
    print("POLYNOMIAL ORDER:")
    print(f"{'─'*70}")

    if p_order == 1:
        print(f"  ⚠ Order-1: discontinuous ∇ψ")
        print(f"  💡 Use order-2 for smooth velocity")
    elif p_order == 2:
        print(f"  ✓✓ Order-2: optimal choice")
    elif p_order == 3:
        print(f"  ✓ Order-3: accurate but ~5× expensive")
        print(f"  💡 Consider order-2 for efficiency")
    else:
        print(f"  ⚠ Order-{p_order}: likely overkill")

    # Stabilization
    print(f"\n{'─'*70}")
    print("STABILIZATION:")
    print(f"{'─'*70}")

    recommendations = []

    if Pe_h < 1.0:
        print(f"  Pe_h = {Pe_h:.3f} < 1")
        print(f"  ✓ Standard Galerkin stable")
        print(f"  💡 NO SUPG needed!")
        recommendations.append("Remove SUPG stabilization")
    elif Pe_h < 2.0:
        print(f"  Pe_h = {Pe_h:.3f} ∈ [1, 2]")
        print(f"  ⚠ Borderline stability")
        print(f"  💡 Try without SUPG first")
        recommendations.append("Try standard Galerkin first")
    else:
        print(f"  Pe_h = {Pe_h:.3f} > 2")
        print(f"  ✗ Likely unstable")
        print(f"  💡 Increase ε or use SUPG")
        recommendations.append(f"Increase ε to {(u0*h_actual)/(2*p_order):.3f}")

    # Underrelaxation
    if omega is not None:
        print(f"\n{'─'*70}")
        print("UNDERRELAXATION:")
        print(f"{'─'*70}")
        print(f"  ω = {omega:.2f}")

        if omega < 0.3:
            print(f"  ⚠ Too aggressive → slow convergence")
            print(f"  💡 Increase to ω = 0.7")
            recommendations.append("Increase ω to 0.7")
        elif omega <= 0.9:
            print(f"  ✓ Good range")
        else:
            print(f"  ⚠ Too weak → may diverge")

    # Summary
    print(f"\n{'='*70}")
    print("6. QUICK REFERENCE")
    print(f"{'='*70}")

    print(f"\n{'Parameter':<25} {'Value':<15} {'Status':<30}")
    print(f"{'-'*70}")
    print(f"{'ε (diffusion)':<25} {f'{epsilon:.4f} m²':<15} {('✓ Good' if epsilon >= 0.05 else '⚠ Small'):<30}")
    print(f"{'δ (viscosity)':<25} {f'{delta:.4f} m':<15} {'✓ Typical':<30}")
    print(f"{'h (element size)':<25} {f'{h_actual:.4f} m':<15} {('✓ Good' if h_actual <= h_recommended else '⚠ Too coarse'):<30}")
    print(f"{'p (poly order)':<25} {f'{p_order}':<15} {('⭐ Optimal' if p_order == 2 else '○ OK'):<30}")
    print(f"{'Pe_h':<25} {f'{Pe_h:.3f}':<15} {stability_continuity:<30}")
    print(f"{'λ_min/(h·p)':<25} {f'{points_per_wavelength:.2f}':<15} {stability_helmholtz:<30}")
    if omega is not None:
        omega_status = '✓ Good' if 0.5 <= omega <= 0.9 else ('⚠ Too small' if omega < 0.5 else '⚠ Too large')
        print(f"{'ω (underrelaxation)':<25} {f'{omega:.2f}':<15} {omega_status:<30}")

    print(f"\n{'='*70}")
    print("END OF ANALYSIS")
    print(f"{'='*70}\n")

    # Return results dictionary
    return {
        'Pe_h': Pe_h,
        'Pe_global': Pe_global,
        'Pe_eff': Pe_eff,
        'lambda_min': lambda_min,
        'l_epsilon': l_epsilon,
        'l_delta': l_delta,
        'h_actual': h_actual,
        'h_recommended': h_recommended,
        'points_per_wavelength': points_per_wavelength,
        'stability_continuity': stability_continuity,
        'stability_helmholtz': stability_helmholtz,
        'limiting_factor': limiting_factor,
        'recommendations': recommendations
    }
