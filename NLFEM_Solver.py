# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 21:36:33 2025

@author: dines
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.ticker as mticker

def analytical_solution(variant, n_steps):
    E, sigma_0, eta, m, A1, A2, L1, L2, t_tot, F_final = variant
    time_steps = np.linspace(0, t_tot, n_steps)
    
    R1_array = []
    stress_1_array, stress_2_array = [], []  # Separate stress lists
    strain_1_array, strain_2_array = [], []  # Separate strain lists
    displacement_at_L1 = []

    # Variables to track plasticity initiation
    Fi = None  # Force at which plasticity starts
    time_at_plasticity_start_analytical = None
    displacement_analytical_at_plasticity_start = None

    for t in time_steps:
        F = (F_final / t_tot) * t  # Linearly increasing force
        denominator = (L1 / A1) + (L2 / A2)
        R1 = (F * L2 / A2) / denominator
        R2 = (F * L1 / A1) / denominator

        sigma_1 = R1 / A1  # Stress in segment 1
        sigma_2 = R2 / A2  # Stress in segment 2
        
        strain_1 = sigma_1 / E  # Strain in segment 1
        strain_2 = sigma_2 / E  # Strain in segment 2
        
        displacement_L1 = R1 * L1 / (A1 * E)  # Displacement at L1

        # Store values separately
        R1_array.append(R1)
        stress_1_array.append(sigma_1)
        stress_2_array.append(sigma_2)
        strain_1_array.append(strain_1)
        strain_2_array.append(strain_2)
        displacement_at_L1.append(displacement_L1)
                
        if Fi is None and sigma_0 is not None and sigma_1 is not None and sigma_2 is not None:
            if abs(sigma_1) >= sigma_0 or abs(sigma_2) >= sigma_0:
                Fi = F  # Store the force at which plasticity initiates
                time_at_plasticity_start_analytical = t
                displacement_analytical_at_plasticity_start = displacement_L1

    return time_steps, strain_1_array, strain_2_array, stress_1_array, stress_2_array, displacement_at_L1, Fi, time_at_plasticity_start_analytical, displacement_analytical_at_plasticity_start



# Ansatzfunktionen und ihre Ableitungen
def shape_functions(xi):
    N1 = 0.5 * (1 - xi)
    N2 = 0.5 * (1 + xi)
    return np.array([N1, N2])

def shape_function_derivatives():
    return np.array([-0.5, 0.5])  # Konstante Ableitungen

# Element-Routine
# Gauß-Quadratur für die lokale Steifigkeitsmatrix
def element_routine(E, A, L, material_routine, eps_p, sigma_0, eta, m, dt):
    gauss_points = [-1/np.sqrt(3), 1/np.sqrt(3)]  # 2-Punkt-Quadratur
    weights = [1, 1]
    K_local = np.zeros((2, 2))
    for xi, w in zip(gauss_points, weights):
        dN_dxi = shape_function_derivatives()
        J = L / 2  # Jakobideterminante
        B = dN_dxi / J
        strain = 0  # Placeholder for strain computation during stiffness matrix assembly
        if material_routine == viscoplastic_material_routine:
            _, _, C_t = material_routine(strain, eps_p, sigma_0, eta, m, dt, E)
        else:
            _, C_t = material_routine(strain, E)
        K_local += w * (C_t * A * np.outer(B, B) * J)
    return K_local

# Linear Material-Routine
def linear_material_routine(eps, E):
    sigma = E * eps
    C_t = E
    return sigma, C_t

# Viscoplastic Material-Routine
def viscoplastic_material_routine(eps, eps_p, sigma_0, eta, m, dt, E):
    # Trial stress
    sigma_trial = E * (eps - eps_p)
    if abs(sigma_trial) > sigma_0:
        # Plastic case
        lambda_val = ((abs(sigma_trial) / sigma_0 - 1) ** m)
        delta_eps_p = eta * dt * np.sign(sigma_trial) * lambda_val
        eps_p += delta_eps_p
        sigma = E * (eps - eps_p)
        C_t = E / (1 + (eta * E * dt / sigma_0))
    else:
        # Elastic case
        sigma = sigma_trial
        C_t = E
    return sigma, eps_p, C_t

# Berechnung der internen Kräfte
def compute_internal_forces(u, E, node_positions, L1, A1, A2, element_length, total_elements, eps_p, sigma_0, eta, m, dt, material_routine):
    internal_forces = np.zeros_like(u)
    for i in range(total_elements):
        element_start = node_positions[i]
        A = A1 if element_start < L1 else A2  # Select appropriate area
        gauss_points = [-1/np.sqrt(3), 1/np.sqrt(3)]  # 2-Punkt-Quadratur
        weights = [1, 1]
        for xi, w in zip(gauss_points, weights):
            dN_dxi = shape_function_derivatives()
            J = element_length / 2  # Jakobideterminante
            B = dN_dxi / J
            strain = np.dot(B, u[i:i+2])
            if material_routine == viscoplastic_material_routine:
                sigma, eps_p[i], _ = material_routine(strain, eps_p[i], sigma_0, eta, m, dt, E)
            else:
                sigma, _ = material_routine(strain, E)
            P = sigma * A
            internal_forces[i:i+2] += w * P * B * J
    return internal_forces

# Zeitschritt durchführen
def time_step(u, K_mod, F_mod, total_elements, element_length, E, node_positions, L1, A1, A2, sigma_0, eta, m, dt, eps_p, material_routine):
    delta_u = np.zeros_like(u[1:-1])  # Initialisierung von delta_u
    for iteration in range(10):   
        internal_forces = compute_internal_forces(u, E, node_positions, L1, A1, A2, element_length, total_elements, eps_p, sigma_0, eta, m, dt, material_routine)
        R = internal_forces[1:-1] - F_mod
                   
        # **Compute Infinity Norms (Max Absolute Component)**
        norm_R_inf = np.max(np.abs(R))  
        norm_F_inf = np.max(np.abs(F_mod))  
        norm_delta_u_inf = np.max(np.abs(delta_u))  
        norm_u_inf = np.max(np.abs(u[1:-1]))

        # **Check Convergence Criteria**
        if (norm_R_inf < 0.005 * norm_F_inf) and (norm_delta_u_inf < 0.005 * norm_u_inf):
            break
        
        delta_u = np.linalg.solve(K_mod, -R)
        u[1:-1] += delta_u
        
    return u

# Haupt-FEM-Solver
def fem_solver(variant, total_elements, material_routine, dt):
    print("-" * 80)  # Separator Line
    print(f"\n🚀 Starting FEM Solver Call: Elements = {total_elements}, Time Step = {dt:.6f}")
    # Extract parameters
    E, sigma_0, eta, m, A1, A2, L1, L2, t_tot, F_hat = variant

    # Discretize the beam into nodes
    node_positions = np.linspace(0, L1 + L2, total_elements + 1)

    # Time-stepping setup
    n_steps = int(t_tot / dt)

    # Compute Analytical Solution **inside** fem_solver
    time_analytical, strain_1_analytical, strain_2_analytical, stress_1_analytical, stress_2_analytical, displacement_analytical, Fi, time_at_plasticity_start_analytical, displacement_analytical_at_plasticity_start = analytical_solution(variant, n_steps)

    # Initialize system matrices and vectors
    K_global = np.zeros((total_elements + 1, total_elements + 1))
    F_ext = np.zeros(total_elements + 1)

    # Assembly of global stiffness matrix
    element_length = (L1 + L2) / total_elements
    eps_p = np.zeros(total_elements) if material_routine == viscoplastic_material_routine else None
    for i in range(total_elements):
        element_start = node_positions[i]
        A = A1 if element_start < L1 else A2
        K_local = element_routine(E, A, element_length, material_routine, eps_p[i] if eps_p is not None else 0, sigma_0, eta, m, dt)
        K_global[i:i + 2, i:i + 2] += K_local

    # Apply boundary conditions
    K_mod = K_global[1:-1, 1:-1]
    F_mod = F_ext[1:-1]

    # Storage for results
    time = []
    displacements_at_force_node = []
    element_strains = np.zeros(total_elements)
    element_stresses = np.zeros(total_elements)

    # Storage for stress-strain plots
    max_stresses = []
    max_strains = []
    max_plastic_strains = []

    # Initialize displacement field
    u = np.zeros(total_elements + 1)

    # Variables to track plasticity initiation
    load_at_plasticity_start = None
    time_at_plasticity_start = None
    max_stress_at_plasticity_start = None
    max_strain_at_plasticity_start = None


    # Add a flag before the loop to track if verification has been printed
    plasticity_verified = False  # Ensure this is declared outside the time-stepping loop

    # Time-stepping loop
    t = 0.0
    plasticity_initiation_index = None
    for step in range(n_steps):
        t += dt
        force_node_index = np.argmin(np.abs(node_positions - L1))
        F_ext[force_node_index] = F_hat * (t / t_tot)
        F_mod = F_ext[1:-1]

        u = time_step(u, K_mod, F_mod, total_elements, element_length, E, node_positions, L1, A1, A2, sigma_0, eta, m, dt, eps_p, material_routine)

        # Compute strain and stress for all elements
        for i in range(total_elements):
            element_strains[i] = (u[i + 1] - u[i]) / element_length
            if material_routine == viscoplastic_material_routine:
                element_stresses[i], eps_p[i], _ = material_routine(element_strains[i], eps_p[i], sigma_0, eta, m, dt, E)
            else:
                element_stresses[i], _ = material_routine(element_strains[i], E)

        if material_routine == viscoplastic_material_routine and load_at_plasticity_start is None:
            if np.max(element_stresses) >= sigma_0:  # Check if stress exceeds yield stress
                load_at_plasticity_start = F_ext[force_node_index]
                time_at_plasticity_start = t
                max_stress_at_plasticity_start = np.max(element_stresses)
                max_strain_at_plasticity_start = np.max(element_strains)
                plasticity_initiation_index = step
                
        if material_routine == linear_material_routine and np.max(element_stresses) >= sigma_0:  # Check if stress exceeds yield stress
                load_at_plasticity_start = F_ext[force_node_index]
                time_at_plasticity_start = t
                max_stress_at_plasticity_start = np.max(element_stresses)
                max_strain_at_plasticity_start = np.max(element_strains)
                plasticity_initiation_index = step
        
        # Track the max plastic strain at each time step
        max_plast_str = np.max(eps_p) if isinstance(eps_p, np.ndarray) and eps_p.size > 0 else 0  # Store zero if eps_p is empty
        max_plastic_strains.append(max_plast_str)
        
        # Compute max stress and strain
        max_strain = np.max(element_strains)
        max_stress = np.max(element_stresses)

        max_strains.append(max_strain)
        max_stresses.append(max_stress)

        # Store displacement at force node
        displacements_at_force_node.append(u[force_node_index])

        # Store time
        time.append(t)

        # Inside the time-stepping loop:
        if (material_routine == viscoplastic_material_routine and not plasticity_verified and load_at_plasticity_start is not None and Fi is not None):
            print("\n🔍 **Plasticity Initiation Verification**")
            print(f"🔹 Analytical Fi: {Fi:.2f} N at t = {time_at_plasticity_start_analytical:.2f} s")
            print(f"🔹 Numerical Force: {load_at_plasticity_start:.2f} N at t = {time_at_plasticity_start:.2f} s")
            print(f"🔹 Displacement at Plasticity Start in loading point (Analytical): {displacement_analytical_at_plasticity_start:.6f} m")
            print(f"🔹 Displacement at Plasticity Start in loading point (Numerical): {displacements_at_force_node[plasticity_initiation_index]:.6f} m")
        
            # Check if they match within a small error
            force_error = abs((load_at_plasticity_start - Fi) / Fi) * 100
            time_error = abs((time_at_plasticity_start - time_at_plasticity_start_analytical) / time_at_plasticity_start_analytical) * 100
        
            print(f"\n✅ Force Error: {force_error:.2f}%")
            print(f"✅ Time Error: {time_error:.2f}%")
            
            # **Set flag to True** so it doesn't print again
            plasticity_verified = True

        if (material_routine == linear_material_routine and not plasticity_verified and load_at_plasticity_start is not None and Fi is not None):
            print(f"🔹 Analytical Fi: {Fi:.2f} N at t = {time_at_plasticity_start_analytical:.2f} s")
            print(f"🔹 Numerical Force: {load_at_plasticity_start:.2f} N at t = {time_at_plasticity_start:.2f} s")
            print(f"🔹 Displacement at Force Fi in loading point (Analytical): {displacement_analytical_at_plasticity_start:.6f} m")
            print(f"🔹 Displacement at Force Fi in loading point (Numerical): {displacements_at_force_node[plasticity_initiation_index]:.6f} m")
        
            # Check if they match within a small error
            force_error = abs((load_at_plasticity_start - Fi) / Fi) * 100
            time_error = abs((time_at_plasticity_start - time_at_plasticity_start_analytical) / time_at_plasticity_start_analytical) * 100
        
            print(f"\n✅ Force Error: {force_error:.2f}%")
            print(f"✅ Time Error: {time_error:.2f}%")
            
            # **Set flag to True** so it doesn't print again
            plasticity_verified = True
            
    print("✅ FEM Solver Call Completed:\n")    
    print("-" * 80)  # Separator Line
    return (time, displacements_at_force_node, time_analytical, displacement_analytical_at_plasticity_start, max_strains, max_stresses, strain_1_analytical, strain_2_analytical, stress_1_analytical, stress_2_analytical, max_strain_at_plasticity_start, plasticity_initiation_index, displacement_analytical, max_plastic_strains)


# Beispiel-Variante (aus der sechsten Zeile)
variant = [200e9, 400e6, 0.2, 2.5, 10e-6, 20e-6, 40e-3, 80e-3, 2.5, 13000]  # [E, sigma_0, eta, m, A1, A2, L1, L2, t_tot, F_hat]

# Benutzerabfrage für Materialmodell
material_model = input("Choose material model ('linear' or 'viscoplastic'): ").strip().lower()
if material_model == "linear":
    material_routine = linear_material_routine
elif material_model == "viscoplastic":
    material_routine = viscoplastic_material_routine
else:
    print("Invalid choice. Defaulting to linear elastic model.")
    material_routine = linear_material_routine

# Default execution settings
dt_default = 0.01
total_elements_default = 200
print(f"Running with default settings: dt = {dt_default}, elements = {total_elements_default}")

(time, displacements_at_force_node, time_analytical, displacement_analytical_at_plasticity_start, max_strains, max_stresses, strain_1_analytical, strain_2_analytical, stress_1_analytical, stress_2_analytical, max_strain_at_plasticity_start, plasticity_initiation_index, displacement_analytical, max_plastic_strains) = fem_solver(variant, total_elements_default, material_routine, dt_default)


# **Plot displacement at force node over time**
plt.figure(figsize=(10, 6))
plt.plot(time, displacements_at_force_node, label=f"Numerical Displacement ({material_routine.__name__.replace('_material_routine', '').capitalize()} Model)", color='blue')
plt.plot(time_analytical, displacement_analytical, label="Analytical Displacement", linestyle='dashed', color='purple')
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.title("Displacement at Force Node vs Time")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(max_strains, max_stresses, label=f"Numerical Stress-Strain ({material_routine.__name__.replace('_material_routine', '').capitalize()} Model)", color='red')
plt.plot(strain_1_analytical, stress_1_analytical, label="Analytical Segment 1 (Tension/Compression)", linestyle='dashed', color='blue')
plt.plot(strain_2_analytical, stress_2_analytical, label="Analytical Segment 2 (Tension/Compression)", linestyle='dashed', color='green')
if material_routine == viscoplastic_material_routine:
    plt.axvline(x=max_strain_at_plasticity_start, color='pink', linestyle='--', label="Plasticity Initiation Point")
plt.xlabel("Strain")
plt.ylabel("Stress (Pa)")
plt.title("Stress vs Strain: Numerical vs Analytical")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(time, max_plastic_strains, label="Max Plastic Strain", color='red')
plt.xlabel("Time (s)")
plt.ylabel("Max Plastic Strain")
plt.title("Maximum Plastic Strain Evolution Over Time")
plt.grid()
plt.legend()
plt.show()

# Ask if user wants to perform convergence analysis
perform_convergence = input("Perform Convergence Analysis? (yes/no): ").strip().lower()

if perform_convergence == "yes":
    dt_levels = [0.1, 0.01, 0.001, 0.0001]
    element_levels = [5, 25, 625, 3125]

    time_step_errors = []
    mesh_errors = []

      
    print("\nRunning Time Step Convergence Analysis...")
    for dt in dt_levels:
        (time, displacements, time_analytical, displacement_analytical_at_plasticity_start,
         max_strains, max_stresses, strain_1_analytical, strain_2_analytical,
         stress_1_analytical, stress_2_analytical, max_strain_at_plasticity_start,
         plasticity_initiation_index, displacement_analytical, max_plastic_strains) = fem_solver(variant, total_elements_default, material_routine, dt)
    
        # Compute relative error
        error_array = np.abs(np.array(displacements[:plasticity_initiation_index]) - np.array(displacement_analytical[:plasticity_initiation_index])) 
        numerical_tolerance = 1e-8  # Small threshold to avoid division by zero
        relative_error = error_array / np.maximum(np.abs(np.array(displacement_analytical[:plasticity_initiation_index])), numerical_tolerance)

        avg_error = (np.mean(relative_error[1:])) * 100 # Skip first step
        time_step_errors.append(avg_error)


        
    # Time Step Convergence Plot
    plt.figure(figsize=(10, 6))
    plt.plot(dt_levels, time_step_errors, marker='o', linestyle='-', color='blue')
    plt.xlabel("Time Step (dt)")
    plt.ylabel("Error in Displacement (%)")
    plt.title("Time Step Convergence Based on Error")   
    plt.xscale('log')
    plt.yscale('log', nonpositive='clip')   
    plt.grid(True)
    plt.gca().invert_xaxis()
    plt.xticks(dt_levels, labels=[str(dt) for dt in dt_levels])  # Ensure labels appear
    plt.yticks(time_step_errors, labels=[f"{dt:.2f}" for dt in time_step_errors])  # Ensure labels appear
    plt.show()


    print("\nRunning Mesh (h) Convergence Analysis...")
    for elements in element_levels:
        (time, displacements, time_analytical, displacement_analytical_at_plasticity_start,
         max_strains, max_stresses, strain_1_analytical, strain_2_analytical,
         stress_1_analytical, stress_2_analytical, max_strain_at_plasticity_start,
         plasticity_initiation_index, displacement_analytical, max_plastic_strains) = fem_solver(variant, elements, material_routine, dt_default)
 
        
        # Compute relative error
        error_array = np.abs(np.array(displacements[:plasticity_initiation_index]) - np.array(displacement_analytical[:plasticity_initiation_index])) 
        numerical_tolerance = 1e-8  # Small threshold to avoid division by zero
        relative_error = error_array / np.maximum(np.abs(np.array(displacement_analytical[:plasticity_initiation_index])), numerical_tolerance)

        avg_error = (np.mean(relative_error[1:])) * 100 # Skip first step
        mesh_errors.append(avg_error)

    # Mesh Convergence Plot
    plt.figure(figsize=(10, 6))
    plt.plot(element_levels, mesh_errors, marker='o', linestyle='-', color='red')
    plt.xlabel("Number of Elements")
    plt.ylabel("Error in Displacement (%)")
    plt.title("Mesh (h) Convergence Based on Error")
    plt.xscale('log')
    plt.yscale('log', nonpositive='clip')
    plt.yticks(mesh_errors, labels=[f"{error:.3f}" for error in mesh_errors])
    plt.xticks(element_levels, labels=[str(el) for el in element_levels])
    plt.grid(True, which="major", linestyle="--", linewidth=0.5)
    plt.grid(False, which="minor")
    plt.show()
    
    
else:

    # ✅ Ensuring script does not execute further after convergence
    print("\n✅ FEM Solver Execution Completed. No further actions required.")
