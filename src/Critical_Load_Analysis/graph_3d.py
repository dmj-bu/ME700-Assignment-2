import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Critical_Load_Analysis.criticalloadanalysis_tutorial import (
    critical_load_analysis, structure_solver, get_problem_setup, main
)

def get_free_dofs(supports):
    """Returns an array of free DOF indices based on support conditions."""
    free_dofs = []
    for i, node_support in enumerate(supports):
        for j in range(6):  # 6 DOFs per node
            global_dof = i * 6 + j  # Global DOF index
            if node_support[j + 1] == 0:  # If not fixed, it's free
                free_dofs.append(global_dof)
    return np.array(free_dofs)

def extract_analysis_results():
    """Extracts nodal displacements from linear and buckling analysis."""
    nodes, connection, loads, supports = get_problem_setup()
    disp_linear, _ = structure_solver(nodes, connection, loads, supports)
    eigvals, eigvecs = critical_load_analysis(nodes, connection, loads, supports)
    print("DEBUG: eigenvalues from test:", eigvals)
    print("DEBUG: first few eigenvectors shape:", eigvecs.shape)
    free_dofs = get_free_dofs(supports)

    # Extract smallest positive eigenvalue (Critical Load Factor)
    relevant_eigs = eigvals[eigvals < -1e-3]

    if len(relevant_eigs) == 0:
        # No valid negative eigenvalues => no buckling factor
        lambda_crit = None
    else:
        # Invert and take absolute value
        pos_buckling_eigs = 1 / np.abs(relevant_eigs)
        lambda_crit = np.min(pos_buckling_eigs)

    # Initialize full displacement vectors
    full_disp_buckling = np.zeros(len(nodes) * 6)

    # Map free DOFs from eigvecs into full vector
    full_disp_buckling[free_dofs] = eigvecs[:, 0]  # First buckling mode

    # Reshape to (num_nodes, 6)
    disp_buckling = full_disp_buckling.reshape(len(nodes), 6)

    return nodes, connection, disp_linear, disp_buckling, lambda_crit

def hermite_interpolation(start_disp, end_disp, start_rot, end_rot, num_points=20):
    """Generates a curved buckling shape using Hermite interpolation."""
    t = np.linspace(0, 1, num_points)  # Interpolation parameter

    # Hermite shape functions for beam bending
    N1 = 1 - 3*t**2 + 2*t**3
    N2 = t - 2*t**2 + t**3
    N3 = 3*t**2 - 2*t**3
    N4 = -t**2 + t**3

    # Compute displacement at interpolated points
    disp_interp = (
        N1[:, None] * start_disp +
        N2[:, None] * start_rot +
        N3[:, None] * end_disp +
        N4[:, None] * end_rot
    )

    return disp_interp

def plot_3d_displacements(mode_type="linear", scale_factor=5):
    """Plots the deformed structure in 3D using correct eigenvectors."""
    
    # Extract analysis results
    nodes, connection, disp_linear, disp_buckling, lambda_crit = extract_analysis_results()

    # Ensure frame is correctly aligned
    nodes = np.array(nodes)

    # Select displacement mode
    displacements = disp_linear if mode_type == "linear" else disp_buckling

    # Ensure displacement is correctly shaped
    if displacements.ndim == 1:
        displacements = displacements.reshape(len(nodes), 6)

    # Scale displacement for visibility
    displacements[:, :3] *= scale_factor  

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot original structure (black dashed lines)
    for elem in connection:
        start, end = elem[0], elem[1]
        ax.plot(*zip(*nodes[[start, end]]), 'k--', linewidth=1)

    # Plot deformed structure using Hermite interpolation
    for elem in connection:
        start, end = elem[0], elem[1]
        x = np.linspace(nodes[start, 0], nodes[end, 0], 20)
        y = np.linspace(nodes[start, 1], nodes[end, 1], 20)
        z = np.linspace(nodes[start, 2], nodes[end, 2], 20)

        # Extract displacements and rotations
        u1, v1, w1, rx1, ry1, rz1 = displacements[start]
        u2, v2, w2, rx2, ry2, rz2 = displacements[end]

        start_disp = np.array([u1, v1, w1])
        end_disp = np.array([u2, v2, w2])
        start_rot = np.array([rx1, ry1, rz1])
        end_rot = np.array([rx2, ry2, rz2])

        # Compute interpolated curved displacement using Hermite shape functions
        disp_interp = hermite_interpolation(start_disp, end_disp, start_rot, end_rot, num_points=20)

        # Apply deformed shape to original beam
        x_buckled = x + disp_interp[:, 0]
        y_buckled = y + disp_interp[:, 1]
        z_buckled = z + disp_interp[:, 2]

        ax.plot(x_buckled, y_buckled, z_buckled, 'm-', linewidth=2)

    # Label nodes
    for i, (x, y, z) in enumerate(nodes):
        ax.text(x, y, z, f'N{i}', color='red', fontsize=8)

    # Axis labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    title = f"3D Frame - {mode_type.capitalize()} Mode Shape"
    if lambda_crit:
        title += f" (λ = {lambda_crit:.5f})"
    ax.set_title(title)

    # Automatically adjust view based on node positions
    ax.set_xlim(sorted(ax.get_xlim()))
    ax.set_ylim(sorted(ax.get_ylim()))
    ax.set_zlim(sorted(ax.get_zlim()))

    # Adjust viewing angle
    ax.view_init(elev=20, azim=-60)

    plt.show()

if __name__ == "__main__":
    plot_choice = input("Enter 'linear' for static deformation or 'buckling' for mode shape: ").strip().lower()
    if plot_choice in ["linear", "buckling"]:
        plot_3d_displacements(mode_type=plot_choice, scale_factor=5)
    else:
        print("Invalid input. Please enter 'linear' or 'buckling'.")