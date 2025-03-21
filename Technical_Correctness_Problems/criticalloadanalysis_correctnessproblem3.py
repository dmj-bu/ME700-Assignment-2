import numpy as np
import scipy.linalg
from Direct_Stiffness_Method.math_utils import (
    local_elastic_stiffness_matrix_3D_beam,
    local_geometric_stiffness_matrix_3D_beam,
    transformation_matrix_3D,
    rotation_matrix_3D
)

# --------------------------------------------------------------------------------
# Node / Element classes, structure_solver, etc.
# --------------------------------------------------------------------------------

class Node:
    def __init__(self, node_id, x, y, z, F=None):
        self.id = node_id
        self.x = x
        self.y = y
        self.z = z
        self.F = np.zeros(6) if F is None else np.array(F)
        self.supported_dofs = []
    
    def set_support(self, support):
        # True/1 → this DOF is fixed
        self.supported_dofs = [i for i, val in enumerate(support) if val]

class Element:
    def __init__(self, node_start, node_end, E, nu, A, Iy, Iz, Ip, J, z_axis):
        self.node_start = node_start
        self.node_end = node_end
        self.E = E
        self.nu = nu
        self.A = A
        self.Iy = Iy
        self.Iz = Iz
        self.Ip = Ip  # polar moment of area
        self.J = J    # torsional constant
        self.z_axis = np.array(z_axis)

        # Element length
        self.L = np.sqrt((node_start.x - node_end.x)**2
                         + (node_start.y - node_end.y)**2
                         + (node_start.z - node_end.z)**2)
        if self.L == 0:
            raise ValueError("Nodes coincide – zero length element.")

        # Local elastic stiffness (3D Bernoulli beam)
        self.k_e = local_elastic_stiffness_matrix_3D_beam(
            E=self.E,
            nu=self.nu,
            A=self.A,
            L=self.L,
            Iy=self.Iy,
            Iz=self.Iz,
            J=self.J
        )
        # Build rotation from global->local
        self.gamma = rotation_matrix_3D(
            node_start.x, node_start.y, node_start.z,
            node_end.x,   node_end.y,   node_end.z,
            self.z_axis
        )
        self.Gamma = transformation_matrix_3D(self.gamma)

        # Element stiffness in global coords
        self.k_global = self.Gamma.T @ self.k_e @ self.Gamma

def structure_solver(nodes, connection, loads, supports):
    """Assemble global K, apply boundary conditions, solve for displacements."""
    
    # Build Node objects
    node_objs = []
    for i, (x, y, z) in enumerate(nodes):
        node_objs.append(Node(i, x, y, z, F=loads[i]))

    # Apply each support
    for sup in supports:
        node_id = sup[0]
        node_objs[node_id].set_support(sup[1:])

    # Build Elements
    elem_objs = []
    for row in connection:
        n_start, n_end = row[0], row[1]
        E_, nu_, A_, Iy_, Iz_, Ip_, J_, z_ax_ = row[2:]
        elem_objs.append(
            Element(node_objs[n_start], node_objs[n_end],
                    E_, nu_, A_, Iy_, Iz_, Ip_, J_, z_ax_)
        )

    # Global stiffness
    ndof = 6 * len(node_objs)
    K_global = np.zeros((ndof, ndof))
    for elem in elem_objs:
        dofs = np.array([6*elem.node_start.id + i for i in range(6)]
                      + [6*elem.node_end.id   + i for i in range(6)])
        for i_local in range(12):
            for j_local in range(12):
                K_global[dofs[i_local], dofs[j_local]] += elem.k_global[i_local, j_local]

    # Identify constrained DOFs
    fixed_dofs = []
    for sup in supports:
        base_id = sup[0] * 6
        for dof_local, is_fixed in enumerate(sup[1:]):
            if is_fixed == 1:
                fixed_dofs.append(base_id + dof_local)

    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)

    # Build global load vector
    F_global = np.concatenate([n.F for n in node_objs])

    # Partition K and F to solve
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    F_f  = F_global[free_dofs]

    # Solve the reduced system
    disp_free = np.linalg.solve(K_ff, F_f)

    # Insert back into full displacement vector
    disp_all = np.zeros(ndof)
    disp_all[free_dofs] = disp_free

    # Reaction = K_global @ disp_all
    reac_all = K_global @ disp_all

    return disp_all, reac_all

# --------------------------------------------------------------------------------
# Critical Load Analysis
# --------------------------------------------------------------------------------

def critical_load_analysis(nodes, connection, loads, supports):
    """
    1) Calls the standard structure_solver to get displacements (linear).
    2) From those displacements, computes element end‐forces.
    3) Assembles the global geometric stiffness matrix K_g.
    4) Applies the same boundary conditions (BCs) to K and K_g.
    5) Solves the generalized eigenvalue problem for critical load factor(s).
    
    Returns:
        eigvals (np.ndarray): The array of eigenvalues (load factors).
        eigvecs (np.ndarray): The corresponding eigenvectors (in free DOFs).
    """
 # 1) Linear solve
    disp_all, reac_all = structure_solver(nodes, connection, loads, supports)

    # 2) Rebuild global K and element objects
    ndof = 6 * len(nodes)
    K_global = np.zeros((ndof, ndof))

    node_objs = []
    for i, (x, y, z) in enumerate(nodes):
        node_objs.append(Node(i, x, y, z, F=loads[i]))
    for sup in supports:
        node_objs[sup[0]].set_support(sup[1:])

    elem_objs = []
    for row in connection:
        n_start, n_end = row[0], row[1]
        E_, nu_, A_, Iy_, Iz_, Ip_, J_, z_ax_ = row[2:]
        elem_objs.append(
            Element(node_objs[n_start], node_objs[n_end],
                    E_, nu_, A_, Iy_, Iz_, Ip_, J_, z_ax_)
        )

    for elem in elem_objs:
        dofs = np.array([6*elem.node_start.id + i for i in range(6)]
                      + [6*elem.node_end.id   + i for i in range(6)])
        for i_local in range(12):
            for j_local in range(12):
                K_global[dofs[i_local], dofs[j_local]] += elem.k_global[i_local, j_local]
    
    # Identify fixed/free DOFs again
    fixed_dofs = []
    for sup in supports:
        base_id = sup[0] * 6
        for dof_local, is_fixed in enumerate(sup[1:]):
            if is_fixed == 1:
                fixed_dofs.append(base_id + dof_local)
    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)

    # 3) Compute each element’s internal end‐forces in local coords
    elem_forces_local = []
    for elem in elem_objs:
        edofs = np.array([6*elem.node_start.id + i for i in range(6)]
                       + [6*elem.node_end.id   + i for i in range(6)])
        elem_disp_global = disp_all[edofs]
        d_local = elem.Gamma @ elem_disp_global
        f_local = elem.k_e @ d_local

        # Must match local DOF ordering
        Fx2 = f_local[6]
        Mx2 = f_local[9]   # torsion at end 2
        My1 = f_local[4]   # strong‐axis moment at end 1
        Mz1 = f_local[5]   # weak‐axis moment at end 1
        My2 = f_local[10]  
        Mz2 = f_local[11]

        elem_forces_local.append((Fx2, Mx2, My1, Mz1, My2, Mz2))

    # 4) Build the global geometric stiffness matrix
    K_geo_global = np.zeros((ndof, ndof))
    for (elem, fvals) in zip(elem_objs, elem_forces_local):
        Fx2, Mx2, My1, Mz1, My2, Mz2 = fvals
        
        k_g_local = local_geometric_stiffness_matrix_3D_beam(
            L=elem.L,
            A=elem.A,
            I_rho=elem.Ip,  # polar moment
            Fx2=Fx2, Mx2=Mx2,
            My1=My1, Mz1=Mz1,
            My2=My2, Mz2=Mz2
        )
        k_g_global_elem = elem.Gamma.T @ k_g_local @ elem.Gamma

        dofs = np.array([6*elem.node_start.id + i for i in range(6)]
                      + [6*elem.node_end.id   + i for i in range(6)])
        for i_local in range(12):
            i_glob = dofs[i_local]
            for j_local in range(12):
                j_glob = dofs[j_local]
                K_geo_global[i_glob, j_glob] += k_g_global_elem[i_local, j_local]

    # 5) Partition for free DOFs, solve the single‐argument eigenproblem
    K_ff   = K_global[np.ix_(free_dofs, free_dofs)]
    Kg_ff  = K_geo_global[np.ix_(free_dofs, free_dofs)]

    # We want to solve (K_ff + λ K_g_ff) v = 0 => K_g_ff v = -λ K_ff v => K_g_ff v = μ K_ff v with μ = -λ.
    # So we do A = -K_ff^-1 @ Kg_ff, and solve A v = λ v. Then each λ = the negative of μ => is the negative of
    # the buckling λ in the original equation. We'll pick out the positive ones at the end.
    A = -np.linalg.pinv(K_ff) @ (Kg_ff * np.max(np.abs(K_ff)) / np.max(np.abs(Kg_ff)))


    eigvals_raw, eigvecs_raw = scipy.linalg.eig(Kg_ff, K_ff)
    # print("Raw Eigenvalues:", eigvals_raw) # Use if needed

    # Extract real parts, sort ascending
    eigvals = np.real(eigvals_raw)
    eigvecs = np.real(eigvecs_raw)
    idx_sort = np.argsort(eigvals)
    eigvals = eigvals[idx_sort]
    eigvecs = eigvecs[:, idx_sort]

    return eigvals, eigvecs


# --------------------------------------------------------------------------------
# Example Usage
# --------------------------------------------------------------------------------
def get_problem_setup():
    # Length Constants
    L1 = 15.0
    L2 = 30.0
    L3 = 14.0
    L4 = 16.0
    # Nodal positions
    nodes = np.array([
        [0, 0, 0],          # N0
        [L1, 0, 0],         # N1
        [L1, L2, 0],        # N2
        [0, L2, 0],         # N3
        [0, 0, L3],         # N4
        [L1, 0, L3],        # N5
        [L1, L2, L3],       # N6
        [0, L2, L3],        # N7
        [0, 0, L3+L4],      # N8
        [L1, 0, L3+L4],     # N9
        [L1, L2, L3+L4],    # N10
        [0, L2, L3+L4]      # N11
    ])


    # Material & section properties
    # Element Type A Properties
    E_a = 10000
    nu_a = 0.3
    r_a = 1
    A_a = np.pi * r_a**2
    I_y_a = np.pi * r_a**4 / 4.0
    I_z_a = np.pi * r_a**4 / 4.0
    I_rho_a = np.pi * r_a**4 / 2.0
    J_a = np.pi * r_a**4 / 2.0
    # Element Type B Properties
    E_b = 50000
    nu_b = 0.3
    b = 0.5
    h = 1
    A_b = b * h
    I_y_b = (b * h**3) / 12.0
    I_z_b = (h * b**3) / 12.0
    I_rho_b = (b * h) / 12.0 * (b**2 + h**2)
    J_b = 0.028610026401666667


    # Element connectivity:
    connection = np.array([
        # Vertical elements (Type A)
        [0, 4, E_a, nu_a, A_a, I_y_a, I_z_a, I_rho_a, J_a, [1,0,0]],
        [1, 5, E_a, nu_a, A_a, I_y_a, I_z_a, I_rho_a, J_a, [1,0,0]],
        [3, 7, E_a, nu_a, A_a, I_y_a, I_z_a, I_rho_a, J_a, [1,0,0]],
        [2, 6, E_a, nu_a, A_a, I_y_a, I_z_a, I_rho_a, J_a, [1,0,0]],
        [4, 8, E_a, nu_a, A_a, I_y_a, I_z_a, I_rho_a, J_a, [1,0,0]],
        [5, 9, E_a, nu_a, A_a, I_y_a, I_z_a, I_rho_a, J_a, [1,0,0]],
        [7, 11, E_a, nu_a, A_a, I_y_a, I_z_a, I_rho_a, J_a, [1,0,0]],
        [6, 10, E_a, nu_a, A_a, I_y_a, I_z_a, I_rho_a, J_a, [1,0,0]],


        # Horizontal elements (Type B)
        [4, 5, E_b, nu_b, A_b, I_y_b, I_z_b, I_rho_b, J_b, [0,0,1]],  
        [5, 6, E_b, nu_b, A_b, I_y_b, I_z_b, I_rho_b, J_b, [0,0,1]],  
        [6, 7, E_b, nu_b, A_b, I_y_b, I_z_b, I_rho_b, J_b, [0,0,1]],  
        [7, 4, E_b, nu_b, A_b, I_y_b, I_z_b, I_rho_b, J_b, [0,0,1]],  
        [8, 9, E_b, nu_b, A_b, I_y_b, I_z_b, I_rho_b, J_b, [0,0,1]],  
        [9, 10, E_b, nu_b, A_b, I_y_b, I_z_b, I_rho_b, J_b, [0,0,1]],  
        [10, 11, E_b, nu_b, A_b, I_y_b, I_z_b, I_rho_b, J_b, [0,0,1]],  
        [11, 8, E_b, nu_b, A_b, I_y_b, I_z_b, I_rho_b, J_b, [0,0,1]],
    ], dtype=object)


    # Supports:
    supports = np.array([
        [0, 1,1,1, 1,1,1],  # N0 fully fixed
        [1, 1,1,1, 1,1,1],  # N1 fully fixed
        [2, 1,1,1, 1,1,1],  # N2 fully fixed
        [3, 1,1,1, 1,1,1],  # N3 fully fixed
        [4, 0,0,0, 0,0,0],  # N4 is free
        [5, 0,0,0, 0,0,0],  # N5 is free
        [6, 0,0,0, 0,0,0],  # N6 is free
        [7, 0,0,0, 0,0,0],   # N7 is free
        [8, 0,0,0, 0,0,0],  # N8 is free
        [9, 0,0,0, 0,0,0],  # N9 is free
        [10, 0,0,0, 0,0,0],  # N10 is free
        [11, 0,0,0, 0,0,0]   # N11 is free
    ])


    # External loads:
    loads = np.zeros((12,6))
    loads[8] = [0, 0, -1, 0, 0, 0]  # N8
    loads[9] = [0, 0, -1, 0, 0, 0]  # N9
    loads[10] = [0, 0, -1, 0, 0, 0] # N10
    loads[11] = [0, 0, -1, 0, 0, 0] # N11


    return nodes, connection, loads, supports

if __name__ == "__main__":

    nodes, connection, loads, supports = get_problem_setup()

    # Solve linear displacements
    displacements, reactions = structure_solver(nodes, connection, loads, supports)

    print("Nodal Displacements & Rotations:")
    for i in range(len(nodes)):
        ux, uy, uz, rx, ry, rz = displacements[6*i : 6*i+6]
        print(f"  Node {i}: U=({ux:.6e}, {uy:.6e}, {uz:.6e}),"
              f" R=({rx:.6e}, {ry:.6e}, {rz:.6e})")

    print("\nReaction Forces & Moments at Constrained Nodes:")
    for i in range(len(nodes)):
        if any(supports[i,1:] == 1):
            fx, fy, fz, mx, my, mz = reactions[6*i : 6*i+6]
            print(f"  Node {i}: F=({fx:.6e}, {fy:.6e}, {fz:.6e}),"
                  f" M=({mx:.6e}, {my:.6e}, {mz:.6e})")

    # Critical Load Analysis
    eigvals, eigvecs = critical_load_analysis(nodes, connection, loads, supports)

    # The smallest positive eigenvalue is your critical factor (since we
    # have λ on the left side in (K_ff + λK_g_ff) = 0, we look for +λ).
    positive_eigs = 1 / np.abs(eigvals[eigvals < -1e-3])
    if len(positive_eigs) == 0:
        print("No positive buckling eigenvalues found.")
    else:
        lambda_crit = np.min(positive_eigs)  # First meaningful eigenvalue
        print(f"Critical Load Factor = {lambda_crit:.5f}")
