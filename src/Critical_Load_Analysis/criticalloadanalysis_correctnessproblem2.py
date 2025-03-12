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

    # Nodal positions
    nodes = np.array([
        [0,  0,   0],   # N0 (Fixed Support)
        [3,  28/3,  22/3],   # N1
        [6,  56/3,  44/3],   # N2
        [9,  84/3,  66/3],   # N3
        [12, 112/3, 88/3],   # N4
        [15, 140/3, 110/3],  # N5
        [18, 56, 44]         # N6 (Free End)
    ], dtype=float)

    # Material & section properties
    r = 1  # Radius of circular cross-section
    E = 10000  # Young's Modulus
    nu = 0.3  # Poisson's ratio

    # Computed section properties
    A = np.pi * r**2
    I_y = np.pi * r**4 / 4.0
    I_z = np.pi * r**4 / 4.0
    I_rho = np.pi * r**4 / 2.0
    J = np.pi * r**4 / 2.0

    # Element connectivity:
    connection = np.array([
        [
            i, i+1, E, nu, A, I_y, I_z, I_rho, J, 
            [
                -(nodes[i+1][1] - nodes[i][1]) / np.sqrt((nodes[i+1][0] - nodes[i][0])**2 + (nodes[i+1][1] - nodes[i][1])**2),
                (nodes[i+1][0] - nodes[i][0]) / np.sqrt((nodes[i+1][0] - nodes[i][0])**2 + (nodes[i+1][1] - nodes[i][1])**2),
                0
            ]
        ] 
        for i in range(len(nodes) - 1)
    ], dtype=object)

    # Supports:
    supports = np.array([
        [0, 1,1,1, 1,1,1],  # N0 (Fully fixed)
        [1, 0,0,0, 0,0,0],  # N1 (Free)
        [2, 0,0,0, 0,0,0],  # N2 (Free)
        [3, 0,0,0, 0,0,0],  # N3 (Free)
        [4, 0,0,0, 0,0,0],  # N4 (Free)
        [5, 0,0,0, 0,0,0],  # N5 (Free)
        [6, 0,0,0, 0,0,0]   # N6 (Free end)
    ])

    # External loads:
    # Load Constants
    x1 = 18
    x0 = 0
    y1 = 56
    y0 = 0
    z1 = 44
    z0 = 0
    P = 1
    L = np.sqrt((x1-x0) ** 2.0 + (y1 - y0) ** 2.0 + (z1 - z0) ** 2.0)
    Fx = -1.0 * P * (x1 - x0) / L
    Fy = -1.0 * P * (y1 - y0) / L
    Fz = -1.0 * P * (z1 - z0) / L
    loads = np.zeros((7,6))  # Initialize 7 nodes with zero force
    loads[6] = [Fx, Fy, Fz, 0.0, 0.0, 0.0]  # Apply force & moment at N6

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
