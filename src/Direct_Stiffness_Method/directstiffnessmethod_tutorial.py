import numpy as np
from Direct_Stiffness_Method.math_utils import (
    local_elastic_stiffness_matrix_3D_beam,
    transformation_matrix_3D,
    rotation_matrix_3D
)

# Node and Element classes

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
        # Indices for the 2 nodes (each has 6 DOFs)
        dofs = np.array([6*elem.node_start.id + i for i in range(6)]
                      + [6*elem.node_end.id   + i for i in range(6)])
        # Add element matrix
        for i in range(12):
            for j in range(12):
                K_global[dofs[i], dofs[j]] += elem.k_global[i, j]

    # Identify constrained DOFs
    fixed_dofs = []
    for sup in supports:
        base_id = sup[0] * 6
        # For each DOF in {0..5}, check if sup[i+1]==1
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


# Define example geometry, properties, BCs

# Nodal positions
nodes = np.array([
    [ 0.0,  0.0,  0.0],   # N0
    [30.0,  40.0, 0.0],   # N1
])

# Material & section properties
E  = 1000
nu = 0.3
r  = 1.0
A  = np.pi * r**2           # cross‐sectional area
I_y = np.pi * r**4 / 4.0     # for a circular section
I_z = np.pi * r**4 / 4.0
I_p = np.pi * r**4 / 2.0     # polar moment = I_y + I_z for a circle
J  = np.pi * r**4 / 2.0     # torsional constant

# Element connectivity:
connection = np.array([
    [0, 1, E, nu, A, I_y, I_z, I_p, J, [0,0,1]],  # E0
], dtype=object)

# Supports:
supports = np.array([
    [0, 1,1,1, 1,1,1],  # Node3 fully fixed
    [1, 0,0,0, 0,0,0],  # Node4 pinned: fix translations, free rotations
])

# External loads:
loads = np.zeros((2,6))
loads[1] = [-3/5, -4/5, 0.0, 0.0, 0.0, 0.0]

# Solve and show results

displacements, reactions = structure_solver(nodes, connection, loads, supports)

print("Nodal Displacements & Rotations:")
for i in range(len(nodes)):
    ux, uy, uz, rx, ry, rz = displacements[6*i : 6*i+6]
    print(f"  Node {i}: U=({ux:.6e}, {uy:.6e}, {uz:.6e}),"
          f" R=({rx:.6e}, {ry:.6e}, {rz:.6e})")

print("\nReaction Forces & Moments at Constrained Nodes:")
for i in range(len(nodes)):
    # If any DOF at node i is fixed, the code will produce a reaction
    if any(supports[i,1:] == 1):
        fx, fy, fz, mx, my, mz = reactions[6*i : 6*i+6]
        print(f"  Node {i}: F=({fx:.6e}, {fy:.6e}, {fz:.6e}),"
              f" M=({mx:.6e}, {my:.6e}, {mz:.6e})")
