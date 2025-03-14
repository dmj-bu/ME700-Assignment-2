import numpy as np
import scipy.linalg
from Direct_Stiffness_Method.math_utils import (
    local_elastic_stiffness_matrix_3D_beam,
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
# Example Usage
# --------------------------------------------------------------------------------
def get_problem_setup():
    # Nodal positions
    nodes = np.array([
    [0, 0, 0],    # N0
    [10, 0, 0],   # N1
    [10, 20, 0],  # N2
    [0, 20, 0],   # N3
    [0, 0, 25],   # N4
    [10, 0, 25],  # N5
    [10, 20, 25], # N6
    [0, 20, 25]   # N7
    ])

    # Material & section properties
    E = 500
    nu = 0.3
    r = 0.5
    A  = np.pi * r**2
    I_y = np.pi * r**4 / 4.0
    I_z = np.pi * r**4 / 4.0
    I_p = np.pi * r**4 / 2.0
    J   = np.pi * r**4 / 2.0

    # Element connectivity:
    connection = np.array([
        [0, 4, E, nu, A, I_y, I_z, I_p, J, [1,0,0]],  # N0 → N4 (vertical) 
        [1, 5, E, nu, A, I_y, I_z, I_p, J, [1,0,0]],  # N1 → N5 (vertical) 
        [2, 6, E, nu, A, I_y, I_z, I_p, J, [1,0,0]],  # N2 → N6 (vertical) 
        [3, 7, E, nu, A, I_y, I_z, I_p, J, [1,0,0]],  # N3 → N7 (vertical) 
        
        [4, 5, E, nu, A, I_y, I_z, I_p, J, [0,0,1]],  # N4 → N5 (horizontal X) 
        [5, 6, E, nu, A, I_y, I_z, I_p, J, [0,0,1]],  # N5 → N6 (horizontal Y) 
        [6, 7, E, nu, A, I_y, I_z, I_p, J, [0,0,1]],  # N6 → N7 (horizontal X) 
        [7, 4, E, nu, A, I_y, I_z, I_p, J, [0,0,1]]   # N7 → N4 (horizontal Y)
    ], dtype=object)

    # Supports: Node 0 fully fixed, Node 1 pinned
    supports = np.array([
        [0, 1,1,1, 1,1,1],  # N0 fully fixed
        [1, 1,1,1, 1,1,1],  # N1 fully fixed
        [2, 1,1,1, 1,1,1],  # N2 fully fixed
        [3, 1,1,1, 1,1,1],  # N3 fully fixed
        [4, 0,0,0, 0,0,0],  # N4 is free
        [5, 0,0,0, 0,0,0],  # N5 is free
        [6, 0,0,0, 0,0,0],  # N6 is free
        [7, 0,0,0, 0,0,0]   # N7 is free
    ])

    # External loads: a single compressive load along the bar from Node1->Node0
    loads = np.zeros((8,6))
    loads[4] = [0, 0, -1, 0, 0, 0]  # N4
    loads[5] = [0, 0, -1, 0, 0, 0]  # N5
    loads[6] = [0, 0, -1, 0, 0, 0]  # N6
    loads[7] = [0, 0, -1, 0, 0, 0]  # N7

    return nodes, connection, loads, supports

def main():
    nodes, connection, loads, supports = get_problem_setup()
    displacements, reactions = structure_solver(nodes, connection, loads, supports)

    print("Nodal Displacements & Rotations:")
    for i in range(len(nodes)):
        ux, uy, uz, rx, ry, rz = displacements[6*i : 6*i+6]
        print(f" Node {i}: U=({ux:.6e}, {uy:.6e}, {uz:.6e}), R=({rx:.6e}, {ry:.6e}, {rz:.6e})")

    print("\nReaction Forces & Moments at Constrained Nodes:")
    for i in range(len(nodes)):
        if any(supports[i,1:] == 1):
            fx, fy, fz, mx, my, mz = reactions[6*i : 6*i+6]
            print(f" Node {i}: F=({fx:.6e}, {fy:.6e}, {fz:.6e}), M=({mx:.6e}, {my:.6e}, {mz:.6e})")


if __name__ == "__main__":
    main()