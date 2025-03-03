import numpy as np
from math_utils import local_elastic_stiffness_matrix_3D_beam, transformation_matrix_3D, rotation_matrix_3D

# Node Class
class Node:
    def __init__(self, node_id, x, y, z, F=None):
        self.id = node_id
        self.x = x
        self.y = y
        self.z = z
        self.F = np.zeros(6) if F is None else np.array(F)
        self.supported_dofs = []
    
    def set_support(self, support):
        self.supported_dofs = [i for i, val in enumerate(support) if val]

# Element Class
class Element:
    def __init__(self, node_start, node_end, E, nu, A, Iz, Iy, Ip, J, z_axis):
        self.node_start = node_start
        self.node_end = node_end
        self.E = E
        self.nu = nu
        self.A = A
        self.Iz = Iz
        self.Iy = Iy
        self.Ip = Ip
        self.J = J
        self.z = np.array(z_axis)
        self.L = np.sqrt((node_start.x - node_end.x)**2 + (node_start.y - node_end.y)**2 + (node_start.z - node_end.z)**2)
        if self.L == 0:
            raise ValueError("Nodes Repeated")

        # Local Stiffness Matrix
        self.k_e = local_elastic_stiffness_matrix_3D_beam(self.E, self.nu, self.A, self.L, self.Iy, self.Iz, self.J)
        self.gamma = rotation_matrix_3D(self.node_start.x, self.node_start.y, self.node_start.z, self.node_end.x, self.node_end.y, self.node_end.z, self.z)
        self.Gamma = transformation_matrix_3D(self.gamma)
        self.k_global = self.Gamma.T @ self.k_e @ self.Gamma

# Structure Solver
def structure(nodes, connection, load, supports):
    Nodes = [Node(i, nodes[i][0], nodes[i][1], nodes[i][2], load[i]) for i in range(len(nodes))]
    for support in supports:
        node_index = support[0]
        Nodes[node_index].set_support(support[1:])
    
    Elements = [Element(Nodes[connection[i][0]], Nodes[connection[i][1]], *connection[i][2:]) for i in range(len(connection))]
    
    total_dofs = 6 * len(Nodes)
    k_global = np.zeros((total_dofs, total_dofs))
    for elem in Elements:
        dof_indices = np.array([elem.node_start.id * 6 + j for j in range(6)] + [elem.node_end.id * 6 + j for j in range(6)])
        for i in range(12):
            for j in range(12):
                k_global[dof_indices[i], dof_indices[j]] += elem.k_global[i, j]
    
    # Apply Boundary Conditions
    fixed_dofs = []
    for support in supports:
        node_id = support[0]
        dof_start = node_id * 6
        for i, is_fixed in enumerate(support[1:]):
            if is_fixed:
                fixed_dofs.append(dof_start + i)
    free_dofs = np.setdiff1d(np.arange(total_dofs), fixed_dofs)
    
    k_uu = k_global[np.ix_(free_dofs, free_dofs)]
    f_u = np.concatenate([node.F for node in Nodes]).flatten()[free_dofs]
    
    # Ensure the matrix is not singular
    if np.linalg.cond(k_uu) > 1e12:
        raise ValueError("Singular matrix detected. Check boundary conditions.")
    
    # Solve for Displacements
    del_u = np.linalg.solve(k_uu, f_u)
    del_f = np.zeros(total_dofs)
    del_f[free_dofs] = del_u
    f_all = k_global @ del_f
    
    return del_f, f_all

# Define Inputs
# Define Material Properties
E, nu = 1000, 0.3
b, h = 0.5, 1.0
A = b * h
I_y = h * b ** 3 / 12
I_z = b * h ** 3 / 12
I_rho = b * h / 12 * (b**2 + h**2) 
J = 0.02861
# Define nodes
nodes = np.array([
    [0, 0, 10],  
    [15, 0, 10], 
    [15, 0, 0]  
])
# Define Element Connections
connection = np.array([
    [0, 1, E, nu, b, I_y, I_z, I_rho, J, [0, 0, 1]], # Element 0 between Node 0 and 1
    [1, 2, E, nu, b, I_y, I_z, I_rho, J, [1, 0, 0]]  # Element 1 between Node 1 and 2
], dtype=object)
# Define Boundary Constraints
supports = np.array([
    [0, 1, 1, 1, 1, 1, 1],  # Node 0 is fully fixed (all DOFs constrained)
    [1, 0, 0, 0, 0, 0, 0],  # Node 1 is free (all DOFs unconstrained)
    [2, 1, 1, 1, 0, 0, 0]   # Node 2 is pinned (constraining translations, but not rotations)
])

load = np.array([
    [0, 0, 0, 0, 0, 0],
    [0.1, 0.05, -0.07, 0.05, -0.1, 0.25],
    [0, 0, 0, 0, 0, 0]])

displacement, forces = structure(nodes, connection, load, supports)

print("Computed Displacements: ", displacement)
print("Reaction Forces: ", forces)
