import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math_utils import (local_elastic_stiffness_matrix_3D_beam,
                        transformation_matrix_3D,
                        rotation_matrix_3D,
                        local_geometric_stiffness_matrix_3D_beam)

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

# Function to Compute Internal Forces in Local Coordinates
def compute_internal_forces(Elements, displacements):
    internal_forces = []
    for elem in Elements:
        dof_indices = np.array([elem.node_start.id * 6 + j for j in range(6)] + [elem.node_end.id * 6 + j for j in range(6)])
        local_disp = displacements[dof_indices]
        f_local = elem.k_e @ local_disp
        internal_forces.append(f_local)
    return np.array(internal_forces)

# Function to Plot Deformed Structure in 3D
def plot_deformed_shape_3D(nodes, displacements, scale=10):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for node in nodes:
        dx = scale * displacements[node.id * 6]
        dy = scale * displacements[node.id * 6 + 1]
        dz = scale * displacements[node.id * 6 + 2]
        ax.scatter(node.x, node.y, node.z, color='b', label='Original' if node.id == 0 else "")
        ax.scatter(node.x + dx, node.y + dy, node.z + dz, color='r', label='Deformed' if node.id == 0 else "")
        ax.plot([node.x, node.x + dx], [node.y, node.y + dy], [node.z, node.z + dz], 'k--')
    
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")
    ax.set_title("Deformed Shape of Structure")
    plt.legend()
    plt.show()

# Elastic Critical Load Analysis
def elastic_critical_load_analysis(Elements, displacements):
    critical_loads = []
    for elem in Elements:
        k_g = local_geometric_stiffness_matrix_3D_beam(elem.L, elem.A, elem.Ip, *displacements[:6])
        eigvals = np.linalg.eigvals(k_g)
        critical_loads.append(min(eigvals))
    return np.array(critical_loads)

# Define Inputs
E, nu = 1000, 0.3
b, h = 0.5, 1.0
A = b * h
I_y = h * b ** 3 / 12
I_z = b * h ** 3 / 12
I_rho = b * h / 12 * (b**2 + h**2) 
J = 0.02861
nodes = [Node(0, 0, 0, 10), Node(1, 15, 0, 10), Node(2, 15, 0, 0)]
elements = [Element(nodes[0], nodes[1], E, nu, A, I_y, I_z, I_rho, J, [0, 0, 1]),
            Element(nodes[1], nodes[2], E, nu, A, I_y, I_z, I_rho, J, [1, 0, 0])]
supports = [[0, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0], [2, 1, 1, 1, 0, 0, 0]]
load = np.array([[0, 0, 0, 0, 0, 0], [0.1, 0.05, -0.07, 0.05, -0.1, 0.25], [0, 0, 0, 0, 0, 0]])

displacements = np.random.random(18)  # Placeholder for actual solver results
reaction_forces = compute_internal_forces(elements, displacements)
critical_loads = elastic_critical_load_analysis(elements, displacements)

plot_deformed_shape_3D(nodes, displacements)

print("Computed Displacements:", displacements.reshape(-1, 6))
print("Reaction Forces:", reaction_forces)
print("Elastic Critical Loads:", critical_loads)
