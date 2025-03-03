import numpy as np
import matplotlib.pyplot as plt
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
    return internal_forces

# Function to Plot Internal Forces
def plot_internal_forces(Elements, internal_forces):
    for i, elem in enumerate(Elements):
        x = np.linspace(0, elem.L, 10)
        axial_force = np.full_like(x, internal_forces[i][0])
        shear_force_y = np.full_like(x, internal_forces[i][1])
        shear_force_z = np.full_like(x, internal_forces[i][2])
        plt.figure()
        plt.plot(x, axial_force, label='Axial Force')
        plt.plot(x, shear_force_y, label='Shear Force Y')
        plt.plot(x, shear_force_z, label='Shear Force Z')
        plt.xlabel("Position along Element (m)")
        plt.ylabel("Force (N)")
        plt.title(f"Internal Forces for Element {i}")
        plt.legend()
        plt.show()

# Function to Plot Deformed Structure
def plot_deformed_shape(nodes, displacements, scale=10):
    deformed_nodes = np.array([[node.x + scale * displacements[node.id * 6],
                                 node.y + scale * displacements[node.id * 6 + 1],
                                 node.z + scale * displacements[node.id * 6 + 2]] for node in nodes])
    original_nodes = np.array([[node.x, node.y, node.z] for node in nodes])
    plt.figure()
    plt.plot(original_nodes[:, 0], original_nodes[:, 2], 'bo-', label="Original Structure")
    plt.plot(deformed_nodes[:, 0], deformed_nodes[:, 2], 'ro-', label="Deformed Shape")
    plt.xlabel("X Position (m)")
    plt.ylabel("Z Position (m)")
    plt.legend()
    plt.title("Deformed Shape of Structure")
    plt.show()

# Elastic Critical Load Analysis
def elastic_critical_load_analysis(Elements, displacements):
    critical_loads = []
    for elem in Elements:
        k_g = local_geometric_stiffness_matrix_3D_beam(elem.L, elem.A, elem.Ip, *displacements[:6])
        eigvals = np.linalg.eigvals(k_g)
        critical_loads.append(min(eigvals))
    return critical_loads

# Running the Updated Solver
nodes = [Node(0, 0, 0, 10), Node(1, 15, 0, 10), Node(2, 15, 0, 0)]
elements = [Element(nodes[0], nodes[1], 1000, 0.3, 0.5, 0.02, 0.02, 0.02, 0.02861, [0, 0, 1]),
            Element(nodes[1], nodes[2], 1000, 0.3, 0.5, 0.02, 0.02, 0.02, 0.02861, [1, 0, 0])]
load = np.array([0, 0, 0, 0, 0, 0, 0.1, 0.05, -0.07, 0.05, -0.1, 0.25, 0, 0, 0, 0, 0, 0])
displacements = np.random.random(18)  # Placeholder for actual solver results

internal_forces = compute_internal_forces(elements, displacements)
plot_internal_forces(elements, internal_forces)
plot_deformed_shape(nodes, displacements)
critical_loads = elastic_critical_load_analysis(elements, displacements)
print("Elastic Critical Loads:", critical_loads)
