import numpy as np
from math_utils import rotation_matrix_3D, transformation_matrix_3D, local_elastic_stiffness_matrix_3D_beam

class Node:
    def __init__(self, node_id, x, y, z):
        self.id = node_id
        self.x = x
        self.y = y
        self.z = z

class BoundaryCondition:
    def __init__(self, node_id, ux, uy, uz, rx, ry, rz):
        self.node_id = node_id
        self.ux = ux
        self.uy = uy
        self.uz = uz
        self.rx = rx
        self.ry = ry
        self.rz = rz

class Load:
    def __init__(self, node_id, fx=0.0, fy=0.0, fz=0.0, mx=0.0, my=0.0, mz=0.0):
        self.node_id = node_id
        self.fx = fx
        self.fy = fy
        self.fz = fz
        self.mx = mx
        self.my = my
        self.mz = mz

class Element:
    def __init__(self, element_id, node_start, node_end, E, nu, A, I_y, I_z, J):
        self.id = element_id
        self.node_start = node_start
        self.node_end = node_end
        self.E = E
        self.nu = nu
        self.A = A
        self.I_y = I_y
        self.I_z = I_z
        self.J = J

    def length(self, nodes):
        s, e = nodes[self.node_start], nodes[self.node_end]
        return np.sqrt((e.x - s.x) ** 2 + (e.y - s.y) ** 2 + (e.z - s.z) ** 2)

    def global_stiffness_matrix(self, nodes):
        L = self.length(nodes)
        k_local = local_elastic_stiffness_matrix_3D_beam(self.E, self.nu, self.A, L, self.I_y, self.I_z, self.J)
        s, e = nodes[self.node_start], nodes[self.node_end]
        gamma = rotation_matrix_3D(s.x, s.y, s.z, e.x, e.y, e.z)
        T = transformation_matrix_3D(gamma)
        return T.T @ k_local @ T

class FrameStructure:
    def __init__(self):
        self.nodes, self.elements, self.loads, self.boundary_conditions = [], [], [], []

    def add_node(self, node):
        self.nodes.append(node)

    def add_element(self, element):
        self.elements.append(element)

    def add_boundary_condition(self, bc):
        self.boundary_conditions.append(bc)

    def add_load(self, load):
        self.loads.append(load)

    def apply_boundary_conditions(self, K, F):
        fixed_indices = []
        for bc in self.boundary_conditions:
            dof_start = bc.node_id * 6
            for i, fixed in enumerate([bc.ux, bc.uy, bc.uz, bc.rx, bc.ry, bc.rz]):
                if fixed:
                    fixed_indices.append(dof_start + i)
        K = np.delete(K, fixed_indices, axis=0)
        K = np.delete(K, fixed_indices, axis=1)
        F = np.delete(F, fixed_indices)
        return K, F, fixed_indices

    def solve_displacements(self):
        total_dofs = len(self.nodes) * 6
        K = np.zeros((total_dofs, total_dofs))
        F = np.zeros(total_dofs)

        for e in self.elements:
            k_global = e.global_stiffness_matrix(self.nodes)
            idx = list(range(e.node_start * 6, e.node_start * 6 + 6)) + list(range(e.node_end * 6, e.node_end * 6 + 6))
            for i in range(12):
                for j in range(12):
                    K[idx[i], idx[j]] += k_global[i, j]

        for l in self.loads:
            F[l.node_id * 6:l.node_id * 6 + 6] = [l.fx, l.fy, l.fz, l.mx, l.my, l.mz]

        K_bc, F_bc, fixed_indices = self.apply_boundary_conditions(K, F)
        U_reduced = np.linalg.solve(K_bc, F_bc)
        U = np.zeros(total_dofs)
        free_dofs = np.setdiff1d(np.arange(total_dofs), fixed_indices)
        U[free_dofs] = U_reduced
        print("Displacements:", U)

        reaction_forces = K @ U - F
        print("Reaction Forces and Moments:", reaction_forces)

        return U, reaction_forces

if __name__ == "__main__":
    E, nu = 1000, 0.3
    b, h = 0.5, 1.0
    A = b * h
    I_y = h * b ** 3 / 12
    I_z = b * h ** 3 / 12
    J = 0.02861

    frame = FrameStructure()
    frame.add_node(Node(0, 0, 0, 10))
    frame.add_node(Node(1, 15, 0, 10))
    frame.add_node(Node(2, 15, 0, 0))

    frame.add_element(Element(0, 0, 1, E, nu, A, I_y, I_z, J))
    frame.add_element(Element(1, 1, 2, E, nu, A, I_y, I_z, J))

    frame.add_boundary_condition(BoundaryCondition(0, True, True, True, True, True, True))
    frame.add_boundary_condition(BoundaryCondition(2, True, True, True, False, False, False))

    frame.add_load(Load(1, fx=0.1, fy=0.05, fz=-0.07, mx=0.05, my=-0.1, mz=0.25))
    frame.solve_displacements()
