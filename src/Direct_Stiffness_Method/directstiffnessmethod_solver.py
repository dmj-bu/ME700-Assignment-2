import numpy as np

class Node:
    def __init__(self, node_id, x, y, z):
        self.id = node_id
        self.x = x
        self.y = y
        self.z = z

class Element:
    def __init__(self, element_id, node_start, node_end, E, A, I_y, I_z, J):
        self.id = element_id
        self.node_start = node_start
        self.node_end = node_end
        self.E = E
        self.A = A
        self.I_y = I_y
        self.I_z = I_z
        self.J = J

    def length(self, nodes):
        start_node = nodes[self.node_start - 1]
        end_node = nodes[self.node_end - 1]
        return np.sqrt((end_node.x - start_node.x) ** 2 +
                       (end_node.y - start_node.y) ** 2 +
                       (end_node.z - start_node.z) ** 2)

    def global_stiffness_matrix(self, nodes):
        L = self.length(nodes)
        k_local = self.local_stiffness_matrix(nodes)
        k_global = k_local  # Assuming no transformation needed for aligned elements
        return k_global

    def local_stiffness_matrix(self, nodes):
        L = self.length(nodes)
        E = self.E
        A = self.A
        I_y = self.I_y
        I_z = self.I_z
        J = self.J

        k_axial = (E * A) / L
        k_bend_y = (12 * E * I_z) / (L ** 3)
        k_bend_z = (12 * E * I_y) / (L ** 3)
        k_torsion = (E * J) / L

        k_local = np.zeros((12, 12))
        k_local[0, 0] = k_axial
        k_local[6, 6] = k_axial
        k_local[0, 6] = -k_axial
        k_local[6, 0] = -k_axial

        k_local[1, 1] = k_bend_y
        k_local[7, 7] = k_bend_y
        k_local[1, 7] = -k_bend_y
        k_local[7, 1] = -k_bend_y

        k_local[2, 2] = k_bend_z
        k_local[8, 8] = k_bend_z
        k_local[2, 8] = -k_bend_z
        k_local[8, 2] = -k_bend_z

        k_local[3, 3] = k_torsion
        k_local[9, 9] = k_torsion
        k_local[3, 9] = -k_torsion
        k_local[9, 3] = -k_torsion

        return k_local

    def internal_forces(self, nodes, displacements):
        k_local = self.local_stiffness_matrix(nodes)
        d_local = displacements  # Assuming aligned coordinates
        internal_forces = k_local @ d_local
        return internal_forces

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

class FrameStructure:
    def __init__(self):
        self.nodes = []
        self.elements = []
        self.loads = []
        self.boundary_conditions = []

    def add_node(self, node):
        self.nodes.append(node)

    def add_element(self, element):
        self.elements.append(element)

    def add_boundary_condition(self, bc):
        self.boundary_conditions.append(bc)

    def add_load(self, load):
        self.loads.append(load)

    def assemble_global_stiffness_matrix(self):
        total_dofs = len(self.nodes) * 6
        K_global = np.zeros((total_dofs, total_dofs))

        for element in self.elements:
            k_global = element.global_stiffness_matrix(self.nodes)
            start_index = (element.node_start - 1) * 6
            end_index = (element.node_end - 1) * 6

            indices = list(range(start_index, start_index + 6)) + list(range(end_index, end_index + 6))
            for i in range(12):
                for j in range(12):
                    K_global[indices[i], indices[j]] += k_global[i, j]

        return K_global

    def apply_boundary_conditions(self, K_global, F_global):
        for bc in self.boundary_conditions:
            index = (bc.node_id - 1) * 6
            fixed_dofs = [bc.ux, bc.uy, bc.uz, bc.rx, bc.ry, bc.rz]
            for i, fixed in enumerate(fixed_dofs):
                if fixed:
                    dof_index = index + i
                    K_global[dof_index, :] = 0
                    K_global[:, dof_index] = 0
                    K_global[dof_index, dof_index] = 1
                    F_global[dof_index] = 0

        return K_global, F_global

    def solve_displacements(self):
        total_dofs = len(self.nodes) * 6
        K_global = self.assemble_global_stiffness_matrix()
        F_global = np.zeros(total_dofs)

        for load in self.loads:
            base_index = (load.node_id - 1) * 6
            F_global[base_index:base_index + 6] = [load.fx, load.fy, load.fz, load.mx, load.my, load.mz]

        K_global_bc, F_global_bc = self.apply_boundary_conditions(K_global.copy(), F_global.copy())
        U = np.linalg.solve(K_global_bc, F_global_bc)
        print("\nDisplacements at Nodes:")
        for i, node in enumerate(self.nodes):
            disp = U[i * 6:(i + 1) * 6]
            print(f"Node {node.id}: {disp}")

        self.compute_reaction_forces(K_global, U, F_global)
        self.compute_internal_forces(U)
        return U

    def compute_reaction_forces(self, K_global, displacements, F_global):
        reactions = K_global @ displacements - F_global
        print("\nReaction Forces at Supports:")
        for bc in self.boundary_conditions:
            base_index = (bc.node_id - 1) * 6
            dof_reactions = reactions[base_index:base_index + 6]
            print(f"Node {bc.node_id} Reactions: {dof_reactions}")

    def compute_internal_forces(self, displacements):
        print("\nInternal Forces in Elements:")
        for element in self.elements:
            start_index = (element.node_start - 1) * 6
            end_index = (element.node_end - 1) * 6
            element_disp = np.concatenate([
                displacements[start_index:start_index + 6],
                displacements[end_index:end_index + 6]
            ])
            internal_forces = element.internal_forces(self.nodes, element_disp)
            print(f"Element {element.id} Internal Forces: {internal_forces}")

if __name__ == "__main__":
    frame = FrameStructure()

    # Define nodes
    frame.add_node(Node(1, 0.0, 0.0, 0.0))
    frame.add_node(Node(2, 0.0, 3.0, 0.0))
    frame.add_node(Node(3, 3.0, 3.0, 0.0))

    # Define elements
    frame.add_element(Element(1, 1, 2, E=210e9, A=0.01, I_y=8.1e-6, I_z=8.1e-6, J=1.6e-5))
    frame.add_element(Element(2, 2, 3, E=210e9, A=0.02, I_y=1.1e-5, I_z=1.1e-5, J=2.0e-5))

    # Apply boundary conditions (fixed at Node 1)
    frame.add_boundary_condition(BoundaryCondition(1, True, True, True, True, True, True))

    # Apply loads (force at Node 3)
    frame.add_load(Load(3, fy=-1000.0))

    # Solve for displacements and compute reactions and internal forces
    frame.solve_displacements()
