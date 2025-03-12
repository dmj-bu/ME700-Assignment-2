import pytest
import numpy as np

import sys
import os

# Get the absolute path of the `src/` directory
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../src"))

# Ensure `src/` is in sys.path
if src_path not in sys.path:
    sys.path.insert(0, src_path)

print("Updated sys.path:", sys.path)  # Debugging output

# Now import your modules
from src.Direct_Stiffness_Method.directstiffnessmethod_tutorial import structure

# Define global material properties
E = 10000
nu = 0.3
r = 1
A = np.pi * r ** 2
I_y = np.pi * r ** 4 / 4
I_z = np.pi * r ** 4 / 4
I_rho = np.pi * r ** 4 / 2
J = np.pi * r ** 4 / 2
L = 25  # Beam length

@pytest.fixture
def setup_structure():
    """Returns nodes, elements, supports, and loads for the test cases."""
    nodes = np.array([
        [0, 0, 0],
        [L / 6, 2 * L / 6, 3 * L / 6],
        [2 * L / 6, 4 * L / 6, 6 * L / 6],
        [3 * L / 6, 6 * L / 6, 9 * L / 6],
        [4 * L / 6, 8 * L / 6, 12 * L / 6],
        [5 * L / 6, 10 * L / 6, 15 * L / 6],
        [L, 2 * L, 3 * L]
    ])

    connection = np.array([
        [i, i + 1, E, nu, A, I_y, I_z, I_rho, J, [0, 0, 1]]
        for i in range(6)  # Connecting each adjacent node
    ], dtype=object)

    supports = np.array([
        [0, 1, 1, 1, 1, 1, 1],  # Node 0 fixed
        [1, 0, 0, 0, 0, 0, 0],  # Free nodes
        [2, 0, 0, 0, 0, 0, 0],
        [3, 0, 0, 0, 0, 0, 0],
        [4, 0, 0, 0, 0, 0, 0],
        [5, 0, 0, 0, 0, 0, 0],
        [6, 0, 0, 0, 0, 0, 0]
    ])

    load = np.zeros((7, 6))
    load[6] = [0.05, -0.1, 0.23, 0.1, -0.025, -0.08]

    return nodes, connection, load, supports

def test_stiffness_matrix_size(setup_structure):
    """Ensure the global stiffness matrix is the correct size."""
    nodes, connection, load, supports = setup_structure
    total_dofs = len(nodes) * 6
    _, forces = structure(nodes, connection, load, supports)

    assert forces.shape[0] == total_dofs, "Stiffness matrix size mismatch!"

def test_force_equilibrium(setup_structure):
    """Check if reaction forces sum up to applied loads (static equilibrium)."""
    nodes, connection, load, supports = setup_structure
    _, reaction_forces = structure(nodes, connection, load, supports)

    total_reaction = np.sum(reaction_forces.reshape(-1, 6), axis=0)
    applied_force = np.sum(load, axis=0)

    np.testing.assert_almost_equal(total_reaction[:3], -applied_force[:3], decimal=4, err_msg="Force equilibrium failed!")
    np.testing.assert_almost_equal(total_reaction[3:], -applied_force[3:], decimal=4, err_msg="Moment equilibrium failed!")

def test_displacement_reasonableness(setup_structure):
    """Ensure displacements are within expected magnitudes (no numerical explosion)."""
    nodes, connection, load, supports = setup_structure
    displacements, _ = structure(nodes, connection, load, supports)

    max_disp = np.max(np.abs(displacements))
    assert max_disp < 1, "Unrealistic displacement magnitude detected!"

def test_fixed_node_displacement(setup_structure):
    """Ensure the fixed node (Node 0) has zero displacement & rotation."""
    nodes, connection, load, supports = setup_structure
    displacements, _ = structure(nodes, connection, load, supports)

    fixed_dof_displacements = displacements[0:6]  # First 6 DOFs (Node 0)
    np.testing.assert_almost_equal(fixed_dof_displacements, np.zeros(6), decimal=6, err_msg="Fixed node displacement nonzero!")

def test_reaction_moments_vs_analytical(setup_structure):
    """Compare reaction moments at Node 0 with analytical solution."""
    nodes, connection, load, supports = setup_structure
    _, reaction_forces = structure(nodes, connection, load, supports)

    computed_My = reaction_forces[4]  # M_y at Node 0
    computed_Mz = reaction_forces[5]  # M_z at Node 0

    expected_My = load[6, 2] * L  # Fz * L
    expected_Mz = -load[6, 1] * L  # -Fy * L

    np.testing.assert_almost_equal(computed_My, expected_My, decimal=3, err_msg="Mismatch in M_y reaction moment!")
    np.testing.assert_almost_equal(computed_Mz, expected_Mz, decimal=3, err_msg="Mismatch in M_z reaction moment!")
