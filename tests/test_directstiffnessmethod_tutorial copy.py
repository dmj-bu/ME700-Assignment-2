import numpy as np
import pytest
from src.Direct_Stiffness_Method.directstiffnessmethod_tutorial import (
    Node, Element, structure_solver, get_problem_setup, main
)

def test_node_creation():
    """Test Node class initialization and support setting."""
    node = Node(0, 0, 0, 0)
    assert node.id == 0
    assert node.x == 0
    assert node.y == 0
    assert node.z == 0
    assert np.all(node.F == np.zeros(6))

    node.set_support([1, 0, 1, 0, 1, 0])
    assert node.supported_dofs == [0, 2, 4]

def test_element_creation():
    """Test Element class initialization and stiffness computation."""
    node1 = Node(0, 0, 0, 0)
    node2 = Node(1, 1, 0, 0)
    E, nu, A, Iy, Iz, Ip, J = 200e9, 0.3, 0.01, 1e-6, 1e-6, 1e-6, 1e-6
    z_axis = [0, 0, 1]

    element = Element(node1, node2, E, nu, A, Iy, Iz, Ip, J, z_axis)
    assert element.L == pytest.approx(1.0, rel=1e-6)
    assert element.k_e.shape == (12, 12)

    with pytest.raises(ValueError, match="Nodes coincide – zero length element."):
        _ = Element(node1, node1, E, nu, A, Iy, Iz, Ip, J, z_axis)

def test_structure_solver():
    """Test structure solver with a simple problem setup."""
    nodes, connection, loads, supports = get_problem_setup()
    
    # Solve for displacements
    displacements, reactions = structure_solver(nodes, connection, loads, supports)

    assert len(displacements) == 6 * len(nodes)
    assert len(reactions) == 6 * len(nodes)

    # Check some expected values (example: first fixed node should have zero displacement)
    assert np.allclose(displacements[:6], np.zeros(6), atol=1e-6)

def test_main_output():
    """Test main function execution to ensure it does not throw errors."""
    try:
        main()
    except Exception as e:
        pytest.fail(f"main() function raised an error: {e}")

if __name__ == "__main__":
    pytest.main()