import sys
import os
import pytest
import numpy as np 

# Forcefully add `src/` to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Correct import using `src`
from src.Critical_Load_Analysis.criticalloadanalysis_tutorial import (
    structure_solver,
    critical_load_analysis,
    get_problem_setup
)

@pytest.fixture
def problem_setup():
    """Fixture to set up nodes, connection, loads, and supports for tests."""
    return get_problem_setup()

def test_structure_solver(problem_setup):
    """Test structure_solver output dimensions and constraints handling."""
    nodes, connection, loads, supports = problem_setup
    displacements, reactions = structure_solver(nodes, connection, loads, supports)

    # Check output array sizes
    assert displacements.shape == (len(nodes) * 6,), "Incorrect displacements shape"
    assert reactions.shape == (len(nodes) * 6,), "Incorrect reactions shape"

    # Ensure fixed nodes have zero displacements
    for support in supports:
        node_id = support[0]
        constrained_dofs = [i for i, val in enumerate(support[1:]) if val]
        for dof in constrained_dofs:
            assert displacements[6 * node_id + dof] == pytest.approx(0), f"Node {node_id} DOF {dof} should be fixed"

def test_critical_load_analysis(problem_setup):
    """Test critical load analysis for expected eigenvalues."""
    nodes, connection, loads, supports = problem_setup
    eigvals, eigvecs = critical_load_analysis(nodes, connection, loads, supports)

    assert eigvals.shape[0] > 0, "Eigenvalues array should not be empty"
    assert eigvecs.shape[1] > 0, "Eigenvectors array should not be empty"

    # Ensure eigenvalues are sorted in ascending order
    assert np.all(np.diff(eigvals) >= 0), "Eigenvalues should be sorted"

def test_invalid_element():
    """Ensure element initialization fails with zero-length elements."""
    from src.Critical_Load_Analysis.criticalloadanalysis_tutorial import Node, Element
    with pytest.raises(ValueError, match="zero length element"):
        node1 = Node(0, 0, 0, 0)
        node2 = Node(1, 0, 0, 0)  # Same coordinates as node1
        Element(node1, node2, 200, 0.3, 0.01, 0.001, 0.001, 0.001, 0.001, [1, 0, 0])

