import numpy as np
import pytest
from src.Critical_Load_Analysis.criticalloadanalysis_tutorial import get_problem_setup
from src.Critical_Load_Analysis.graph_3d import get_free_dofs, extract_analysis_results, hermite_interpolation

@pytest.mark.skip
def test_get_free_dofs():
    """Test that get_free_dofs correctly identifies free degrees of freedom."""
    supports = np.array([[0, 1, 0, 1, 0, 1, 0],
                         [1, 1, 1, 1, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0]])  # Third node is free

    free_dofs = get_free_dofs(supports)
    assert len(free_dofs) == 6, "Expected 6 free DOFs for the fully free node"

@pytest.mark.skip
def test_extract_analysis_results():
    """Test that extract_analysis_results returns expected dimensions for analysis."""
    nodes, connection, disp_linear, disp_buckling, lambda_crit = extract_analysis_results()

    assert len(nodes.shape) == 2, "Nodes array should be 2D"
    assert len(connection.shape) == 2, "Connection array should be 2D"
    assert disp_linear.shape[0] == len(nodes), "Displacement results should match node count"
    assert disp_buckling.shape[0] == len(nodes), "Buckling displacement should match node count"
    assert lambda_crit is None or lambda_crit > 0, "Lambda should be positive or None"


def test_hermite_interpolation():
    """Test Hermite interpolation with given start and end conditions."""
    start_disp = np.array([0, 0, 0])
    end_disp = np.array([1, 1, 1])
    start_rot = np.array([0.1, 0.1, 0.1])
    end_rot = np.array([-0.1, -0.1, -0.1])

    result = hermite_interpolation(start_disp, end_disp, start_rot, end_rot, num_points=10)
    
    assert result.shape == (10, 3), "Interpolation should return a (10, 3) array"
    assert np.all(result[0] == start_disp), "First point should match start displacement"
    assert np.all(result[-1] == end_disp), "Last point should match end displacement"


if __name__ == "__main__":
    pytest.main()
