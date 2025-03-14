import numpy as np
import pytest
from src.Critical_Load_Analysis.graph_3d import get_free_dofs, extract_analysis_results, hermite_interpolation

def test_get_free_dofs():
    """Test that get_free_dofs correctly identifies free degrees of freedom."""
    supports = np.array([[0, 1, 0, 1, 0, 1, 0],  # Node 1
                         [1, 1, 1, 1, 1, 1, 1],  # Node 2 (fully fixed)
                         [0, 0, 0, 0, 0, 0, 0]])  # Node 3 (fully free)

    free_dofs = get_free_dofs(supports)
    
    # Node 3 should have all 6 DOFs free
    assert len(free_dofs) == 6, f"Expected 6 free DOFs, got {len(free_dofs)}"
    assert np.array_equal(free_dofs, [12, 13, 14, 15, 16, 17]), "Incorrect free DOFs"

def test_extract_analysis_results():
    """Test that extract_analysis_results returns expected values."""
    nodes, connection, disp_linear, disp_buckling, lambda_crit = extract_analysis_results()
    
    assert nodes.shape[1] == 3, "Nodes should have 3 columns (x, y, z)"
    assert len(connection.shape) == 2, "Connection should be a 2D array"
    
    # Ensure displacement arrays match the number of nodes
    assert disp_linear.shape[0] == nodes.shape[0], "Linear displacement mismatch"
    assert disp_buckling.shape[0] == nodes.shape[0], "Buckling displacement mismatch"
    
    # Lambda should be positive or None
    assert lambda_crit is None or lambda_crit > 0, "Lambda should be positive or None"

def test_hermite_interpolation():
    """Test Hermite interpolation function."""
    start_disp = np.array([0, 0, 0])
    end_disp = np.array([1, 1, 1])
    start_rot = np.array([0.1, 0.1, 0.1])
    end_rot = np.array([-0.1, -0.1, -0.1])

    interpolated = hermite_interpolation(start_disp, end_disp, start_rot, end_rot, num_points=10)

    assert interpolated.shape == (10, 3), "Incorrect shape for interpolated results"
    assert np.allclose(interpolated[0], start_disp), "Start displacement mismatch"
    assert np.allclose(interpolated[-1], end_disp), "End displacement mismatch"

if __name__ == "__main__":
    pytest.main()
