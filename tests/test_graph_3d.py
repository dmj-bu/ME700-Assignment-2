import numpy as np
import pytest
from src.Critical_Load_Analysis.graph_3d import get_free_dofs, extract_analysis_results, hermite_interpolation

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
