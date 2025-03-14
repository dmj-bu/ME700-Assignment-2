import numpy as np
import pytest
from src.Critical_Load_Analysis.math_utils import (
    local_elastic_stiffness_matrix_3D_beam,
    check_unit_vector,
    check_parallel,
    rotation_matrix_3D,
    transformation_matrix_3D
)


def test_local_elastic_stiffness_matrix_3D_beam():
    """Test computation of the 3D beam stiffness matrix."""
    E, nu, A, L, Iy, Iz, J = 210e9, 0.3, 0.01, 2.0, 5e-6, 5e-6, 1e-5
    k_e = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
    
    assert k_e.shape == (12, 12), "Stiffness matrix should be 12x12"
    assert np.allclose(k_e, k_e.T, atol=1e-8), "Matrix should be symmetric"


def test_check_unit_vector():
    """Test validation of unit vectors."""
    vec = np.array([1.0, 0.0, 0.0])
    check_unit_vector(vec)  # Should not raise an error

    with pytest.raises(ValueError):
        check_unit_vector(np.array([1, 1, 1]))  # Not a unit vector


def test_check_parallel():
    """Test validation of parallel vectors."""
    vec_1 = np.array([1, 0, 0])
    vec_2 = np.array([0, 1, 0])
    check_parallel(vec_1, vec_2)  # Should not raise an error

    with pytest.raises(ValueError):
        check_parallel(vec_1, np.array([2, 0, 0]))  # Parallel vectors should raise error


def test_rotation_matrix_3D():
    """Test 3D rotation matrix generation."""
    gamma = rotation_matrix_3D(0, 0, 0, 1, 1, 1)
    
    assert gamma.shape == (3, 3), "Rotation matrix should be 3x3"
    assert np.allclose(np.linalg.det(gamma), 1.0, atol=1e-6), "Determinant should be 1"


def test_transformation_matrix_3D():
    """Test 3D transformation matrix creation."""
    gamma = np.eye(3)
    Gamma = transformation_matrix_3D(gamma)
    
    assert Gamma.shape == (12, 12), "Transformation matrix should be 12x12"
    assert np.allclose(Gamma[:3, :3], gamma), "Top-left block should match input rotation"


if __name__ == "__main__":
    pytest.main()
