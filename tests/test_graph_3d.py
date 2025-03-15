import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.Critical_Load_Analysis.graph_3d import (
    get_free_dofs,
    extract_analysis_results,
    plot_3d_displacements
)

def test_get_free_dofs():
    """
    Provide a 7-column supports array: 
      [node_id, fix_dof0, fix_dof1, fix_dof2, fix_dof3, fix_dof4, fix_dof5].
    1 = fixed, 0 = free.
    """
    # Here we have 3 "nodes" (by row), each with ID in col 0 and fix/free flags in cols 1..6.
    # - Row0 -> node 0, partially fixed => free DOFs should be [0,2,4].
    # - Row1 -> node 1, fully fixed => no free DOFs.
    # - Row2 -> node 2, fully free  => free DOFs should be [12,13,14,15,16,17].
    supports = np.array([
        [0, 0, 1, 0, 1, 0, 1],  # Node0 partial fix
        [1, 1, 1, 1, 1, 1, 1],  # Node1 fully fixed
        [2, 0, 0, 0, 0, 0, 0]   # Node2 fully free
    ])

    free_dofs = get_free_dofs(supports)
    expected_free_dofs = np.array([0, 2, 4, 12, 13, 14, 15, 16, 17])

    assert np.array_equal(np.sort(free_dofs), np.sort(expected_free_dofs)), (
        f"Expected free DOFs {expected_free_dofs}, but got {free_dofs}"
    )


@pytest.mark.parametrize("mode_type", ["linear", "buckling"])
def test_plot_3d_displacements(mode_type):
    """
    Since plot_3d_displacements() in graph_3d.py 
    does not accept data parameters (it calls extract_analysis_results() internally),
    we just call it and then check the resulting figure.
    """
    plot_3d_displacements(mode_type=mode_type)
    
    # Grab the current figure from pyplot and check its type:
    fig = plt.gcf()
    assert isinstance(fig, plt.Figure), f"Expected a matplotlib Figure for {mode_type} mode"
    plt.close(fig)  # Close it to avoid extra windows popping up


def test_extract_analysis_results():
    nodes, connection, disp_linear, disp_buckling, lambda_crit = extract_analysis_results()

    # The code returns disp_linear as shape (48,) and disp_buckling as shape (8,6).
    # We'll reshape disp_linear ourselves in the test to check (8,6).
    disp_linear_reshaped = disp_linear.reshape(len(nodes), 6)
    disp_buckling_reshaped = disp_buckling  # already (8,6)

    assert disp_linear_reshaped.shape == (nodes.shape[0], 6), (
        f"Expected linear displacement shape {(nodes.shape[0], 6)}, "
        f"got {disp_linear.shape} originally, which reshapes to {disp_linear_reshaped.shape}"
    )
    assert disp_buckling_reshaped.shape == (nodes.shape[0], 6), (
        f"Expected buckling displacement shape {(nodes.shape[0], 6)}, "
        f"but got {disp_buckling_reshaped.shape}"
    )

    # And keep the rest of your checks
    assert connection.shape[1] >= 2, "connection should have at least start/end columns"
    assert (lambda_crit is None) or (lambda_crit > 0), "Lambda critical should be None or positive"

@pytest.mark.skip(reason="graph_3d.py does not define a main() function.")
def test_script_execution(monkeypatch):
    """
    If you ever define a main() in graph_3d.py (or want to test the 
    __main__ block), un-skip this test.
    """
    import src.Critical_Load_Analysis.graph_3d as graph_3d

    # Mock the user input inside main() (if you had one):
    monkeypatch.setattr("builtins.input", lambda _: "linear")

    try:
        graph_3d.main()
        assert True
    except Exception as e:
        pytest.fail(f"Script execution failed: {e}")
