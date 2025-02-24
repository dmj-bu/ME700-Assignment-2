# Direct Stiffness Method Solver Tutorial

[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![os](https://img.shields.io/badge/os-ubuntu%20|%20macos%20|%20windows-blue.svg)](https://github.com/dmj-bu/ME700-Assignment-1)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/dmj-bu/ME700-Assignment-1/blob/main/LICENSE)

[![codecov](https://codecov.io/gh/dmj-bu/ME700-Assignment-1/Elasto_Plastic_Model/graph/badge.svg?token=YOUR_CODECOV_TOKEN)]((https://codecov.io/gh/dmj-bu/ME700-Assignment-1/tree/main/src%2FElasto_Plastic_Model))
[![Run Tests](https://github.com/dmj-bu/ME700-Assignment-2/actions/workflows/tests.yml/badge.svg)](https://github.com/dmj-bu/ME700-Assignment-2/actions/workflows/tests.yml)

---
##  **Introduction**
This tutorial provides a step-by-step walkthrough of analyzing a specific 3D frame structure using a Python-based **Direct Stiffness Method solver**. 

It covers:
- Defining geometry and properties of the frame
- Applying loads and boundary conditions
- Running the solver and interpreting outputs
- Understanding the mathematical background and equations

This example focuses on a cantilever frame subjected to a vertical load.

---
## Installation & Usage

### 1: Clone the Repository

```bash
git clone https://github.com/dmj-bu/ME700-Assignment-2.git
cd ME700-Assignment-2
```
## **create and activate Conda environment**
```bash
conda create --name me700-tutorial python=3.12
conda activate me700-tutorial
```

### **2: Install Dependencies**
```bash
pip install -e .
```

### 4: Run the Tutorial
```bash
cd src/Direct_Stiffness_Method
python directstiffnessmethod_solver.py
```

---

##  **Step 1: Defining the Frame Structure**

### **Mathematical Context**
Each node in a 3D frame has 6 degrees of freedom (DOFs):
```math
\\{DOFs: } \\\\
(u_x, u_y, u_z, \theta_x, \theta_y, \theta_z)

```

###  **Code and Instructions:**
The following code defines the frame:

```python
frame = FrameStructure()

# Define nodes for the frame
frame.add_node(Node(1, 0.0, 0.0, 0.0))  # Fixed support
frame.add_node(Node(2, 0.0, 3.0, 0.0))  # Joint node
frame.add_node(Node(3, 3.0, 3.0, 0.0))  # Free end node

# Define frame elements
frame.add_element(Element(1, 1, 2, E=210e9, A=0.01, I_y=8.1e-6, I_z=8.1e-6, J=1.6e-5))
frame.add_element(Element(2, 2, 3, E=210e9, A=0.02, I_y=1.1e-5, I_z=1.1e-5, J=2.0e-5))
```

###  **Explanation:**
- E: Young's modulus (material stiffness)
- A: Cross-sectional area
- I_y, I_z: Second moments of area (bending)
- J: Torsional constant

Change these parameters to analyze different material or geometric configurations.

---

##  **Step 2: Applying Loads and Boundary Conditions**

###  **Mathematical Context**
The load vector \(F\) contains forces and moments:
```math
\\{F\\} = [F_x, F_y, F_z, M_x, M_y, M_z]^T
```

###  **Code and Instructions:**
```python
# Fully fixed boundary at Node 1
frame.add_boundary_condition(BoundaryCondition(1, True, True, True, True, True, True))

# Apply a vertical load at Node 3
frame.add_load(Load(3, fy=-1000.0))
```

###  **Instruction:**
- Change fy=-1000.0 to test different load magnitudes or directions.

---

##  **Step 3: Understanding the Stiffness Matrix**

###  **Key Equations:**

- **Axial Stiffness:**
```math
k_{axial} = \
\frac{EA}{L}
```
- **Bending Stiffness:**
```math
k_{bend} = \
\frac{12EI}{L^3}
```
- **Torsional Stiffness:**
```math
k_{torsion} = \
\frac{GJ}{L}
```

The **global stiffness matrix**  $[K]_{global}$ is assembled as:
```math
[K]_{global} = \
\sum T^T [K]_{local} T
```
Where \(T\) is the transformation matrix.

---

##  **Step 4: Solving for Displacements**

###  **Mathematical Context**
Solving the system:
```math
[K]_{global} {U\\} = {F\\}
```
Where (U) contains translations and rotations.

###  **Code and Instructions:**
```python
# Solve for displacements
frame.solve_displacements()
```

###  **Example Output:**
```
Displacements at Nodes:
Node 1: [0. 0. 0. 0. 0. 0.]
Node 2: [0.0012 -0.0025 0.0 0.0003 -0.0004 0.0]
Node 3: [0.0025 -0.0048 0.0 0.0007 -0.0009 0.0]
```

###  **Interpretation:**
- Translational results ($`u_x`$,$`u_y`$,$`u_z`$) show how far nodes move.
- Rotational results ($`\theta_x`$,$`\theta_y`$,$`\theta_z`$) show angular displacements.
- Modify material properties or geometry to see their impact.

---

##  **Step 5: Reaction Forces and Internal Forces**

###  **Mathematical Context:**
- **Reaction forces** at supports:
```math
\\{R\\} = [K] {U\\} - {F\\}
```
- **Internal forces** in each element:
```math
\\{F\\}_{internal} = [K]_{local} {U\\}_{local}
```

###  **Example Output:**
```
Reaction Forces at Supports:
Node 1 Reactions: [500.0, 1000.0, 0.0, 20.0, 30.0, 0.0]

Internal Forces in Elements:
Element 1 Internal Forces: [...]
Element 2 Internal Forces: [...]
```

###  **Interpretation:**
- Check if reaction forces match expected supports.
- Ensure internal forces do not exceed material limits.

---

##  **Step 6: Interpreting the Results**

###  **Key Points:**
1. **Displacements** indicate structural flexibility.
   - Excessive displacement may suggest inadequate stiffness.

2. **Reaction forces** validate equilibrium.
   - Compare with applied loads for consistency.

3. **Internal forces** guide material selection and cross-section sizing.
   - High internal moments suggest larger or stiffer sections.

###  **Next Steps for Exploration:**
- Vary E, A, I_y, I_z, and J for sensitivity analysis.
- Add more nodes and elements for complex geometries.
- Introduce lateral loads and moment loads at various nodes.

---
