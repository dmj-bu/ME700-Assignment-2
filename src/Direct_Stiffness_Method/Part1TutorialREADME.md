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

This example focuses on a 3D beam structure subjected to forces and moments, as per the first in-class review.

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
python directstiffnessmethod_tutorial.py
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
# Define nodes
nodes = np.array([
    [0, 0, 10],  # Node 0 at coordinates (x=0, y=0, z=10)
    [15, 0, 10], # Node 1 at coordinates (x=15, y=0, z=10)
    [15, 0, 0]   # Node 2 at coordinates (x=15, y=0, z=0)
])
```
Each row represents a **node** with its **x, y, z coordinates**.

The elements have to be specific to the material properties of the material
```python
# Define Material Properties
E, nu = 1000, 0.3
b, h = 0.5, 1.0
A = b * h
I_y = h * b ** 3 / 12
I_z = b * h ** 3 / 12
I_rho = b * h / 12 * (b**2 + h**2) 
J = 0.02861
```
```python
# Define elements
connection = np.array([
    [0, 1, E, nu, b, I_y, I_z, I_rho, J, [0, 0, 1]], # Element 0 between Node 0 and 1
    [1, 2, E, nu, b, I_y, I_z, I_rho, J, [1, 0, 0]]  # Element 1 between Node 1 and 2
], dtype=object)
```
Each row represents an **element** with the following:
- **First two numbers:** Start and end node indices (e.g., `0,1` means the element connects Node 0 to Node 1)
- **Material properties:** 
  - `E`: Young's modulus (stiffness)
  - `A`: Cross-sectional area
  - `I_y, I_z`: Second moments of area (bending stiffness in y and z directions)
  - `J`: Torsional constant
- **Last entry `[0, 0, 1]` or `[1, 0, 0]`** defines the **local z-axis** direction for the element.

---

##  **Step 2: Applying Loads and Boundary Conditions**

###  **Mathematical Context**
The load vector \(F\) contains forces and moments:
```math
\\{F\\} = [F_x, F_y, F_z, M_x, M_y, M_z]^T
```

###  **Code and Instructions:**
```python
# Define constraints
supports = np.array([
    [0, 1, 1, 1, 1, 1, 1],  # Node 0 is fully fixed (all DOFs constrained)
    [1, 0, 0, 0, 0, 0, 0],  # Node 1 is free (all DOFs unconstrained)
    [2, 1, 1, 1, 0, 0, 0]   # Node 2 is pinned (constraining translations, but not rotations)
])
```
Each row corresponds to a **node index** followed by six values:
- `1` → DOF is constrained
- `0` → DOF is free
### Apply external forces
```python
load = np.array([
    [0, 0, 0, 0, 0, 0],
    [0.1, 0.05, -0.07, 0.05, -0.1, 0.25],
    [0, 0, 0, 0, 0, 0]])
```
Each row represents a **force vector** for a node in the order `[Fx, Fy, Fz, Mx, My, Mz]`.

---

##  **Step 3: Solving the System**

###  **Key Equations (from math_utils):**

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

The system is solved using:
```math
[K]_{global} {U} = {F}
```
Where (U) contains translations and rotations.

###  **Code and Instructions:**
```python
# Solve the system
displacement, forces = structure(nodes, connection, load, supports)
```

---

## **Step 4: Interpreting Results**

### **Example Output:**
```
Computed Displacements: [values...]
Reaction Forces: [values...]
```

### **Interpretation:**
- **Displacements** indicate the deformation of the structure.
- **Reaction forces** validate equilibrium.
- **Modify material properties or loads to explore different cases.**

---
This tutorial provides a fundamental framework for solving 3D frame problems using the **Direct Stiffness Method**. Modify the parameters and structure to analyze different cases!
