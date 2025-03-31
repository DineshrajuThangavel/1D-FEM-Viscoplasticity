# 🧮 Nonlinear Finite Element Simulation (NLFEM)

This project implements a custom finite element simulation in Python to model the nonlinear, viscoplastic behavior of a 1D bar with two material segments. Developed as part of a computational mechanics course.

## 📌 Features
- 1D nonlinear FEM for time-dependent material behavior
- Segment 1: purely elastic, Segment 2: viscoplastic
- Weak form derivation and consistent tangent matrix
- Newton-Raphson method with backward Euler time integration
- Verification via convergence study and analytical comparison

## 🧾 Files
- `69306.py`: Main Python code implementing the FEM solver
- `NLFEM_Report_Dinesh.pdf`: Full documentation with derivations, results, and plots
- `assignment_2024_25.pdf`: Original assignment description

## 📊 Outputs
- Displacement vs time
- Stress-strain curves
- Evolution of plastic strain

## 🎓 Academic Context
Developed as part of a graduate course on nonlinear finite element methods and constitutive modeling.

## 🔧 Technologies
- Python (NumPy, Matplotlib)
- FEM solver implemented from scratch
