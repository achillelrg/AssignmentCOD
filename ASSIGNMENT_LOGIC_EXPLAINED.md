# Assignment Logic & Methodology Explained

This document provides a comprehensive, fact-based explanation of the logic, algorithms, and design decisions used throughout the project. The goal is to explain *why* specific choices were made and *how* they solve the core engineering problem.

## 1. The Engineering Problem
The objective was to design a 2D airfoil that maximizes Lift-to-Drag ratio ($L/D$) under specific constraints, using **Computational Fluid Dynamics (XFOIL)**. 
-   **Variables:** The shape is parameterized by 6 CST (Class-Shape Transformation) coefficients (3 upper, 3 lower).
-   **Constraint:** The simulation budget is limited. A full optimization requires tens of thousands of evaluations, which is too slow if done sequentially.

---

## 2. Part A: Automation Foundation
**Logic:** To optimize an airfoil, we need to evaluate thousands of candidates. Doing this manually in the XFOIL terminal interface is impossible.
**Implementation:**
-   **Subprocess Wrapper:** We wrote a Python script (`xfoil_runner.py`) that launches the XFOIL binary as a subprocess.
-   **Stdin/Stdout Piping:** The script sends commands (e.g., `PANEL`, `OPER`, `ALFA 3.0`) directly to XFOIL's standard input and reads the results from standard output in real-time.
-   **Robustness:** 
    -   *Timeout Logic:* XFOIL often hangs (infinite loop) when an airfoil stalls. We implemented a 10-second timeout to kill and restart the process if this happens.
    -   *Result Parsing:* We parse the specific ASCII table output of XFOIL to extract $C_l$, $C_d$, and $C_m$.

---

## 3. Part B: Direct Optimization (Evolutionary Algorithms)
**Logic:** The relationship between Airfoil Shape and Drag is **non-convex** (has many local optima). Gradient-based methods (like Gradient Descent) would get stuck in the first "decent" shape they find. We needed global search algorithms.
**Implementation:**
-   **Particle Swarm Optimization (PSO):**
    -   Simulates a flock of birds. Each candidate design ("particle") remembers its personal best and knows the swarm's global best.
    -   *Why?* PSO is excellent at exploring the entire design space quickly.
-   **Genetic Algorithm (GA):**
    -   Simulates natural selection (Crossover, Mutation, Selection).
    -   *Why?* GA is robust against noisy data.
**Outcome:**
-   **Observation:** While effective, these methods are computationally heavy. One optimization run required **20,000 XFOIL evaluations** (40 population $\times$ 500 generations). At ~2 seconds per run, this takes over 11 hours per seed. This motivated the shift to Surrogate Modeling (Part C).

---

## 4. Part C: Surrogate Modeling (Machine Learning)
**Logic:** Instead of running the expensive physics simulation (XFOIL) 20,000 times, we can run it 1,000 times to create a dataset, and then train an AI (Gaussian Process) to *predict* the physics.
**Implementation:**
-   **Latin Hypercube Sampling (LHS):**
    -   We used `scipy.stats.qmc.LatinHypercube` to generate 2000 initial candidates.
    -   *Why?* Pure random sampling "clumps" points together. LHS ensures the samples are perfectly spread out across the 6-dimensional design space, maximizing information gain.
-   **Gaussian Process (Kriging):**
    -   We trained a GP with a **Matern Kernel**.
    -   *Why?* Unlike a Neural Network, a GP provides an exact fit to the data points and, crucially, provides an **Uncertainty Estimate** (it knows when it is guessing).
-   **Log-Transform:**
    -   Drag ($C_d$) values vary by orders of magnitude ($0.005$ to $0.1$). We trained the model to predict $\log_{10}(C_d)$ instead of raw $C_d$. This linearized the problem and significantly improved accuracy ($R^2$ score).

### 5. Part C2: Robustness via The "Penalty Method"
**The Problem (Survivorship Bias):**
-   Initially, we discarded any run where XFOIL crashed (Stall).
-   The AI model only saw "Successes". It assumed that the entire design space was valid.
-   *Consequence:* The optimizer found an aggressive high-lift shape ($C_l=1.4$) that looked great to the AI but crashed the real XFOIL solver.

**The Solution:**
-   We modified the data generator to **capture failures**.
-   If XFOIL crashes, we assign a "Penalty Value": $C_d = 0.5$ (Wall) and $C_l = 0.0$.
-   **Visual Proof:** The Parity Plots now show a vertical "Wall" of points at 0.5. The model learned that "If I go here, Drag is terrible."
-   **Result:** The optimizer successfully navigated around these walls to find a **valid**, high-performance design ($C_l \approx 1.19$) that was verified in reality.

---

## 6. Part C1: Uncertainty Quantification (UQ)
**Logic:** In real aviation, the Angle of Attack ($\alpha$) is never perfectly constant due to turbulence and pilot error.
**Implementation:**
-   Instead of optimizing for a single $\alpha = 3^\circ$, we statistically analyzed the performance under a **Normal Distribution**: 
    $$ \alpha \sim \mathcal{N}(\mu=3.0, \sigma=0.1) $$
-   We ran a Monte Carlo simulation (100 samples) on the optimal design to calculate the Mean and Standard Deviation of the Lift and Drag. This proves the design is robust to minor flight perturbations.
