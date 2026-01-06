# Parameter Study & Recommendations (Part B)

## Methodology
We swept the following parameters to find the optimal configuration:
- **n_points (Resolution)**: 150, 200, 250
- **n_iter (XFOIL Limit)**: 100, 200
- **pop (Swarm Size)**: 20, 40

## Findings

### 1. Geometry Resolution
- **150 Points**: **UNSTABLE**. Produced non-physical results in our test (e.g., Cl > 10). Likely too coarse for XFOIL's boundary layer solver.
- **200 Points**: **STABLE**. Manual verification showed reasonable physical values (Cl approx 0.5, Cd approx 0.006).

### 2. XFOIL Iterations
- **100 Iterations**: Sufficient for convergence on valid airfoils. Fast execution (~0.2s per call).
- **200 Iterations**: Caused hangs/stalls in the batch study. Not recommended for automated pipelines without strict timeouts.

### 3. Population Size
- **20**: Faster, sufficient for quick exploration.
- **40**: Better chance of escaping local optima, but doubles the wall-clock time per iteration.

## Analysis Guide: How to Read the Log

```text
[Iter 16] Evals: 320 | Best: -4.687316e+02 | Mean: -2.559372e+01
```

1.  **Iter (Generation)**: The PSO update step.
    *   *Total Evaluations* = `Iter` × `Pop`.
    *   Example: Iter 20 × Pop 20 = 400 Evals.
2.  **Best ($J$)**: The fitness of the single best design found so far.
    *   Goal: Minimize this value (more negative is better).
    *   **Good**: -100 to -200 means a valid airfoil with decent lift/drag.
    *   **Great**: -300 to -500+ means highly optimized for the target.
3.  **Mean**: Average fitness of the swarm.
    *   If **Mean** converges to **Best**, the swarm has "collapsed" (everyone agrees on the solution).
    *   If **Mean** is erratic but **Best** is stable, the swarm is still exploring.

## "Overshooting" & Stopping Early
**How to know if you are wasting time?**
Look at the **Best** value over the last 10-20 iterations.
- If it stays exactly the same (e.g., -468.7316), the optimizer is stuck (converged) in a local optimum.
- If it changes only slightly (e.g., -468.73 -> -468.74), you are in "diminishing returns". Stopping now is usually fine unless you need 0.01% perfection.

**New Feature: Graceful Exit**
You can now press `Ctrl+C` at any time during the run.
- The simulation will **STOP** immediately.
- It will **SAVE** all progress made so far.
- It will **GENERATE PLOTS** for the best design found up to that moment.
- You do not need to wait for the counter to reach zero!

## FAQ: Why does the ETA increase?
You might notice the Estimated Time rising (e.g., from 12m to 20m) during the first few iterations.
**This is Normal** and actually a **Good Sign**.
1.  **Bad Shapes Fail Fast**: At the start, the swarm has many garbage shapes. XFOIL rejects them instantly (< 0.1s).
2.  **Good Shapes Take Time**: As the optimizer improves, it finds valid airfoils. XFOIL must run the full flow solver (approx 0.7s - 1.5s).
3.  **Result**: The "Average Time per Iteration" increases because the problem is getting *harder* (better quality), so the ETA corrects upwards.

## Parameter Influence Guide

| Parameter | Impact on Runtime | Impact on Quality | Recommendation |
| :--- | :--- | :--- | :--- |
| **`--evals`** | Linear (More = Slower) | **High**. More fuel means finding better global optima. | **Increase** (e.g., 1000+) for final designs. |
| **`--pop`** | Linear (More = Slower) | **High**. Larger swarms escape local traps better. | **40-50** for final runs. 20 for testing. |
| **`--points`** | High (Quadratic?) | **Critical**. Too low = Garbage. Too high = Slow/Hang. | **200** is the sweet spot. Never use <160. |
| **`--iter`** | Low (if valid) | **Low**. 100 is enough for XFOIL to decide "Valid/Invalid". | **100**. Higher just wastes time on bad shapes. |

## "Perfect" Recommendations
For a balance of reliability and speed, use:

```python
settings = {
    "n_points": 200,      # Standard XFOIL resolution
    "n_iter": 100,        # Fast, sufficient convergence
    "pop": 20,            # Efficient (use 40 for final high-quality runs)
    "evals": 400          # Budget (20 * 20 generations)
}
```

## How to use these parameters
You can pass these directly to `run_airfoil.py` or modify the defaults in the script:

```bash
# Recommended "Perfect" Run (Fast Test)
python3 -m experiments.run_airfoil --part B --pop 20 --evals 400

# Recommended "Final Quality" Run
python3 -u -m experiments.run_airfoil --part B --clean --points 200 --iter 100 --pop 40 --evals 1000
```
