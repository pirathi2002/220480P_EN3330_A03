# EN3330 Assignment 03 - Lagrangian Duality

**Name:** Pirathishanth A. **Index No:** 220480P **Module:** EN3330 Introduction to Engineering Optimization

Python code that checks the calculations and draws the graphs used in my Assignment 03 report.

## Problem

```
minimize    x^2
subject to  x >= 1
```

Lagrangian: `L(mu, x) = x^2 + mu(1 - x)`, `mu >= 0`. Results: `x(mu) = mu/2`, `d(mu) = mu - mu^2/4`, `mu* = 2`, `d* = p* = 1`.

## Files

| File | Purpose |
| ---- | ------- |
| `en3330_a03_verify.py` | SymPy + numerical checks of every result (Q1-Q3), prints PASS/FAIL |
| `en3330_a03_figures.py` | Draws all report figures from the functions into `images/` |
| `images/` | Generated graphs (already included) |
| `requirements.txt` | Python packages |

| Graph | Question | Shows |
| ----- | -------- | ----- |
| `q1a_lagrangian.png` | Q1(a) | `L(mu,x)` for `mu = 0,1,2,3` with minimizers |
| `q1b_dual_function.png` | Q1(b) | `d(mu)` with maximum at `mu* = 2` |
| `q1c_primal_function.png` | Q1(c) | `g(x)`: `x^2` for `x >= 1`, `+inf` for `x < 1` |
| `q2a_left_inequality.png`, `q2a_right_inequality.png` | Q2(a) | `L(mu,x*)` vs `mu` and `L(mu*,x)` vs `x` |
| `q3b_lower_envelope.png` | Q3(b) | Lines for `x = 0,1,2`, lower envelope and `d(mu)` |

## How to run

```
pip install -r requirements.txt
python en3330_a03_verify.py
python en3330_a03_figures.py
```

The graphs are saved as PNG files in `images/`, next to the scripts.
