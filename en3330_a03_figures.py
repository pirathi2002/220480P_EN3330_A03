"""Generates all report figures from the functions L(mu,x)=x^2+mu(1-x).
Run:  python en3330_a03_figures.py   (writes ./images/*.png)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

L = lambda m, x: x**2 + m*(1 - x)
d = lambda m: m - m**2/4          # analytic dual function
cols = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

# ---- Q1(a)
x = np.linspace(-1, 3, 400)
fig, ax = plt.subplots(figsize=(7, 4.5))
for m, c in zip(range(4), cols):
    xm = m/2
    ax.plot(x, L(m, x), color=c, label=fr"$\mu={m}$, minimizer $x(\mu)={xm:g}$")
    ax.plot(xm, L(m, xm), "o", color=c, ms=8, mec="k")
ax.axvline(1, color="gray", ls="--", lw=1)
ax.text(1.05, 9.3, r"$x=1$ (boundary)", fontsize=9, color="gray")
ax.set_ylim(-0.5, 10); ax.set_xlabel("$x$"); ax.set_ylabel(r"$L(\mu,x)$")
ax.set_title(r"Lagrangian $L(\mu,x)=x^2+\mu(1-x)$ with minimizers marked")
ax.legend(loc="upper left"); fig.savefig(f"{OUT}/q1a_lagrangian.png"); plt.close(fig)

# ---- Q1(b)
m = np.linspace(0, 5, 400)
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(m, d(m), color="tab:blue", label=r"$d(\mu)=\mu-\mu^2/4$")
ax.plot(2, d(2), "o", color="tab:red", ms=9, mec="k", label=r"$\mu^*=2,\ d^*=1$")
ax.axhline(1, color="gray", ls="--", lw=1, label=r"$p^*=1$")
ax.annotate(r"$(\mu^*,d^*)=(2,1)$", (2, 1), textcoords="offset points", xytext=(12, -22))
ax.set_xlabel(r"$\mu$"); ax.set_ylabel(r"$d(\mu)$"); ax.set_ylim(-1.5, 1.4)
ax.set_title(r"Dual function ($\mu\geq 0$)"); ax.legend(loc="lower left")
fig.savefig(f"{OUT}/q1b_dual_function.png"); plt.close(fig)

# ---- Q1(c)
x = np.linspace(-1, 3, 800)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
for mv, c in zip([0, 1, 3, 10, 40], plt.cm.viridis(np.linspace(0, 0.9, 5))):
    a1.plot(x, L(mv, x), color=c, label=fr"$\mu={mv}$")
a1.axvline(1, color="k", ls="--", lw=1)
a1.set_ylim(0, 10); a1.set_xlabel("$x$"); a1.set_ylabel(r"$L(\mu,x)$")
a1.set_title(r"$L(\mu,x)$ for increasing $\mu$"); a1.legend(fontsize=8, loc="upper right")
a1.text(-0.95, 0.4, r"$x<1$: $L\to\infty$ as $\mu\to\infty$", fontsize=9)
a1.text(1.1, 0.4, r"$x\geq1$: max at $\mu=0$", fontsize=9)
xf = np.linspace(1, 3, 300)
a2.plot(xf, xf**2, color="tab:blue", lw=2, label=r"$g(x)=x^2$ (feasible, $x\geq1$)")
a2.axvspan(-1, 1, color="tab:red", alpha=0.12)
for xa in (0, -0.6):
    a2.annotate("", xy=(xa, 9.8), xytext=(xa, 5), arrowprops=dict(arrowstyle="-|>", color="tab:red", lw=2, mutation_scale=18))
a2.text(-0.95, 3.5, r"infeasible: $g(x)=+\infty$", color="tab:red", fontsize=10)
a2.plot(1, 1, "o", color="tab:red", mec="k", ms=8, label=r"minimum: $x^*=1$, $g=1$")
a2.set_xlim(-1, 3); a2.set_ylim(0, 10); a2.set_xlabel("$x$"); a2.set_ylabel(r"$g(x)$")
a2.set_title(r"Primal function $g(x)=\max_{\mu\geq0}L(\mu,x)$"); a2.legend(loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/q1c_primal_function.png"); plt.close(fig)

# ---- Q2(a)
m = np.linspace(0, 5, 200)
fig, ax = plt.subplots(figsize=(4.6, 3.3))
ax.plot(m, L(m, 1.0), color="tab:blue", label=r"$L(\mu,x^*)=1$")
ax.plot(2, L(2, 1.0), "o", color="tab:red", mec="k", ms=8, label=r"$L(\mu^*,x^*)=1$")
ax.set_xlabel(r"$\mu$"); ax.set_ylabel(r"$L(\mu,x^*)$"); ax.set_ylim(0, 2)
ax.set_title(r"Left inequality ($x^*=1$)", fontsize=11); ax.legend(loc="upper right", fontsize=9)
fig.savefig(f"{OUT}/q2a_left_inequality.png"); plt.close(fig)

x = np.linspace(-1, 3, 300)
fig, ax = plt.subplots(figsize=(4.6, 3.3))
ax.plot(x, L(2.0, x), color="tab:blue", label=r"$L(\mu^*,x)=(x-1)^2+1$")
ax.plot(1, L(2.0, 1.0), "o", color="tab:red", mec="k", ms=8, label=r"$L(\mu^*,x^*)=1$")
ax.axhline(1, color="gray", ls="--", lw=1)
ax.set_xlabel("$x$"); ax.set_ylabel(r"$L(\mu^*,x)$"); ax.set_ylim(0, 6)
ax.set_title(r"Right inequality ($\mu^*=2$)", fontsize=11); ax.legend(loc="upper center", fontsize=9)
fig.savefig(f"{OUT}/q2a_right_inequality.png"); plt.close(fig)

# ---- Q3(b)
m = np.linspace(0, 4, 800)
env = np.min([L(m, xv) for xv in (0, 1, 2)], axis=0)
assert np.allclose(env, np.minimum.reduce([m, np.ones_like(m), 4 - m]))
fig, ax = plt.subplots(figsize=(7, 4.4))
labs = {0: r"$x=0$: $L=\mu$", 1: r"$x=1$: $L=1$", 2: r"$x=2$: $L=4-\mu$"}
for xv, c in zip((0, 1, 2), cols):
    ax.plot(m, L(m, xv), color=c, lw=1.4, label=labs[xv])
ax.plot(m, env, "k", lw=3, alpha=0.8, label=r"lower envelope $\min(\mu,\,1,\,4-\mu)$")
ax.plot(m, d(m), "m--", lw=1.8, label=r"true $d(\mu)=\mu-\mu^2/4$")
ax.plot(2, 1, "o", color="tab:red", mec="k", ms=8)
ax.set_ylim(-0.5, 4.2); ax.set_xlabel(r"$\mu$"); ax.set_ylabel(r"$L(\mu,x)$")
ax.set_title("Lagrangian lines and their lower envelope"); ax.legend(loc="upper right", fontsize=9)
fig.savefig(f"{OUT}/q3b_lower_envelope.png"); plt.close(fig)
print("figures written to", os.path.normpath(OUT))
