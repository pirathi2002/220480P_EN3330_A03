"""Symbolic + numerical verification for EN3330 Assignment 03.
Everything is computed from the definitions L(mu,x) = x^2 + mu(1-x).
Run:  python en3330_a03_verify.py
"""
import numpy as np
import sympy as sp

mu, x = sp.symbols('mu x', real=True)
f = x**2
h = 1 - x
L = f + mu*h
ok = lambda b: "PASS" if b else "FAIL"

print("=== Q1(a) minimisers of L(mu,x) ===")
dL = sp.diff(L, x)
xmin = sp.solve(dL, x)[0]
print("dL/dx =", dL, "| d2L/dx2 =", sp.diff(L, x, 2), "| x(mu) =", xmin)
for m in range(4):
    xm = xmin.subs(mu, m)
    print(f"  mu={m}: x(mu)={xm}, L(mu,x(mu))={L.subs({mu: m, x: xm})}")

print("\n=== Q1(b) dual function ===")
d = sp.expand(L.subs(x, xmin))
print("d(mu) =", d)
print("identity d = 1-(mu-2)^2/4 :", ok(sp.simplify(d - (1 - (mu-2)**2/4)) == 0))
mus = sp.solve(sp.diff(d, mu), mu)[0]
print("mu* =", mus, "| d* =", d.subs(mu, mus), "| d'' =", sp.diff(d, mu, 2), "(concave)")
xs = np.linspace(-5, 8, 130001)
mgrid = np.linspace(0, 6, 601)
dnum = np.array([np.min(xs**2 + m*(1 - xs)) for m in mgrid])
dfun = sp.lambdify(mu, d)
print("max |d_numeric - d_analytic| =", np.max(np.abs(dnum - dfun(mgrid))))
print("numeric argmax mu =", mgrid[np.argmax(dnum)], " numeric d* =", dnum.max())
pstar = float(np.min(xs[xs >= 1]**2))
print("primal optimum p* =", pstar, "at x=1")

print("\n=== Q1(c) primal function g(x)=max_{mu>=0} L ===")
for xv in [-1, 0, 0.5, 1, 2, 3]:
    vals = [float(L.subs({mu: m, x: xv})) for m in [0, 1, 10, 100, 1e4]]
    print(f"  x={xv}: L at mu=0,1,10,100,1e4 -> {vals}  ->",
          "g = x^2 = " + str(xv**2) if xv >= 1 else "grows without bound: g=+inf")

print("\n=== Q2 saddle point (x*,mu*)=(1,2) ===")
xs_, ms_ = 1, 2
print("(i)   L(mu*,x*) =", L.subs({mu: ms_, x: xs_}))
print("(ii)  L(mu,x*) =", sp.simplify(L.subs(x, xs_)), "(independent of mu)")
print("(iii) L(mu*,x) =", sp.expand(L.subs(mu, ms_)),
      "| equals (x-1)^2+1 :", ok(sp.simplify(L.subs(mu, ms_) - ((x-1)**2 + 1)) == 0))
mg = np.linspace(0, 1e3, 200001); xg = np.linspace(-1e2, 1e2, 200001)
left = np.max(mg*(1 - xs_) + xs_**2 - 1)
right = np.min(xg**2 + ms_*(1 - xg) - 1)
print("grid check left  : max[L(mu,x*)-L(mu*,x*)] =", left, ok(left <= 1e-12))
print("grid check right : min[L(mu*,x)-L(mu*,x*)] =", right, ok(right >= -1e-12))
print("-> premise (x*,mu*)=(1,2) is CONSISTENT: genuine saddle point; no inconsistency found.")
print("complementary slackness mu* h(x*) =", ms_*h.subs(x, xs_))

print("\n=== Q2(c) modified constraint x >= -1, h2 = -1 - x ===")
h2 = -1 - x
L2 = f + mu*h2
print("h2(0) =", h2.subs(x, 0), "-> feasible, inactive (strict)")
x2 = sp.solve(sp.diff(L2, x), x)[0]
d2 = sp.expand(L2.subs(x, x2))
print("x(mu) =", x2, "| d2(mu) =", d2, "| d2'(mu) =", sp.diff(d2, mu))
print("d2'(mu)<0 for all mu>=0 -> argmax at mu*=0, d2* =", d2.subs(mu, 0), "= p* = 0")
print("left ineq: L2(mu,0) =", sp.expand(L2.subs(x, 0)), "<= L2(0,0) =", L2.subs({mu: 0, x: 0}))

print("\n=== Q3(a) weak duality check ===")
print(" mu   d(mu)   p*   p*-d(mu)=(mu-2)^2/4   d<=p*")
for m in range(4):
    dv = d.subs(mu, m)
    print(f" {m}   {float(dv):.4f}   1   {float(1-dv):.4f}   {ok(dv <= 1)}")

print("\n=== Q3(b) lines and lower envelope ===")
for xv in (0, 1, 2):
    print(f"  x={xv}: L(mu,{xv}) =", sp.expand(L.subs(x, xv)))
mm = np.linspace(0, 4, 4001)
env = np.minimum.reduce([mm, np.ones_like(mm), 4 - mm])
print("envelope = min(mu,1,4-mu): breakpoints mu=1 and mu=3")
print("envelope concave (second differences <= 0):", ok(np.all(np.diff(env, 2) <= 1e-12)))
print("d(mu) <= envelope everywhere:", ok(np.all(dfun(mm) <= env + 1e-12)),
      "| touching at mu =", mm[np.isclose(dfun(mm), env)])
