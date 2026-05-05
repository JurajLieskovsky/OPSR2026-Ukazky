from car_import import constraints
import numpy as np
import scipy
import matplotlib.pyplot as plt
import cvxpy as cp

import meshcat

# Dynamický model
import birotor_with_payload_dynamics as dyn

# Visualizér
import birotor_with_payload_visualizer as vis

# Časový krok
h = 1e-2

# Rovnovážný stav
x_eq = np.array([0.0, 2, 0, 0, 0, 0, 0, 0])
u_eq = dyn.g * (dyn.m_P + dyn.m_Q) / 2 * np.ones(2)

cont_A, cont_B = dyn.df(0, x_eq, u_eq)

A = np.identity(8) + h * cont_A
B = h * cont_B

# Návrh LQR s nekonečným horizontem
Q = 1e1 * np.identity(8)
R = np.identity(2)

infP = scipy.linalg.solve_discrete_are(A, B, Q, R)
infK = np.linalg.solve(R + B.T @ infP @ B, B.T @ infP @ A)

# Návrh LQR s konečným horizontem
N = 500

x = cp.Variable((8, N + 1))
u = cp.Variable((2, N))
x_init = cp.Parameter(8)

constraints = [
    x[:, 1:] == A @ x[:, :-1] + B @ u,
    x[:, 0] == x_init,
    u >= -u_eq[:, np.newaxis],
]

LQ = np.linalg.cholesky(Q)
LR = np.linalg.cholesky(R)

objective = cp.Minimize(
    cp.sum_squares(LQ.T @ x[:, :-1])
    + cp.sum_squares(LR.T @ u)
    + cp.quad_form(x[:, N], infP)
)

problem = cp.Problem(objective, constraints)

x0 = np.array([5, 5, 0, 0, 0, 0, 0, 0])
x_init.value = x0 - x_eq
problem.solve()

# Vizualizace
tspan = np.linspace(0, h * N, N + 1)

xs = [x_eq + x.value[:, k] for k in range(N + 1)]
us = [u_eq + u.value[:, k] for k in range(N)]

us.append(us[N - 1])

fig, ax = plt.subplots(2)

for i in range(4):
    ax[0].plot(tspan, [x[i] for x in xs], label=f"x{i}")

for i in range(2):
    ax[1].step(tspan, [u[i] for u in us], where="post", label=f"u{i}")

ax[0].legend()
ax[1].legend()
plt.show(block=False)

# Animation
visualizer = meshcat.Visualizer()

vis.set_birotor(visualizer, 2 * dyn.a, 0.04, 0.09, dyn.l)
vis.set_birotor_state(visualizer, x0)

anim = meshcat.animation.Animation(default_framerate=1 / h)

for i, x in enumerate(xs):
    with anim.at_frame(visualizer, i) as frame:
        vis.set_birotor_state(frame, x)

visualizer.set_animation(anim, play=False)

input("Press Enter to continue...")
