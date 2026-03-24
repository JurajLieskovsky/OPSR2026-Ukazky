import numpy as np
import scipy
import matplotlib.pyplot as plt

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

# Návrh LQR
Q = 1e1 * np.identity(8)
R = np.identity(2)

P = scipy.linalg.solve_discrete_are(A, B, Q, R)
K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)

# Simulace
N = 500
x0 = np.array([3, 1, 0, 0, 0, 0, 0, 0])

solver = scipy.integrate.ode(dyn.f)
solver.set_integrator("dopri5")
solver.set_initial_value(x0)

xs = [np.zeros(8) for _ in range(N + 1)]
us = [np.zeros(2) for _ in range(N + 1)]

xs[0] = solver.y

for k in range(N):
    u = u_eq - K @ (solver.y - x_eq)
    solver.set_f_params(u)
    solver.integrate(solver.t + h)

    us[k] = u
    xs[k + 1] = solver.y

us[N] = us[N-1]

# Vizualizace
tspan = np.linspace(0, h * N, N + 1)

fig, ax = plt.subplots(2)

for i in range(4):
    ax[0].plot(tspan, [x[i] for x in xs], label=f"x{i}")

for i in range(2):
    ax[1].step(tspan, [u[i] for u in us], where='post', label=f"u{i}")

ax[0].legend()
ax[1].legend()
plt.show(block=False)

# Animation
visualizer = meshcat.Visualizer()

vis.set_birotor(visualizer, 2 * dyn.a, 0.04, 0.09, dyn.l)

anim = meshcat.animation.Animation(default_framerate=1 / h)

for i, x in enumerate(xs):
    with anim.at_frame(visualizer, i) as frame:
        vis.set_birotor_state(frame, x)

visualizer.set_animation(anim, play=False)

input("Press Enter to continue...")
