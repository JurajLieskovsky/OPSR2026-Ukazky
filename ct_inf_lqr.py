import numpy as np
import scipy
import matplotlib.pyplot as plt

import meshcat

# Dynamický model
import birotor_with_payload_dynamics as dyn

# Visualizér
import birotor_with_payload_visualizer as vis

# Rovnovážný stav
x_eq = np.array([0.0, 2, 0, 0, 0, 0, 0, 0])
u_eq = dyn.g * (dyn.m_P + dyn.m_Q) / 2 * np.ones(2)

A, B = dyn.df(0, x_eq, u_eq)

# Návrh LQR
Q = 1e1 * np.identity(8)
R = np.identity(2)

P = scipy.linalg.solve_continuous_are(A, B, Q, R)
K = np.linalg.solve(R, B.T @ P)

# Simulace
T = 5

tspan = np.linspace(0, T, 100)
dspan = [np.random.normal(0.0, 1e0) for _ in tspan]

x0 = np.array([3, 1, 0, 0, 0, 0, 0, 0])

sol = scipy.integrate.solve_ivp(
    lambda t, x: dyn.f(t, x, u_eq - K @ (x - x_eq), np.array([np.interp(t, tspan, dspan)])),
    [0, T],
    x0,
    dense_output=True,
)

# Vizualizace

xs = [sol.sol(t) for t in tspan]
us = [u_eq - K @ (x - x_eq) for x in xs]

fig, ax = plt.subplots(2)

for i in range(4):
    ax[0].plot(tspan, [x[i] for x in xs], label=f"x{i}")

for i in range(2):
    ax[1].plot(tspan, [u[i] for u in us], label=f"u{i}")

ax[0].legend()
ax[1].legend()
plt.show(block=False)

# Animation
visualizer = meshcat.Visualizer()

vis.set_birotor(visualizer, 2 * dyn.a, 0.04, 0.09, dyn.l)
vis.set_birotor_state(visualizer, x0)

anim = meshcat.animation.Animation(default_framerate=len(tspan) / T)

for i, t in enumerate(tspan):
    with anim.at_frame(visualizer, i) as frame:
        vis.set_birotor_state(frame, sol.sol(t))

visualizer.set_animation(anim, play=False)

input("Press Enter to continue...")
