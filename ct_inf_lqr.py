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
x0 = np.array([3, 1, 0, 0, 0, 0, 0, 0])
T = 5

sol = scipy.integrate.solve_ivp(
    lambda t, x: dyn.f(t, x, u_eq - K @ (x - x_eq)),
    [0, T],
    x0,
    dense_output=True,
)

# Vizualizace
tspan = np.linspace(0, T, 100)

fig, ax = plt.subplots()

for i in range(4):
    plt.plot(tspan, sol.sol(tspan)[i, :], label=f"x{i}")

ax.legend()
plt.show(block=False)

# Animation
visualizer = meshcat.Visualizer()

vis.set_birotor(visualizer, 2 * dyn.a, 0.04, 0.09, dyn.l)

anim = meshcat.animation.Animation(default_framerate=len(tspan) / T)

for i, t in enumerate(tspan):
    with anim.at_frame(visualizer, i) as frame:
        vis.set_birotor_state(frame, sol.sol(t))

visualizer.set_animation(anim, play=False)

input("Press Enter to continue...")
