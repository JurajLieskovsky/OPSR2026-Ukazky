import numpy as np
import scipy
import matplotlib.pyplot as plt

import meshcat

# Dynamický model
import birotor_with_payload_dynamics as dyn

# Visualizér
import birotor_with_payload_visualizer as vis


# Generátor trajektorie
def ref_state(k, N):
    return np.array(
        [
            np.sin(2 * np.pi * k / N),
            np.cos(2 * np.pi * k / N),
            0,
            0,
            2 * np.pi / (h * N) * np.cos(2 * np.pi * k / N),
            -2 * np.pi / (h * N) * np.sin(2 * np.pi * k / N),
            0,
            0,
        ]
    )

# Časový krok a délka horizontu
h = 1e-2
N = 500

# Dynamika v diskrétním čase
def dt_dynamics(k, x, u):
    return x + h * dyn.f(k * h, x, u)

# Rovnovážný stav
x_eq = np.zeros(8)
u_eq = dyn.g * (dyn.m_P + dyn.m_Q) / 2 * np.ones(2)

cont_A, cont_B = dyn.df(0, x_eq, u_eq)

A = np.identity(8) + h * cont_A
B = h * cont_B

# Návrh LQR s nekonečným horizontem
Q = 1e1 * np.diag([10.0, 10, 10, 10, 1, 1, 1, 1])
R = np.identity(2)

infP = scipy.linalg.solve_discrete_are(A, B, Q, R)
infK = np.linalg.solve(R + B.T @ infP @ B, B.T @ infP @ A)

# Návrh LQR s konečným horizontem
d = [np.zeros(2) for _ in range(N)]
K = [np.zeros((2, 8)) for _ in range(N)]

p = np.zeros(8)
P = infP

for k in reversed(range(N)):
    c = dt_dynamics(k, ref_state(k, N), u_eq) - ref_state(k+1, N)
    q = np.zeros(8)
    r = np.zeros(2)

    M = R + B.T @ P @ B
    invM = np.linalg.inv(M)

    d[k] = invM @ (r + B.T @ (p + P @ c))
    K[k] = invM @ B.T @ P @ A

    p = (q + A.T @ (p + P @ c)) - K[k].T @ M @ d[k]
    P = Q + A.T @ P @ A - K[k].T @ M @ K[k]

# Simulace
x0 = ref_state(0, N)

solver = scipy.integrate.ode(dyn.f)
solver.set_integrator("dopri5")
solver.set_initial_value(x0)

xs = [np.zeros(8) for _ in range(N + 1)]
us = [np.zeros(2) for _ in range(N + 1)]
cs = np.zeros(N + 1)

xs[0] = solver.y

for k in range(N):
    u = u_eq - d[k] - K[k] @ (solver.y - ref_state(k, N))
    # solver.set_f_params(u, np.random.normal(0.0, 1e0, 1))
    solver.set_f_params(u)
    solver.integrate(solver.t + h)

    us[k] = u
    xs[k + 1] = solver.y

us[N] = us[N - 1]

# Vizualizace
tspan = np.linspace(0, h * N, N + 1)

# Série
fig1, ax1 = plt.subplots(2)

for i in range(4):
    ax1[0].plot(tspan, [x[i] for x in xs], label=f"x{i}")
    ax1[0].plot(tspan, [ref_state(k, N)[i] for k in range(N + 1)], label=f"x{i}_ref")

for i in range(2):
    ax1[1].step(tspan, [u[i] for u in us], where="post", label=f"u{i}")

ax1[0].legend()
ax1[1].legend()

# Prostor
fig2, ax2 = plt.subplots()

ax2.plot(
    [ref_state(k, N)[0] for k in range(N + 1)],
    [ref_state(k, N)[1] for k in range(N + 1)],
    label="x_ref",
)
ax2.plot([x[0] for x in xs], [x[1] for x in xs], label="x")

ax2.set_aspect("equal", "box")
ax2.legend()

# Show
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
