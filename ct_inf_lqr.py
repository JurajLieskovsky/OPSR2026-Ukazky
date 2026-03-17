import autograd
import autograd.numpy as np
import scipy
import matplotlib.pyplot as plt

# Dynamický model
import birotor_with_payload_dynamics as dyn

# Rovnovážný stav
x_eq = np.zeros(8)
u_eq = dyn.g * (dyn.m_P + dyn.m_Q) / 2 * np.ones(2)

dfdx = autograd.jacobian(lambda x: dyn.f(0.0, x, u_eq))
dfdu = autograd.jacobian(lambda u: dyn.f(0.0, x_eq, u))

A = dfdx(x_eq)
B = dfdu(u_eq)

# Návrh LQR
Q = 1e1 * np.identity(8)
R = np.identity(2)

P = scipy.linalg.solve_continuous_are(A, B, Q, R)
K = np.linalg.solve(R, B.T @ P)

# Simulace
x0 = np.array([1, -0.1, 0, 0, 0, 0, 0, 0])
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
plt.show()
