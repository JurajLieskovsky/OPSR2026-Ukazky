import numpy as np
import scipy
import matplotlib.pyplot as plt

# Dynamický model
import birotor_with_payload_dynamics as dyn

# Rovnovážný stav
x_eq = np.zeros(8)
u_eq = dyn.g * (dyn.m_P + dyn.m_Q) / 2 * np.ones(2)

# Simulace
T = 1

sol = scipy.integrate.solve_ivp(
    lambda t, x: dyn.f(t, x, u_eq),
    [0, T],
    x_eq,
    dense_output=True,
)

# Vizualizace
tspan = np.linspace(0, T, 100)

fig, ax = plt.subplots()

for i in range(4):
    plt.plot(tspan, sol.sol(tspan)[i, :], label=f"x{i}")

ax.legend()
plt.show()
