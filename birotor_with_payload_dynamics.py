import numpy as np

## Parametry
g = 9.81  # m / s^2
m_Q = 0.5  # kg
I_Q = 4e-3  # kg * m^2
a = 0.175  # m

m_P = 0.1  # kg
l = 0.5  # m


## Dynamika
def f(t, x, u):
    (y, z, theta, phi, y_dot, z_dot, theta_dot, phi_dot) = x

    # Mass matrix
    M = np.array(
        [
            [m_P + m_Q, 0, 0, l * m_P * np.cos(phi)],
            [0, m_P + m_Q, 0, l * m_P * np.sin(phi)],
            [0, 0, I_Q, 0],
            [l * m_P * np.cos(phi), l * m_P * np.sin(phi), 0, l**2 * m_P],
        ]
    )

    # Coriolis/centripetal vector
    c = np.array(
        [
            -l * m_P * np.sin(phi) * phi_dot**2,
            l * m_P * np.cos(phi) * phi_dot**2,
            0,
            0,
        ]
    )

    # Potential/gravity vector
    tau_p = np.array(
        [
            0,
            g * (-m_P - m_Q),
            0,
            -g * l * m_P * np.sin(phi),
        ]
    )

    # Input matrix
    B = np.array(
        [
            [-np.sin(theta), -np.sin(theta)],
            [np.cos(theta), np.cos(theta)],
            [-a, a],
            [0, 0],
        ]
    )

    return np.concatenate(
        (
            x[4:],
            np.linalg.solve(M, -c + tau_p + B @ u),
        )
    )



