import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

m = 10.0
p = 0.1
k = 3.0

def y_dot(v):
    return v

def v_dot(y, v):
    return -(p / m) * v - (k / m) * y

def system(state, t):
    y, v = state
    dydt = v
    dvdt = -(p/m) * v - (k/m) * y
    return [dydt, dvdt]

def check():
    global V, Y, dt, T
    solution = odeint(system, [1.0, 0.0], T)

    y_odeint = solution[:, 0]
    v_odeint = solution[:, 1]

    if not np.allclose(y_odeint, Y, atol=dt):
        print("y not equal")
    if not np.allclose(v_odeint, V, atol=dt):
        print("v not equal")

    fig, ax = plt.subplots(3)
    fig.set_figheight(10)
    fig.set_figwidth(12)

    ax[0].plot(T, Y, label='y')
    ax[0].plot(T, V, label='dy/dt')
    ax[0].legend()

    ax[1].plot(T, y_odeint, label='y odeint')
    ax[1].plot(T, Y, label='y Runge-Kutta', linestyle='dashed')
    ax[1].legend()

    ax[2].plot(T, v_odeint, label='dy/dt odeint')
    ax[2].plot(T, V, label='dy/dt Runge-Kutta', linestyle='dashed')
    ax[2].legend()
    
    plt.show()

x0 = -10.0
x1 = 10.0
dt = 0.001

n = int((x1 - x0) / dt)
T = np.linspace(x0, x1, n)
Y = np.zeros(n)
V = np.zeros(n)

Y[0] = 1.0  # Начальные условия
V[0] = 0.0

for i in range(1, n):
    k1_y = dt * y_dot(V[i-1])
    k1_v = dt * v_dot(Y[i-1], V[i-1])
    
    k2_y = dt * y_dot(V[i-1] + 0.5 * k1_v)
    k2_v = dt * v_dot(Y[i-1] + 0.5 * k1_y, V[i-1] + 0.5 * k1_v)
    
    k3_y = dt * y_dot(V[i-1] + 0.5 * k2_v)
    k3_v = dt * v_dot(Y[i-1] + 0.5 * k2_y, V[i-1] + 0.5 * k2_v)
    
    k4_y = dt * y_dot(V[i-1] + k3_v)
    k4_v = dt * v_dot(Y[i-1] + k3_y, V[i-1] + k3_v)
    
    Y[i] = Y[i-1] + (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
    V[i] = V[i-1] + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6

check()