import numpy as np
import matplotlib.pyplot as plt

def double_pendulum_derivs(state, t):
    """Calcula las derivadas de las variables de estado para un péndulo doble
    state: arreglo de variables de estado [theta1, omega1, theta2, omega2]
    t: tiempo
    returns: derivadas de las variables de estado [dtheta1/dt, domega1/dt, dtheta2/dt, domega2/dt]
    """
    l = 1.0 # longitud del péndulo
    g = 9.81 # aceleración debida a la gravedad
    m = 1.0 # masa de las partículas
    
    theta1, omega1, theta2, omega2 = state
    
    dtheta1_dt = omega1
    domega1_dt = (-g*(2*m + m)*np.sin(theta1) - m*g*np.sin(theta1 - 2*theta2) - 2*m*np.sin(theta1 - theta2)*(omega2**2*l + omega1**2*l*np.cos(theta1 - theta2))) / (l*(2*m + m - m*np.cos(2*theta1 - 2*theta2)))
    dtheta2_dt = omega2
    domega2_dt = (2*np.sin(theta1 - theta2)*((omega1**2)*l*(m + m) + g*(m + m)*np.cos(theta1) + omega2**2*l*m*np.cos(theta1 - theta2))) / (l*(2*m + m - m*np.cos(2*theta1 - 2*theta2)))
    
    return [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]

# Condiciones iniciales: theta1=0.5, omega1=0, theta2=0.5, omega2=0
init_state = [0.5, 0, 0.5, 0]

# Tiempo de integración de 0 a 10 segundos
t = np.linspace(0, 10, 10001)

# Integración de las ecuaciones de movimiento con RK4
from scipy.integrate import odeint
state = odeint(double_pendulum_derivs, init_state, t)

# Extracción de los ángulos de los péndulos
theta1 = state[:, 0]
theta2 = state[:, 2]

# Cálculo de las posiciones de las partículas a partir de los ángulos
x1 = np.sin(theta1)
y1 = -np.cos(theta1)
x2 = x1 + np.sin(theta2)
y2 = y1 - np.cos(theta2)

# Gráfico de la trayectoria del péndulo doble
fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
line, = ax.plot([], [], 'o-', lw=2)

def init():
    line.set_data([], [])
    return line,

def animate(i):
    line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
    return line,

from matplotlib.animation import FuncAnimation
ani = FuncAnimation(fig, animate, frames=len(t), interval=10, blit=True, init_func=init)

plt.show()

# Cálculo
