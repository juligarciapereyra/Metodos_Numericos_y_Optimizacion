import numpy as np
import matplotlib.pyplot as plt
from pylab import *


# Define las constantes y las condiciones iniciales
L = 1  # longitud del péndulo
g = 9.81  # aceleración debida a la gravedad
theta1_0 = np.pi/2 # ángulo inicial del primer péndulo
theta2_0 = np.pi/2  # ángulo inicial del segundo péndulo
omega1_0 = 0  # velocidad angular inicial del primer péndulo
omega2_0 = 0  # velocidad angular inicial del segundo péndulo
t_0 = 0  # tiempo inicial
t_f = 10  # tiempo final
dt = 0.001  # paso de tiempo

# Define las funciones para las ecuaciones diferenciales
def f(theta1, theta2, omega1, omega2, t):
    delta = theta1 - theta2
    alpha = (omega1**2)*np.sin(delta)*np.cos(delta) - (g/L)*np.sin(theta1)
    beta = (omega2**2)*np.sin(delta)*np.cos(delta) - (g/L)*np.sin(theta2)
    gamma = 2*np.sin(delta)*np.cos(delta) - (np.cos(theta1 - theta2)**2)
    omega1_dot = (beta*np.sin(delta) - alpha*np.cos(delta))/gamma
    omega2_dot = (alpha*np.sin(delta) - beta*np.cos(delta))/gamma
    return omega1, omega2, omega1_dot, omega2_dot

# Define la función de RK4
def RK4(f, theta1, theta2, omega1, omega2, t, dt):
    k1 = dt*np.array(f(theta1, theta2, omega1, omega2, t))
    k2 = dt*np.array(f(theta1 + 0.5*k1[0], theta2 + 0.5*k1[1], omega1 + 0.5*k1[2], omega2 + 0.5*k1[3], t + 0.5*dt))
    k3 = dt*np.array(f(theta1 + 0.5*k2[0], theta2 + 0.5*k2[1], omega1 + 0.5*k2[2], omega2 + 0.5*k2[3], t + 0.5*dt))
    k4 = dt*np.array(f(theta1 + k3[0], theta2 + k3[1], omega1 + k3[2], omega2 + k3[3], t + dt))
    theta1_new = theta1 + (1/6)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    theta2_new = theta2 + (1/6)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    omega1_new = omega1 + (1/6)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    omega2_new = omega2 + (1/6)*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
    return theta1_new, theta2_new, omega1_new, omega2_new

# Inicializa los vectores para almacenar los resultados
t_values = np.arange(t_0, t_f, dt)
theta1_values = np.zeros_like(t_values)
theta2_values = np.zeros_like(t_values)
omega1_values = np.zeros_like(t_values)
omega2_values = np.zeros_like(t_values)


# Usa RK4 para calcular las soluciones numéricas
theta1 = theta1_0
theta2 = theta2_0
omega1 = omega1_0
omega2 = omega2_0
for i, t in enumerate(t_values):
    theta1_values[i] = theta1
    theta2_values[i] = theta2
    omega1_values[i] = omega1
    omega2_values[i] = omega2

    theta1, theta2, omega1, omega2 = RK4(f, theta1, theta2, omega1, omega2, t, dt)

    # Calcular la energía total, cinética y potencial
    m = 3  # masa inicial
    T_values = 0.5*m*(L**2)*(omega1_values**2 + omega2_values**2)
    V_values = m*g*L*(2*np.cos(theta1_values) + np.cos(theta2_values))
    E_values = T_values + V_values


    #Caluclo las energias de cada pendulo por separado 
    T1_values = 0.5*m*(L**2)*(omega1_values**2)
    V1_values = m*g*L*(2*np.cos(theta1_values))
    E1_values = T1_values + V1_values

    T2_values = 0.5*m*(L**2)*(omega2_values**2)
    V2_values = m*g*L*(2*np.cos(theta2_values))
    E2_values = T2_values + V2_values



# Calcula las posiciones x, y del péndulo doble
x1 = L*np.sin(theta1_values)
y1 = -L*np.cos(theta1_values)
x2 = x1 + L*np.sin(theta2_values)
y2 = y1 - L*np.cos(theta2_values)

# Grafica la trayectoria del péndulo doble
fig, ax = plt.subplots()
ax.set_aspect('equal', 'datalim')
ax.plot(x1, y1, label='Péndulo 1')
ax.plot(x2, y2, label='Péndulo 2')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


#Grafica angulo en funcion del tiempo de cada pendulo
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

ax1.plot(t_values, theta1_values)
ax1.set_ylabel('Ángulo péndulo 1')

ax2.plot(t_values, theta2_values)
ax2.set_xlabel('Tiempo')
ax2.set_ylabel('Ángulo péndulo 2')

plt.show()


# Graficar las energías cinética, potencial y total en función del tiempo
#fig, (ax3, ax4) = plt.subplots(nrows=2, sharex=True)

fig, ax3 = plt.subplots()

ax3.plot(t_values, T_values)
ax3.plot(t_values, V_values)
ax3.plot(t_values, E_values)
ax3.legend(['Energía cinética', 'Energía potencial', 'Energía total'])
ax3.set_ylabel('Energía')

'''
ax4.plot(t_values, theta1_values)
ax4.plot(t_values, theta2_values)
ax4.set_xlabel('Tiempo')
ax4.set_ylabel('Ángulo')
ax4.legend(['Péndulo 1', 'Péndulo 2'])
'''
plt.show()


# Graficar las energicas de cada pendulo por separado
fig, (ax4, ax5) = plt.subplots(nrows=2, sharex=True)

ax4.plot(t_values, T1_values)
ax4.plot(t_values, V1_values)
ax4.plot(t_values, E1_values)
ax4.legend(['Energía cinética', 'Energía potencial', 'Energía total'])
ax4.set_ylabel('Energía Pendulo 1')

ax5.plot(t_values, T2_values)
ax5.plot(t_values, V2_values)
ax5.plot(t_values, E2_values)
ax5.legend(['Energía cinética', 'Energía potencial', 'Energía total'])
ax5.set_ylabel('Energía Pendulo 2')

plt.show()
















