import numpy as np
from pylab import *
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


L = 1  
g = -9.8  
m = 3
theta1_0 = np.pi/4
theta2_0 = np.pi/6
omega1_0 = 0  
omega2_0 = 0  
h = 0.0001  


def f(ang1, ang2, w1, w2, t, length = 1):
    delta = ang1 - ang2
    f1 = ((-9.8/length)*((2*np.sin(ang1) - np.cos(delta)*np.sin(ang2))) - np.sin(delta)*(w2**2 + np.cos(delta)*w1**2))/(2-np.cos(delta)**2)
    f2 = ((-9.8/length)*((np.sin(ang2) - 2*np.cos(delta)*np.sin(ang1))) + np.sin(delta)*(w1**2 + np.cos(delta)*w2**2))/(2-np.cos(delta)**2)
    return w1, w2, f1, f2

def RK4(f, theta1, theta2, omega1, omega2, t, h):
    k1 = h*np.array(f(theta1, theta2, omega1, omega2, t))
    k2 = h*np.array(f(theta1 + 0.5*k1[0], theta2 + 0.5*k1[1], omega1 + 0.5*k1[2], omega2 + 0.5*k1[3], t + 0.5*h))
    k3 = h*np.array(f(theta1 + 0.5*k2[0], theta2 + 0.5*k2[1], omega1 + 0.5*k2[2], omega2 + 0.5*k2[3], t + 0.5*h))
    k4 = h*np.array(f(theta1 + k3[0], theta2 + k3[1], omega1 + k3[2], omega2 + k3[3], t + h))
    theta1_new = theta1 + (1/6)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    theta2_new = theta2 + (1/6)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    omega1_new = omega1 + (1/6)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    omega2_new = omega2 + (1/6)*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
    return theta1_new, theta2_new, omega1_new, omega2_new


t_values = np.arange(0, 10, h)
theta1_values = np.zeros_like(t_values)
theta2_values = np.zeros_like(t_values)
omega1_values = np.zeros_like(t_values)
omega2_values = np.zeros_like(t_values)

theta1 = theta1_0
theta2 = theta2_0
omega1 = omega1_0
omega2 = omega2_0
for i, t in enumerate(t_values):
    theta1_values[i] = theta1
    theta2_values[i] = theta2
    omega1_values[i] = omega1
    omega2_values[i] = omega2

    theta1, theta2, omega1, omega2 = RK4(f, theta1, theta2, omega1, omega2, t, h)


x1 = L*np.sin(theta1_values)
y1 = -L*np.cos(theta1_values)
x2 = x1 + L*np.sin(theta2_values)
y2 = y1 - L*np.cos(theta2_values)

fig, ax = plt.subplots()
ax.set_aspect('equal', 'datalim')
ax.plot(x1, y1, label='Particula 1')
ax.plot(x2, y2, label='Particula 2')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
show()


#Grafica angulo en funcion del tiempo de cada pendulo
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

ax1.plot(t_values, theta1_values)
ax1.set_ylabel('Ángulo particula 1')

ax2.plot(t_values, theta2_values)
ax2.set_xlabel('Tiempo')
ax2.set_ylabel('Ángulo particula 2')
show()


T1 = 0.5*m*(L**2)*(omega1_values**2)
V1 = -m*(9.8)*L*np.cos(theta1_values) + m*9.8*L
E1 = T1 + V1
plot(t_values, T1,  label='Energia cinetica P1')
plot(t_values, V1,  label='Energia potencial P1')
plot(t_values, E1,  label='Energia total P1')
xlabel('$t$', fontsize=10)
ylabel('$energia$', fontsize=10)
legend(fontsize=10)
show()


T2 = 0.5*m*(L**2)*(omega2_values**2)
V2 = -m*(9.8)*L*np.cos(theta2_values) + m*9.8*L
E2 = T2 + V2
plot(t_values, T2,  label='Energia cinetica P2')
plot(t_values, V2,  label='Energia potencial P2')
plot(t_values, E2,  label='Energia total P2')
xlabel('$t$', fontsize=10)
ylabel('$energia$', fontsize=10)
legend(fontsize=10)
show()

T = T1+T2
V = V1+V2
E = T + V
plot(t_values, T,  label='Energia cinetica TOTAL')
plot(t_values, V,  label='Energia potencial TOTAL')
plot(t_values, E,  label='Energia total TOTAL')
xlabel('$t$', fontsize=10)
ylabel('$energia$', fontsize=10)
legend(fontsize=10)
show()


fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-3, 3), ylim=(-3, 3))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'Time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    global x1, y1, x2, y2
    line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
    time_text.set_text(time_template % (i*h))
    return line, time_text


ani = animation.FuncAnimation(fig, animate, frames=len(t_values),interval=50, blit=True, init_func=init)

plt.show()


#grafico multiple - mismo angulo las dos pelotas

L = 1  
g = -9.8  
m = 3
h = 0.0001 

theta11_0 = np.pi/2
theta12_0 = np.pi/2
omega11_0 = 0  
omega12_0 = 0  

theta21_0 = np.pi/4
theta22_0 = np.pi/4
omega21_0 = 0  
omega22_0 = 0  

theta31_0 = np.pi/12
theta32_0 = np.pi/12
omega31_0 = 0  
omega32_0 = 0  

t_values = np.arange(0, 10, h)

theta11_values = np.zeros_like(t_values)
theta12_values = np.zeros_like(t_values)
omega11_values = np.zeros_like(t_values)
omega12_values = np.zeros_like(t_values)

theta21_values = np.zeros_like(t_values)
theta22_values = np.zeros_like(t_values)
omega21_values = np.zeros_like(t_values)
omega22_values = np.zeros_like(t_values)

theta31_values = np.zeros_like(t_values)
theta32_values = np.zeros_like(t_values)
omega31_values = np.zeros_like(t_values)
omega32_values = np.zeros_like(t_values)

theta11 = theta11_0
theta12 = theta12_0
omega11 = omega11_0
omega12 = omega12_0

theta21 = theta21_0
theta22 = theta22_0
omega21 = omega21_0
omega22 = omega22_0

theta31 = theta31_0
theta32 = theta32_0
omega31 = omega31_0
omega32 = omega32_0


for i, t in enumerate(t_values):
    theta11_values[i] = theta11
    theta12_values[i] = theta12
    omega11_values[i] = omega11
    omega12_values[i] = omega12

    theta11, theta12, omega11, omega12 = RK4(f, theta11, theta12, omega11, omega12, t, h)


for i, t in enumerate(t_values):
    theta21_values[i] = theta21
    theta22_values[i] = theta22
    omega21_values[i] = omega21
    omega22_values[i] = omega22

    theta21, theta22, omega21, omega22 = RK4(f, theta21, theta22, omega21, omega22, t, h)


for i, t in enumerate(t_values):
    theta31_values[i] = theta31
    theta32_values[i] = theta32
    omega31_values[i] = omega31
    omega32_values[i] = omega32

    theta31, theta32, omega31, omega32 = RK4(f, theta31, theta32, omega31, omega32, t, h)



fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

ax1.plot(t_values, theta11_values, label=f'Angulo aproximado: CI pi/2')
ax1.plot(t_values, theta21_values, label=f'Angulo aproximado: CI pi/4')
ax1.plot(t_values, theta31_values, label=f'Angulo aproximado: CI pi/12')
ax1.set_ylabel('Ángulo particula 1')
ax1.legend()


ax2.plot(t_values, theta12_values, label=f'Angulo aproximado: CI pi/2')
ax2.plot(t_values, theta22_values, label=f'Angulo aproximado: CI pi/4')
ax2.plot(t_values, theta32_values, label=f'Angulo aproximado: CI pi/12')
ax2.set_xlabel('Tiempo')
ax2.set_ylabel('Ángulo particula 2')
ax2.legend()
show()




#grafico multiple - distino angulo las dos pelotas

L = 1  
g = -9.8  
m = 3
h = 0.0001 

theta11_0 = np.pi/2
theta12_0 = np.pi/12
omega11_0 = 0  
omega12_0 = 0  

theta21_0 = np.pi/12
theta22_0 = np.pi/2
omega21_0 = 0  
omega22_0 = 0  

theta31_0 = np.pi/3
theta32_0 = np.pi/6
omega31_0 = 0  
omega32_0 = 0  

t_values = np.arange(0, 10, h)

theta11_values = np.zeros_like(t_values)
theta12_values = np.zeros_like(t_values)
omega11_values = np.zeros_like(t_values)
omega12_values = np.zeros_like(t_values)

theta21_values = np.zeros_like(t_values)
theta22_values = np.zeros_like(t_values)
omega21_values = np.zeros_like(t_values)
omega22_values = np.zeros_like(t_values)

theta31_values = np.zeros_like(t_values)
theta32_values = np.zeros_like(t_values)
omega31_values = np.zeros_like(t_values)
omega32_values = np.zeros_like(t_values)

theta11 = theta11_0
theta12 = theta12_0
omega11 = omega11_0
omega12 = omega12_0

theta21 = theta21_0
theta22 = theta22_0
omega21 = omega21_0
omega22 = omega22_0

theta31 = theta31_0
theta32 = theta32_0
omega31 = omega31_0
omega32 = omega32_0


for i, t in enumerate(t_values):
    theta11_values[i] = theta11
    theta12_values[i] = theta12
    omega11_values[i] = omega11
    omega12_values[i] = omega12

    theta11, theta12, omega11, omega12 = RK4(f, theta11, theta12, omega11, omega12, t, h)


for i, t in enumerate(t_values):
    theta21_values[i] = theta21
    theta22_values[i] = theta22
    omega21_values[i] = omega21
    omega22_values[i] = omega22

    theta21, theta22, omega21, omega22 = RK4(f, theta21, theta22, omega21, omega22, t, h)


for i, t in enumerate(t_values):
    theta31_values[i] = theta31
    theta32_values[i] = theta32
    omega31_values[i] = omega31
    omega32_values[i] = omega32

    theta31, theta32, omega31, omega32 = RK4(f, theta31, theta32, omega31, omega32, t, h)



fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

ax1.plot(t_values, theta11_values, label=f'Angulo aproximado: CI pi/2')
ax1.plot(t_values, theta21_values, label=f'Angulo aproximado: CI pi/12')
ax1.plot(t_values, theta31_values, label=f'Angulo aproximado: CI pi/3')
ax1.set_ylabel('Ángulo particula 1')
ax1.legend()


ax2.plot(t_values, theta12_values, label=f'Angulo aproximado: CI pi/12')
ax2.plot(t_values, theta22_values, label=f'Angulo aproximado: CI pi/2')
ax2.plot(t_values, theta32_values, label=f'Angulo aproximado: CI pi/6')
ax2.set_xlabel('Tiempo')
ax2.set_ylabel('Ángulo particula 2')
ax2.legend()
show()


L = 1  
g = -9.8  
m = 3
h = 0.0001 

theta11_0 = np.pi/2
theta12_0 = np.pi/12
omega11_0 = 0  
omega12_0 = 0  

theta21_0 = np.pi/12
theta22_0 = np.pi/2
omega21_0 = 0  
omega22_0 = 0  


t_values = np.arange(0, 10, h)

theta11_values = np.zeros_like(t_values)
theta12_values = np.zeros_like(t_values)
omega11_values = np.zeros_like(t_values)
omega12_values = np.zeros_like(t_values)

theta21_values = np.zeros_like(t_values)
theta22_values = np.zeros_like(t_values)
omega21_values = np.zeros_like(t_values)
omega22_values = np.zeros_like(t_values)


theta11 = theta11_0
theta12 = theta12_0
omega11 = omega11_0
omega12 = omega12_0

theta21 = theta21_0
theta22 = theta22_0
omega21 = omega21_0
omega22 = omega22_0



for i, t in enumerate(t_values):
    theta11_values[i] = theta11
    theta12_values[i] = theta12
    omega11_values[i] = omega11
    omega12_values[i] = omega12

    theta11, theta12, omega11, omega12 = RK4(f, theta11, theta12, omega11, omega12, t, h)


for i, t in enumerate(t_values):
    theta21_values[i] = theta21
    theta22_values[i] = theta22
    omega21_values[i] = omega21
    omega22_values[i] = omega22

    theta21, theta22, omega21, omega22 = RK4(f, theta21, theta22, omega21, omega22, t, h)


x11 = L*np.sin(theta11_values)
y11 = -L*np.cos(theta11_values)
x12 = x1 + L*np.sin(theta12_values)
y12 = y1 - L*np.cos(theta12_values)

x21 = L*np.sin(theta21_values)
y21 = -L*np.cos(theta21_values)
x22 = x1 + L*np.sin(theta22_values)
y22 = y1 - L*np.cos(theta22_values)


fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
ax1.set_aspect('equal', 'datalim')
ax1.plot(x11, y11, label='Particula 1')
ax1.plot(x12, y12, label='Particula 2')
ax1.legend()
ax1.set_xlabel('x')
ax1.set_ylabel('y')

ax2.set_aspect('equal', 'datalim')
ax2.plot(x21, y21, label='Particula 1')
ax2.plot(x22, y22, label='Particula 2')
ax2.legend()
ax2.set_xlabel('x')
ax2.set_ylabel('y')

show()



fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

ax1.plot(t_values, theta11_values, label=f'Angulo aproximado: CI pi/2')
ax1.plot(t_values, theta21_values, label=f'Angulo aproximado: CI pi/12')
ax1.set_ylabel('Ángulo particula 1')
ax1.legend()


ax2.plot(t_values, theta12_values, label=f'Angulo aproximado: CI pi/12')
ax2.plot(t_values, theta22_values, label=f'Angulo aproximado: CI pi/2')
ax2.set_xlabel('Tiempo')
ax2.set_ylabel('Ángulo particula 2')
ax2.legend()
show()


 


