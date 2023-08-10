import numpy as np
from pylab import *

import numpy as np
from pylab import *

def euler_method(wt, ang_i, t, f, h):
    w = wt + h*f(ang_i)
    return w
 
def f(ang_t, length = 1):
    return (-9.8/length)*np.sin(ang_t)


theta0 = np.pi/2# condicion inicial
h = 0.01
length = 1.0  #largo cuerda
tw = np.arange(0, 10, h)
N = len(tw)
w_approx = np.zeros(N) # Velocidad (w = d theta / dt)
ang_approx = np.zeros(N) # Angulos (theta)
ang_approx[0] = theta0
w_approx[0] = 0

# aceleracion (d theta / d^2t = dw/dt = f(theta))

for ti in range(N-1):
  t = ti*h
  w_approx[ti+1] = euler_method(w_approx[ti], ang_approx[ti], t, f, h)
  ang_approx[ti+1] = ang_approx[ti] +h*w_approx[ti]
  

#plot angulo en funcion del tiempo (theta)
plot(tw, ang_approx,  label='Angulo aproximado con Euler')
xlabel('$Tiempo$', fontsize=10)
ylabel('$Angulo$', fontsize=10)
legend(fontsize=10)
show()

print(f" max : {ang_approx}")


#plot velocidad en funcion del tiempo (w)
plot(tw, w_approx,  label='Velocidad aproximada con Euler')
xlabel('$Tiempo$', fontsize=10)
ylabel('$Velocidad$', fontsize=10)
legend(fontsize=10)
show()

#plot trayectoria (r = (x, y))
plot(length * np.sin(ang_approx), -length * np.cos(ang_approx), label= 'Trayectoria aproximada con Euler')
print(f"pos {min(np.sin(ang_approx))},{(max(-length * np.cos(ang_approx)))}")
#plot(length * np.sin(ang_approx), -length * np.cos(ang_approx), 'g.', label= 'Puntos evaluados')
ylabel('$Y$', fontsize=10)
xlabel('$X$', fontsize=10)
legend(fontsize=10)
show()


#plot energia en funcion del tiempo

m = 3 #masa inicial
T = 0.5*m*(length**2)*(w_approx**2)
V = -m*(9.8)*length*np.cos(ang_approx) + m*9.8*length
E = T + V
plot(tw, T,  label='Energia cinetica')
plot(tw, V,  label='Energia potencial')
plot(tw, E,  label='Energia total')
xlabel('$Tiempo$', fontsize=10)
ylabel('$Energia$', fontsize=10)
legend(fontsize=10)
show()


w_approx1 = np.zeros(N)
ang_approx1 = np.zeros(N) 
ang_approx1[0] = np.pi/2
w_approx1[0] = 0


w_approx2 = np.zeros(N)
ang_approx2 = np.zeros(N) 
ang_approx2[0] = 2*np.pi/3
w_approx2[0] = 0


w_approx3 = np.zeros(N)
ang_approx3 = np.zeros(N) 
ang_approx3[0] = np.pi/12
w_approx3[0] = 0


for ti in range(N-1):
  t = ti*h
  w_approx1[ti+1] = euler_method(w_approx1[ti], ang_approx1[ti], t, f, h)
  ang_approx1[ti+1] =ang_approx1[ti] +h*w_approx1[ti]
  w_approx2[ti+1] = euler_method(w_approx2[ti], ang_approx2[ti], t, f, h)
  ang_approx2[ti+1] =ang_approx2[ti] +h*w_approx2[ti]
  w_approx3[ti+1] = euler_method(w_approx3[ti], ang_approx3[ti], t, f, h)
  ang_approx3[ti+1] =ang_approx3[ti] +h*w_approx3[ti]
  



#plot angulo en funcion del tiempo (theta)
plot(tw, ang_approx,  label='Angulo C.I pi/4')
plot(tw, ang_approx1,  label='Angulo C.I pi/2')
plot(tw, ang_approx2,  label='Angulo C.I 2pi/3')
plot(tw, ang_approx3,  label='Angulo C.I pi/12')
xlabel('$Tiempo$', fontsize=10)
ylabel('$Angulo$', fontsize=10)
legend(fontsize=10)
show()

#plot velocidad en funcion del tiempo (w)
plot(tw, w_approx,  label='Velocidad C.I pi/4')
plot(tw, w_approx1,  label='Velocidad C.I pi/2')
plot(tw, w_approx2,  label='Velocidad C.I 2pi/3')
plot(tw, w_approx3,  label='Velocidad C.I pi/12')
xlabel('$Tiempo$', fontsize=10)
ylabel('$Velocidad$', fontsize=10)
legend(fontsize=10)
show()

#plot trayectoria (r = (x, y))
plot(length * np.sin(ang_approx), -length * np.cos(ang_approx), label= 'Trayectoria  C.I pi/4')
plot(length * np.sin(ang_approx1), -length * np.cos(ang_approx1), label= 'Trayectoria C.I pi/2')
plot(length * np.sin(ang_approx2), -length * np.cos(ang_approx2), label= 'Trayectoria C.I 2pi/3')
plot(length * np.sin(ang_approx3), -length * np.cos(ang_approx3), label= 'Trayectoria C.I pi/12')

ylabel('$Y$', fontsize=10)
xlabel('$X$', fontsize=10)
legend(fontsize=10)
show()




theta0 = np.pi/4 # condicion inicial

h_vec = np.arange(0.0001, 1, 0.005)
diff_vec = np.zeros(len(h_vec))
for i, h in enumerate(h_vec):
    t_h = np.arange(0, 10, h)
    n_h = len(t_h)
    ang_approx_h = np.zeros(n_h)
    w_approx_h = np.zeros(n_h)
    ang_approx_h[0] = theta0
    w_approx_h[0] = 0 
    for ti in range(n_h-1):
        t = ti*h
        w_approx_h[ti+1] = euler_method(w_approx_h[ti], ang_approx_h[ti], t, f, h)
        ang_approx_h[ti+1] = ang_approx_h[ti] +h*w_approx_h[ti]
    diff_vec[i] = abs(max(abs(ang_approx_h - theta0)))

semilogy(h_vec, diff_vec,  label=f'Error maximo acumulado')
plot(h_vec, diff_vec, 'g.',  label=f'h evaluados')


xlabel('Paso temporal (h)', fontsize=10)
ylabel('$Error$', fontsize=10)
legend(fontsize=10)
show()





'''


# ANIMACION DEL PENDULO

theta = theta0 #angulo
w = 0.0 #velocidad

# r = (x, y) posiciones del pendulo
x = length * np.sin(theta)
y = -length * np.cos(theta)

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-length-0.2, length+0.2)
ax.set_ylim(-length-0.2, length+0.2)

for t in enumerate(tw):
  
    w = euler_method(w, theta, t, f, h)
    theta = theta + h*w
    
    # posicion actual
    x = length * np.sin(theta)
    y = -length * np.cos(theta)
    
    # animacion
    ax.cla()
    ax.plot([-0.1, 0.1], [0, 0], color='black', linewidth=4)
    ax.plot([0, x], [0, y], 'black', linewidth=2)
    ax.plot(x, y, 'ro', markersize=20)
    ax.set_xlim(-length-0.1, length+0.1)
    ax.set_ylim(-length-0.1, length+0.1)
    ax.set_aspect('equal')
    pause(0.001)

show()


theta_real = np.zeros(N)
theta_real[0] = theta0
w_real = np.zeros(N)
w_real[0] = 0

for ti in range(N-1):
  t = ti*h
  theta_real[ti+1] = theta0*(np.cos(np.sqrt(9.8/length)*t) + 1j*np.sin(np.sqrt(9.8/length)*t))
  w_real[ti+1] = theta0*np.sqrt(9.8/length)*1j*(np.cos(np.sqrt(9.8/length)*t) + 1j*np.sin(np.sqrt(9.8/length)*t))

#plot angulo en funcion del tiempo (theta)
plot(tw, theta_real,  label='Angulo REAL')
plot(tw, ang_approx,  label='Angulo aproximado')
#plot(tw, theta_real, 'g.', label='Puntos evaluados')
xlabel('$time$', fontsize=10)
ylabel('$angle$', fontsize=10)
legend(fontsize=10)
show()

plot(tw, theta_real, 'g.', label='Puntos evaluados')
xlabel('$time$', fontsize=10)
ylabel('$angle$', fontsize=10)
legend(fontsize=10)
show()


#plot velocidad en funcion del tiempo (w)
plot(tw, w_real,  label='Velocidad REAL')
plot(tw, w_approx,  label='Velocidad aproximada')
xlabel('$Tiempo$', fontsize=10)
ylabel('$Velocidad$', fontsize=10)
legend(fontsize=10)
show()
  
#plot trayectoria (r = (x, y))

plot(length * np.sin(ang_approx), -length * np.cos(ang_approx), label= 'Trayectoria aprox')
plot(length * np.sin(theta_real), -length * np.cos(theta_real), 'g.', label= 'Trayectoria REAL')
xlabel('$x$', fontsize=10)
ylabel('$y$', fontsize=10)
legend(fontsize=10)
show()

T = 0.5*m*(length**2)*(w_real**2)
V = -m*(9.8)*length*np.cos(theta_real) + m*9.8*length
E = T + V
plot(tw, T,  label='Energia cinetica REAL')
plot(tw, V,  label='Energia potencial REAL')
plot(tw, E,  label='Energia total REAL')
xlabel('$Tiempo$', fontsize=10)
ylabel('$Energia$', fontsize=10)
legend(fontsize=10)
show()

'''
