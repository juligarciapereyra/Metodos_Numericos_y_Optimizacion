import numpy as np
from pylab import *

def int_rk4_1(f,t,theta_vec1, theta_vec2, h): #devuelve theta_vec1_t+1 
    k1 = h*f(t,theta_vec1, theta_vec2)
    k2 = h*f(t+0.5*h,theta_vec1+0.5*k1, theta_vec2)
    k3 = h*f(t+0.5*h,theta_vec1+0.5*k2, theta_vec2)
    k4 = h*f(t+h,theta_vec1+k3, theta_vec2)
    return theta_vec1 + k1/6 + k2/3 + k3/3 + k4/6 

def int_rk4_2(f,t,theta_vec1, theta_vec2, h): #devuelve theta_vec_t+1 
    k1 = h*f(t,theta_vec1, theta_vec2)
    k2 = h*f(t+0.5*h,theta_vec1, theta_vec2+0.5*k1)
    k3 = h*f(t+0.5*h,theta_vec1, theta_vec2+0.5*k2)
    k4 = h*f(t+h,theta_vec1, theta_vec2+k3)
    return theta_vec2 + k1/6 + k2/3 + k3/3 + k4/6 

def f1(t,theta_vec1, theta_vec2, length = 1):
    ang1_t, w1_t = theta_vec1
    ang2_t, w2_t = theta_vec2
    delta = ang1_t - ang2_t
    return np.array([w1_t,((9.8/length)*((2*np.sin(ang1_t) - np.cos(delta)*np.sin(ang2_t)) - np.sin(delta)*(w2_t**2 + np.cos(delta)*w1_t**2)))/(2-np.cos(delta)**2)])
    
def f2(t, theta_vec1, theta_vec2, length = 1):
    ang1_t, w1_t = theta_vec1
    ang2_t, w2_t = theta_vec2
    delta = ang1_t - ang2_t
    return np.array([w2_t, ((9.8/length)*((np.sin(ang2_t) - 2*np.cos(delta)*np.sin(ang1_t)) + np.sin(delta)*(w1_t**2 + np.cos(delta)*w2_t**2)))/(2-np.cos(delta)**2)])

h = 0.001
length = 1.0  #largo cuerda

#Angulos iniciales
theta0_1 = np.pi/2 
theta0_2 = np.pi/2  

#Velocidad inicial
w0_1= 0 
w0_2= 0

theta_vec0_1 = np.array([theta0_1, w0_1]) #Condicion inicial 1
theta_vec0_2 = np.array([theta0_2, w0_2]) #Condicion inicial 2


tw = np.arange(0, 10, h)
N = len(tw)

theta_vec1 = np.zeros((N, 2))
theta_vec2 = np.zeros((N, 2))
theta_vec1[0] = theta_vec0_1
theta_vec2[0] = theta_vec0_2
theta1 = np.zeros(N)
theta2 = np.zeros(N)
theta1[0] = theta0_1
theta2[0] = theta0_2
w1 = np.zeros(N)
w2 = np.zeros(N)
w1[0] = w0_1
w2[0] = w0_2

for ti in range(N-1):
  t = ti*h
  theta_vec1[ti+1] = int_rk4_1(f1, t, theta_vec1[ti], theta_vec2[ti], h)
  theta1[ti+1] = theta_vec1[ti+1][0]
  w1[ti+1] = theta_vec1[ti+1][1]
  
  theta_vec2[ti+1] = int_rk4_2(f2, t, theta_vec1[ti], theta_vec2[ti], h)
  theta2[ti+1] = theta_vec2[ti+1][0]
  w2[ti+1] = theta_vec2[ti+1][1]


plot(tw, theta1, label='Angulo aproximado 1')
xlabel('tiempo')
ylabel('angulo')
show()

plot(tw, theta2, label='Angulo aproximado 2')
xlabel('tiempo')
ylabel('angulo')
show()


#plot trayectoria (r = (x, y))
plot(length * np.sin(theta1), -length * np.cos(theta1), label= 'Trayectoria 1')
plot(length * np.sin(theta1), -length * np.cos(theta1), 'g.', label= 'Puntos evaluados 1')
xlabel('$x$', fontsize=10)
ylabel('$y$', fontsize=10)
legend(fontsize=10)
show()

plot(length * np.sin(theta2), -length * np.cos(theta2), label= 'Trayectoria 2')
plot(length * np.sin(theta2), -length * np.cos(theta1), 'g.', label= 'Puntos evaluados 2')
xlabel('$x$', fontsize=10)
ylabel('$y$', fontsize=10)
legend(fontsize=10)
show()


#plot energia en funcion del tiempo

m = 3 #masa inicial
T = 0.5*m*(length**2)*(w1**2)
V = -m*(9.8)*length*np.cos(theta1) + m*9.8*length
E = T + V
plot(tw, T,  label='Energia cinetica')
plot(tw, V,  label='Energia potencial')
plot(tw, E,  label='Energia total')
xlabel('$t$', fontsize=10)
ylabel('$energia$', fontsize=10)
legend(fontsize=10)
show()


T = 0.5*m*(length**2)*(w2**2)
V = -m*(9.8)*length*np.cos(theta2) + m*9.8*length
E = T + V
plot(tw, T,  label='Energia cinetica')
plot(tw, V,  label='Energia potencial')
plot(tw, E,  label='Energia total')
xlabel('$t$', fontsize=10)
ylabel('$energia$', fontsize=10)
legend(fontsize=10)
show()


