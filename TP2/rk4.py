import numpy as np
from pylab import *

def int_rk4(f,t,theta_vec,h): #devuelve theta_vec_t+1 
    k1 = h*f(t,theta_vec)
    k2 = h*f(t+0.5*h,theta_vec+0.5*k1)
    k3 = h*f(t+0.5*h,theta_vec+0.5*k2)
    k4 = h*f(t+h,theta_vec+k3)
    return theta_vec + k1/6 + k2/3 + k3/3 + k4/6 

def f(t, theta_vec, length = 1):
    theta, w = theta_vec
    return np.array([w, (-9.8/length)*np.sin(theta)])


'''
theta0 = np.pi/4 #Angulo inicial
w0 = 0 #Velocidad inicial
theta_vec0 = np.array([theta0, w0]) #Condicion inicial
h = 0.01
length = 1.0  #Largo cuerda
tw = np.arange(0,10, h)
N = len(tw)

theta_vec= np.zeros((N, 2))
theta_vec[0] = theta_vec0
theta = np.zeros(N)
theta[0] = theta0
w = np.zeros(N)
w[0] = w0

for ti in range(N-1):
  t = ti*h
  theta_vec[ti+1] = int_rk4(f, t, theta_vec[ti], h)
  theta[ti+1] = theta_vec[ti+1][0]
  w[ti+1] = theta_vec[ti+1][1]


plot(tw, theta, label='Angulo aproximado')
xlabel('tiempo')
ylabel('angulo')
show()

#plot velocidad en funcion del tiempo (w)
plot(tw, w,  label='Velocidad aproximada')
plot(tw, w, 'g.', label='Puntos evaluados')
xlabel('$time$', fontsize=10)
ylabel('$velocity$', fontsize=10)
legend(fontsize=10)
show()


#plot trayectoria (r = (x, y))
plot(length * np.sin(theta), -length * np.cos(theta), label= 'Trayectoria')
plot(length * np.sin(theta), -length * np.cos(theta), 'g.', label= 'Puntos evaluados')
xlabel('$x$', fontsize=10)
ylabel('$y$', fontsize=10)
legend(fontsize=10)
show()


#plot energia en funcion del tiempo

m = 3 #masa inicial
T = 0.5*m*(length**2)*(w**2)
V = -m*(9.8)*length*np.cos(theta) + m*9.8*length
E = T + V
plot(tw, T,  label='Energia cinetica')
plot(tw, V,  label='Energia potencial')
plot(tw, E,  label='Energia total')
xlabel('$tiempo$', fontsize=10)
ylabel('$energia$', fontsize=10)
legend(fontsize=10)
show()


w0 = 0 #Velocidad inicial
theta0 = np.pi/4


#grafico de h
h_vec = np.arange(0.0001, 1, 0.005)
diff_vec = np.zeros(len(h_vec))
for i, h in enumerate(h_vec):
    t_h = np.arange(0, 10, h)
    n_h = len(t_h)
    theta_vec_h= np.zeros((n_h, 2))
    theta_vec0 = np.array([theta0, w0]) 
    theta_vec_h[0] = theta_vec0
    ang_approx_h = np.zeros(n_h)
    w_approx_h = np.zeros(n_h)
    ang_approx_h[0] = theta0
    w_approx_h[0] = w0
    for ti in range(n_h-1):
      t = ti*h
      theta_vec_h[ti+1] = int_rk4(f, t, theta_vec_h[ti], h)
      ang_approx_h[ti+1] = theta_vec_h[ti+1][0]
      w_approx_h[ti+1] = theta_vec_h[ti+1][1]
    print(f"ang_approx_h = {ang_approx_h}")
    print(f"h = {h} err = {max(abs(ang_approx_h)) }")
    diff_vec[i] = abs(max(abs(ang_approx_h - theta0)))
    
plot(h_vec, diff_vec,  label=f'Error maximo acumulado')
plot(h_vec, diff_vec, 'g.',  label=f'h evaluados')
xlabel('Paso temporal (h)', fontsize=10)
ylabel('$Error$', fontsize=10)
legend(fontsize=10)
show()

'''
#grafico multiple


h = 0.9
length = 1.0  #Largo cuerda

theta1 = np.pi/4 
theta2 = np.pi/2
theta3 = 2*np.pi/3
theta4 = np.pi/12



w1 = 0 
w2 = 0 
w3 = 0 
w4 = 0

theta_vec1 = np.array([theta1, w1])
theta_vec2 = np.array([theta2, w2])
theta_vec3 = np.array([theta3, w3]) 
theta_vec4 = np.array([theta4, w4]) 

tw = np.arange(0,10, h)
N = len(tw)

theta_vecc1= np.zeros((N, 2))
theta_vecc2= np.zeros((N, 2))
theta_vecc3= np.zeros((N, 2))
theta_vecc4= np.zeros((N, 2))


theta_vecc1[0] = theta_vec1
theta_vecc2[0] = theta_vec2
theta_vecc3[0] = theta_vec3
theta_vecc4[0] = theta_vec4


thetaa1 = np.zeros(N)
thetaa2 = np.zeros(N)
thetaa3 = np.zeros(N)
thetaa4 = np.zeros(N)

thetaa1[0] = theta1
thetaa2[0] = theta2
thetaa3[0] = theta3
thetaa4[0] = theta4

ww1 = np.zeros(N)
ww2 = np.zeros(N)
ww3 = np.zeros(N)
ww4 = np.zeros(N)

ww1[0] = w1
ww2[0] = w2
ww3[0] = w3
ww4[0] = w4


for ti in range(N-1):
  t = ti*h
  theta_vecc1[ti+1] = int_rk4(f, t, theta_vecc1[ti], h)
  thetaa1[ti+1] = theta_vecc1[ti+1][0]
  ww1[ti+1] = theta_vecc1[ti+1][1]

  theta_vecc2[ti+1] = int_rk4(f, t, theta_vecc2[ti], h)
  thetaa2[ti+1] = theta_vecc2[ti+1][0]
  ww2[ti+1] = theta_vecc2[ti+1][1]

  theta_vecc3[ti+1] = int_rk4(f, t, theta_vecc3[ti], h)
  thetaa3[ti+1] = theta_vecc3[ti+1][0]
  ww3[ti+1] = theta_vecc3[ti+1][1]

  theta_vecc4[ti+1] = int_rk4(f, t, theta_vecc4[ti], h)
  thetaa4[ti+1] = theta_vecc4[ti+1][0]
  ww4[ti+1] = theta_vecc4[ti+1][1]


plot(tw, thetaa1, label=f'Angulo aproximado: CI pi/4')
plot(tw, thetaa2, label=f'Angulo aproximado: CI pi/2')
plot(tw, thetaa3, label=f'Angulo aproximado: CI 2pi/3')
plot(tw, thetaa4, label=f'Angulo aproximado: CI pi/12')
xlabel('$Tiempo$', fontsize=10)
ylabel('$Angulo$', fontsize=10)
legend(fontsize=10)
show()


plot(tw, ww1,  label=f'Velocidad aproximada: CI pi/4')
plot(tw, ww2,  label=f'Velocidad aproximada: CI pi/2')
plot(tw, ww3,  label=f'Velocidad aproximada: CI 2pi/3')
plot(tw, ww4,  label=f'Velocidad aproximada: CI pi/12')

xlabel('$Tiempo$', fontsize=10)
ylabel('$Velocidad$', fontsize=10)
legend(fontsize=10)
show()






