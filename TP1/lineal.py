from pylab import *
import numpy as np
from scipy.interpolate import interp1d

h = 0.01 #puede variar
x_values = np.arange(-1, 1, h)
f = lambda x : -0.4*np.tanh(50*x) + 0.6

#PUNTOS EQUIDISTANTES 

num_points = 100 #puede variar
x = np.linspace(-1, 1, num_points) #puntos para la interpolacion
lineal = interp1d(x, f(x), kind='linear')

plot(x_values, lineal(x_values), label='Lineal a trozos')
plot(x_values, f(x_values), label='$f$')
plot(x, f(x), 'o')

title('Interpolacion Lineal a Trozos')
xlabel('x ')
ylabel('y ')
legend()
show()

err_max_eq = np.max(np.abs(f(x_values)-lineal(x_values)))
err_eq = np.abs(f(x_values) - lineal(x_values))
plot(x_values,err_eq)
semilogy(x_values, err_eq, label='Puntos Eq.')
xlabel('x value')
ylabel('Error (log scale)')
legend()
show()

# PUNTOS NO EQUIDISTANTES
x_min, x_max = -1, 1

#SEGUN EL CAMBIO EN Y
x_points = [x_min, x_max]
y_points = [f(x_min), f(x_max)]

while len(x_points) < num_points:
    dy = np.diff(y_points)
    i = np.argmax(np.abs(dy))
    x_new = (x_points[i] + x_points[i+1]) / 2
    y_new = f(x_new)
    x_points.insert(i+1, x_new)
    y_points.insert(i+1, y_new)
    
f_neq = interp1d(x_points, y_points, kind='linear')


plot(x_values, f(x_values), label='$f$')
plot(x_points, y_points, 'o', label='Puntos no Eq.')
plot(x_points, f_neq(x_points), label='Interpol. no Eq.')
title('Interpolacion Lineal a Trozos')
xlabel('x ')
ylabel('y ')
legend()
show()

#PUNTOS RANDOM
x_points_rand = np.sort(np.random.uniform(-1, 1, size=num_points))
y_points_rand = f(x_points_rand)
f_neq_rand = interp1d(x_points_rand, y_points_rand, kind = 'linear')
plot(x_values, f(x_values), label='$f$')
plot(x_points_rand, y_points_rand, 'o', label='Puntos Random')
plot(x_points_rand, f_neq_rand(x_points_rand), label='Interpol. no Eq. Rand')
title('Interpolacion Lineal a Trozos')
xlabel('x ')
ylabel('y ')
legend()
show()

x_min, x_max = x_points_rand[0], x_points_rand[-1]
x = np.clip(x, x_min, x_max) #esto es para que no se me pase el rango de x del [-1, 1] y pueda graficar todo correctamente


#ERROR CON PUNTOS EQUIDISTANTES
err_max_eq = np.max(np.abs(f(x_values)-lineal(x_values)))
err_eq = np.abs(f(x_values) - lineal(x_values))
plot(x_values,err_eq)
semilogy(x_values, err_eq, label='Puntos Eq.') #para que este en base logaritmica
print(err_max_eq)

#ERROR CON PUNTOS NO EQUIDISTANTES
err_max_neq = np.max(np.abs(f(x)-f_neq(x))) #SEGUN VARIACION
err_neq = np.abs(f(x) - f_neq(x))
err_neq_rand = np.abs(f(x) -  f_neq_rand(x)) #ALEATORIOS
plot(x,np.abs(f(x)-f_neq(x)))
semilogy(x, err_neq,label='Puntos No Eq.')
plot(x,np.abs(f(x)-f_neq_rand(x)))
semilogy(x, err_neq_rand,label='Puntos No Eq. Rand')
xlabel('x value')
ylabel('Error (log scale)')
print(err_max_neq)
legend()
show()