from pylab import *
import numpy as np
from scipy.interpolate import lagrange

h = 0.01
x_values = np.arange(-1, 1, h)
f = lambda x : -0.4*np.tanh(50*x) + 0.6

num_points = 20

x = np.linspace(-1, 1, num_points)
poli_inter = lagrange(x, f(x))


plot(x, f(x), 'bo')
plot(x_values, f(x_values), label='$f$')
plot(x_values, poli_inter(x_values), label='Interpol. Lagrange')
title('Interpolacion con Polinomio de Lagrange')
xlabel('x ')
ylabel('y ')
legend()
show()

#ERROR CON PUNTOS EQUIDISTANTES
semilogy(x_values,np.abs(f(x_values)-poli_inter(x_values)), label='Puntos Eq.')
xlabel('x value')
ylabel('Error Puntos Eq.')
show()
legend()

# PUNTOS NO EQUIDISTANTES
x_min, x_max = -1, 1
x_points = [x_min, x_max]
y_points = [f(x_min), f(x_max)]

while len(x_points) < num_points:
    dy = np.diff(y_points)
    i = np.argmax(np.abs(dy))
    x_new = (x_points[i] + x_points[i+1]) / 2
    y_new = f(x_new)
    x_points.insert(i+1, x_new)
    y_points.insert(i+1, y_new)

poli_inter_neq = lagrange(x_points, y_points)
plot(x_values, f(x_values), label='$f$')
semilogy(x_values, poli_inter_neq(x_values), label='Lagrange NO Eq.')
plot(x_points, y_points, 'ro')
legend()
show()

#ERROR CON PUNTOS SELECCIONADOS POR VARIACION NO EQUIDISTANTES
err_max_neq = np.max(np.abs(f(x)-poli_inter(x)))
plot(x,np.abs(f(x)-poli_inter_neq(x)))
err_neq = np.abs(f(x) - poli_inter_neq(x))
semilogy(x, err_neq,label='Puntos NO equidistantes')
xlabel('x value')
ylabel('Error No Eq. ')
show()
legend()
print(err_max_neq)


#PUNTOS NO EQUIDISTANTES RANDOM
x_points_rand = np.sort(np.random.uniform(-1, 1, size=num_points))
y_points_rand = f(x_points_rand)
lagrange_neq_rand = lagrange(x_points_rand, y_points_rand)
plot(x_values, f(x_values), label='$f$')
plot(x_points_rand, y_points_rand, 'o', label='Puntos Random')
semilogy(x, lagrange_neq_rand(x), label='Lagrange no Eq. Random')
legend()
show()

#ERROR CON PUNTOS RANDOM
err_neq_rand = np.abs(f(x) - lagrange_neq_rand(x))
semilogy(x, err_neq_rand, label = 'Error NO Eq. random')
xlabel('x value')
ylabel('Error (log scale)')
print(err_max_neq)
legend()
show()

