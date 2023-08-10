from pylab import *
import numpy as np
from scipy.interpolate import CubicSpline

from pylab import *
import numpy as np
from scipy.interpolate import CubicSpline

h = 0.05
x_values = np.arange(-1, 1, h)
f = lambda x : -0.4*np.tanh(50*x) + 0.6

x = np.linspace(-1, 1, 50)
spline = CubicSpline(x_values, f(x_values))

plot(x, spline(x), label='Spline Cubico')
plot(x_values, f(x_values), label='$f$')

title('InterpolaciÃ³n con Spline Cubico')
legend()
show()


# PUNTOS NO EQUIDISTANTES

#SEGUN EL CAMBIO EN Y
x_min, x_max = -1, 1
num_points = 50
tol = 0.1

x_points = [x_min, x_max]
y_points = [f(x_min), f(x_max)]

while len(x_points) < num_points:
    dy = np.diff(y_points)
    i = np.argmax(np.abs(dy))
    x_new = (x_points[i] + x_points[i+1]) / 2
    y_new = f(x_new)
    x_points.insert(i+1, x_new)
    y_points.insert(i+1, y_new)

spline_neq = CubicSpline(x_points, y_points)

x = np.linspace(x_min, x_max, 1000)
plot(x_points, y_points, 'o', label='Puntos no equidistantes')
plot(x, spline_neq(x), label='Interpol. con puntos n. e')
legend()
show()

#PUNTOS RANDOM
x_points_rand = np.sort(np.random.uniform(-1, 1, size=num_points))
y_points_rand = f(x_points_rand)
spline_neq_rand = CubicSpline(x_points_rand, y_points_rand)
plot(x_values, f(x_values), label='$f$')
plot(x_points_rand, y_points_rand, 'o', label='Puntos Random')
plot(x_points_rand, spline_neq_rand(x_points_rand), label='Spline. no Eq. Random')
legend()
show()


err_max_eq = np.max(np.abs(f(x)-spline(x)))
err_eq = np.abs(f(x) - spline(x))
plot(x,np.abs(f(x)-spline(x)))
semilogy(x, err_eq, label='Puntos Eq') #para que este en base logaritmica

err_max_neq = np.max(np.abs(f(x)-spline_neq(x)))
err_neq = np.abs(f(x) - spline_neq(x))
plot(x,np.abs(f(x)-spline_neq(x)))
semilogy(x, err_neq,label='Puntos NO Eq')

err_neq_rand = np.abs(f(x) - spline_neq_rand(x))
plot(x, err_neq_rand, label = 'NO Eq random')
xlabel('x value')
ylabel('Error (log scale)')
print(err_max_neq)
legend()
show()
