import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import griddata
from matplotlib.colors import LogNorm


def z(x, y):
    return 0.75*np.exp((-(((9*x)-2)**2)/4)-((((9*y)-2)**2)/4)) + \
           0.75*np.exp((-(((9*x)+1)**2)/49)-((((9*y)+2)**2)/10)) + \
           0.5*np.exp((-(((9*x)-7)**2)/4)-((((9*y)-3)**2)/4)) - \
           0.2*np.exp((-((9*x-7)**2)/4)-(((9*y-3)**2)/4))

x_values = np.linspace(-1, 1, 100)
y_values = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x_values, y_values)
Z = z(X, Y)

#INTERPOLACION CON N PUNTOS NO EQUIDISTANTES RANDOM

n_points = 100
x_new_rand = np.sort(np.random.uniform(-1, 1, n_points))
y_new_rand = np.sort(np.random.uniform(-1, 1, n_points))
X_new_rand, Y_new_rand = np.meshgrid(x_new_rand, y_new_rand)


'''
#CON RECTBIVARIATESPLINE
splines_rand = RectBivariateSpline(x_new_rand, y_new_rand, z(X_new_rand, Y_new_rand), kx=3, ky=3)
Z_spline = splines_rand.ev(X, Y)

fig0 = plt.figure()
ax0= fig0.add_subplot(111, projection='3d')

ax0.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
ax0.set_xlabel('X')
ax0.set_ylabel('Y')
ax0.set_zlabel('Z')
ax0.set_title('Funcion Original')
plt.show()

fig1 = plt.figure()
ax1= fig1.add_subplot(111, projection='3d')
ax1.plot_surface(X, Y, Z_spline, cmap='magma', alpha=0.5)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Interpolacion')
plt.show()

fig = plt.figure()
ax= fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='binary', alpha=0.5)
ax.plot_surface(X, Y, Z_spline, cmap='magma', alpha=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Funcion original vs Interpolacion')
plt.show()

#ERROR
fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')
error = np.abs(Z - Z_spline)
error_max = np.max(error)
print(error_max)
ax2.plot_surface(X, Y, error, cmap='plasma', alpha=0.5)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Error Absoluto Spline')
ax2.set_title('Error')
plt.show()

fig = plt.figure()
ax3 = fig.add_subplot(111, projection='3d')
ax3.scatter(X_new_rand, Y_new_rand, z(X_new_rand, Y_new_rand), color='red', s=1) #para graficar los puntos random seleccionados
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_title('Puntos no equidistantes - random')
plt.show()

'''



#CON GRIDDATA LINEAL
points = np.column_stack((X_new_rand.ravel(), Y_new_rand.ravel()))
values = z(X_new_rand, Y_new_rand).ravel()
Z_lineal = griddata(points, values, (X, Y), method='linear')

fig0 = plt.figure()
ax0= fig0.add_subplot(111, projection='3d')

ax0.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
ax0.set_xlabel('X')
ax0.set_ylabel('Y')
ax0.set_zlabel('Z')
ax0.set_title('Funcion Original')
plt.show()

fig1 = plt.figure()
ax1= fig1.add_subplot(111, projection='3d')
ax1.plot_surface(X, Y, Z_lineal, cmap='magma', alpha=0.5)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Interpolacion')
plt.show()

fig = plt.figure()
ax4 = fig.add_subplot(111, projection='3d', label= 'f')
ax4.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
ax4.plot_surface(X, Y, Z_lineal, cmap='magma', alpha=0.5)
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')
ax4.set_title('Funcion original vs Interpolacion')
plt.show()

#ERROR
fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')
error = np.abs(Z - Z_lineal)
error_max = np.max(error)
print(error_max)
ax2.plot_surface(X, Y, error, cmap='plasma', alpha=0.5, norm=LogNorm())
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Error Absoluto Lineal')
ax2.set_title('Error')
plt.show()


fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.scatter(X_new_rand, Y_new_rand, z(X_new_rand, Y_new_rand), color='red', s=1) #para graficar los puntos random seleccionados
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_title('Puntos no equidistantes - random')
plt.show()

