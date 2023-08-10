import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib.colors import LogNorm

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def z(x, y):
    return 0.75*np.exp((-(((9*x)-2)**2)/4)-((((9*y)-2)**2)/4)) + \
           0.75*np.exp((-(((9*x)+1)**2)/49)-((((9*y)+2)**2)/10)) + \
           0.5*np.exp((-(((9*x)-7)**2)/4)-((((9*y)-3)**2)/4)) - \
           0.2*np.exp((-((9*x-7)**2)/4)-(((9*y-3)**2)/4))

x_values = np.linspace(-1, 1, 100)
y_values = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x_values, y_values)

Z = z(X, Y)

x_interp = np.linspace(-1, 1, 100)
y_interp = np.linspace(-1, 1, 100)
X_interp, Y_interp = np.meshgrid(x_interp, y_interp)

points = np.column_stack((X_interp.ravel(), Y_interp.ravel()))
values = z(X_interp, Y_interp).ravel()

Z_interp = griddata(points, values, (X, Y), method='linear')

fig = plt.figure()
ax= fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
ax.plot_surface(X, Y, Z_interp, cmap='magma', alpha=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Funcion Original vs Interpolacion')
plt.show()

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
ax1.plot_surface(X, Y, Z_interp, cmap='magma', alpha=0.5)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Interpolacion')
plt.show()


fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')

error = np.abs(Z - Z_interp)
error_max = np.max(error)
print(error_max)
ax2.plot_surface(X, Y, error, cmap='plasma', alpha=0.5, norm=LogNorm())
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Error Absoluto')
plt.show()