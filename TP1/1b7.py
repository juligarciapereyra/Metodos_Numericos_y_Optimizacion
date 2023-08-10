import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.stats import beta

# Define the function to interpolate
def z(x, y):
    return (0.75*np.exp(-(((9*x)-2)**2)/4 - (((9*y)-2)**2)/4) 
            + 0.75*np.exp(-(((9*x)+1)**2)/49 - (((9*y)+2)**2)/10) 
            + 0.5*np.exp(-(((9*x)-7)**2)/4 - (((9*y)-3)**2)/4)
            - 0.2*np.exp(-((9*x-7)**2)/4 - ((9*y-3)**2)/4))

x = np.linspace(-1, 1, 50)
y = np.linspace(-1, 1, 50)
X1, Y1 = np.meshgrid(x, y)

# Evaluate the function on the grid
Z1 = z(X1, Y1)

# Define the number of points and the interpolation domain
n_points = 50
x_min, x_max = -1, 1
y_min, y_max = -1, 1

# Define the points to interpolate
n_interp_points = 50

# Define a custom distribution for generating non-equidistant points
dist = beta(2, 5)

# Generate non-equidistant x points from the distribution
x_points = (x_max - x_min) * dist.rvs(n_interp_points) + x_min

# Generate non-equidistant y points from the distribution
y_points = (y_max - y_min) * dist.rvs(n_interp_points) + y_min

# Create a meshgrid of the non-equidistant points
X_interp, Y_interp = np.meshgrid(x_points, y_points, indexing='ij')

# Create a 2D array of the non-equidistant points
points = np.array([X_interp.ravel(), Y_interp.ravel()]).T

# Create the grid of points
X, Y = np.meshgrid(x_points, y_points, indexing='ij')

# Evaluate the function on the grid
Z = z(X, Y)

# Interpolate the function on the new points using griddata
Z_interp_grid = griddata(points, z(points[:, 0], points[:, 1]), (X_interp, Y_interp), method='cubic')

# Create a figure and axis objects
fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': '3d'})

# Plot the original function
ax.contour3D(X1, Y1, Z1, 50, cmap='binary')

# Plot the interpolated function
ax.contour3D(X_interp, Y_interp, Z_interp_grid, 50, cmap='magma')

# Set the labels for the x, y, and z axes
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Set the title for the plot
ax.set_title('Original Function vs. Interpolated Function')

# Create a new figure and axis objects for the non-equidistant points
fig2, ax2 = plt.subplots(figsize=(8,6))

ax2.scatter(points[:, 0], points[:, 1], s=5)
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Non-equidistant Points')

# Create a new figure and axis objects for the non-equidistant points
fig3, ax3 = plt.subplots(figsize=(8,6), subplot_kw={'projection': '3d'})

# Evaluate the function on the non-equidistant points
Z_points = z(points[:, 0], points[:, 1])

# Plot the non-equidistant points on top of the original function
ax3.scatter(points[:, 0], points[:, 1], Z_points, color='r')

# Set the labels for the x, y, and z axes
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z')

# Set the title for the plot
ax3.set_title('Original Function with Non-Equidistant Points')

plt.show()

# Calculate the normal error
error = np.abs(Z1 - Z_interp_grid)
print(f"Normal error: {error}")

# Calculate the maximum and mean normal errors
max_error = np.max(error)
print(f"Max error: {max_error}")
mean_normal_error = np.mean(error)
print(f"Mean normal error: {mean_normal_error}")

# Plot the normal error
plt.figure(figsize=(8, 6))
plt.semilogy(x_points, error[0, :], label='Normal Error')
plt.title('Normal Error')
plt.xlabel('x')
plt.ylabel('Error')
plt.legend()
plt.show()