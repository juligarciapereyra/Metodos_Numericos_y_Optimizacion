import numpy as np
import math
import pandas as pd
import matplotlib as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
from scipy.interpolate import interp1d

def f(x, y, d1, d2, d3):
    x1, y1 = 0, 10
    x2, y2 = -10, -10
    x3, y3 = 5, -5
    return np.array([math.sqrt((x-x1)**2 + (y-y1)**2) - d1, math.sqrt((x-x2)**2 + (y-y2)**2) - d2, math.sqrt((x-x3)**2 + (y-y3)**2) - d3])

def jacob(x, y): #matriz diferencial para f(x, y)
    x1, y1 = 0, 10
    x2, y2 = -10, -10
    x3, y3 = 5, -5
    
    J = np.zeros((3, 2))
    
    J[0, 0] = (x - x1) / math.sqrt((x - x1)**2 + (y - y1)**2)
    J[1, 0] = (x - x2) / math.sqrt((x - x2)**2 + (y - y2)**2)
    J[2, 0] = (x - x3) / math.sqrt((x - x3)**2 + (y - y3)**2)
    J[0, 1] = (y - y1) / math.sqrt((x - x1)**2 + (y - y1)**2)
    J[1, 1] = (y - y2) / math.sqrt((x - x2)**2 + (y - y2)**2)
    J[2, 1] = (y - y3) / math.sqrt((x - x3)**2 + (y - y3)**2)
    
    return J

def newton(d1, d2, d3, tol=1e-14, max_iter=100): 
    x = -5/3 #centroide como primera aproximacion x0, y0
    y = -5/3 
    
    for i in range(max_iter):
        
        f_val = f(x, y, d1, d2, d3)
        J = jacob(x, y)
        dx_dy = np.dot(np.linalg.pinv(J), -f_val) #calculo ((x,y) - (x0, y0)) hasta que se aproxime a 0
        x += dx_dy[0]
        y += dx_dy[1]
        
        if np.linalg.norm(f(x, y, d1, d2, d3)) < tol:
            return x, y
        
    raise Exception("No converge") #si con max_iter la aproximacion no es buena


#Queremos conseguir (x, y) tal que  f = (0, 0, 0) aproximando con el metodo de newton

def getPosition(d1, d2, d3):
    x, y = newton(d1, d2, d3)
    return x, y

def getTrajectory():
    trajectory = []
    df = pd.read_csv('measurements.csv')
    df2 = pd.read_csv('trajectory.csv')
    
    x_col = df2[' x coordinate (m)']
    y_col = df2[' y coordinate (m)']
    
    x_values = []
    y_values = []
    
    for i in range(len(df2)):
        x = x_col[i]
        y = y_col[i]
        x_values.append(x)
        y_values.append(y)
   
    
    time_col = df['# time (s)']
    d1_col = df[' d1 (m)']
    d2_col = df[' d2 (m)']
    d3_col = df[' d3 (m)']
    
    x, y = -5/3, -5/3
    t_x = []
    t_y = []
    
    for i in range(len(df)):
        t = time_col[i]
        d1 = d1_col[i]
        d2 = d2_col[i]
        d3 = d3_col[i]
        x, y =  getPosition(d1, d2, d3)
        trajectory.append([t, x, y])
        t_x.append(x)
        t_y.append(y)
    
    t_x.append(t_x[0]) #para que se una el ultimo elemento
    t_y.append(t_y[0])
    plot(x_values, y_values, label = 'original')
    title('Trayectoria')
    
    legend()
    plot(t_x, t_y, 'o')
    plot(t_x, t_y, label = 'lineal')
    
    traj_spline_x = interp1d([t[0] for t in trajectory], [t[1] for t in trajectory], kind='cubic')
    traj_spline_y = interp1d([t[0] for t in trajectory], [t[2] for t in trajectory], kind='cubic')
    
    times = np.linspace(time_col.iloc[0], time_col.iloc[-1], 1000)
    
    interp_x = traj_spline_x(times)
    interp_y = traj_spline_y(times)
    
    plot(interp_x, interp_y, label = 'spline')
    legend()
    xlabel('x ')
    ylabel('y ')
    show()
        
getTrajectory()