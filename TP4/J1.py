import random
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

M = 5
N = 100

def generar_matriz(m, n):
    return np.random.rand(m, n)

def generar_x(n):
    return np.random.rand(n)

def generar_b(m):
    return np.random.rand(m)

def F(A, x, b):
    return (A @ x - b).T @ (A @ x - b)

def F2(A, x, b, delta2):
    return F(A, x, b) + delta2 * np.linalg.norm(x)**2

def F1(A, x, b, delta1):
    return F(A, x, b) + delta1 * np.linalg.norm(x, ord=1)

def grad_F(A, x, b):
    return 2 * (A.T @ A @ x - A.T @ b)

def grad_F2(A, x, b, delta2):
    return grad_F(A, x, b) + 2 * delta2 * x

def grad_F1(A, x, b, delta1):
    return grad_F(A, x, b) + delta1 * np.sign(x)

def hessiano(A):
    H = 2 * A.T @ A
    return H

def gradiente_descendiente(A, b, x, s, s1, s2, x_svd):
   
    v_F = []
    v_F1 = []
    v_F2 = []

    xf_anterior = x
    xf1_anterior = x
    xf2_anterior = x

    errores = []
    erroresF1 = []
    erroresF2 = []


    for i in range(1000):
        xf_nueva = xf_anterior - s * grad_F(A, xf_anterior, b)
        xf_anterior = xf_nueva

        xf1_nueva = xf1_anterior - s * grad_F1(A, xf1_anterior, b, s1) 
        xf1_anterior = xf1_nueva

        xf2_nueva = xf2_anterior - s * grad_F2(A, xf2_anterior, b, s2) 
        xf2_anterior = xf2_nueva

        v_F.append(F(A, xf_nueva, b)) 
        v_F2.append(F2(A, xf2_nueva, b, s2))
        v_F1.append(F1(A, xf1_nueva, b, s1))

        errores.append(norm(x_svd - xf_nueva))
        erroresF1.append(norm(x_svd - xf1_nueva))
        erroresF2.append(norm(x_svd - xf2_nueva))


    return xf_nueva, xf1_nueva, xf2_nueva, v_F, v_F2, v_F1, errores, erroresF1, erroresF2

A = generar_matriz(M, N)
x = generar_x(N)
b = generar_b(M)

Hessiano = hessiano(A)
autovalores = np.real(np.linalg.eig(Hessiano)[0])
print(autovalores)
max_auto = max(autovalores)
print(max_auto)

s = 1 / max_auto
s1 = 1e-03 * max_auto
s2 = 1e-02 * max_auto

U, S, VT = np.linalg.svd(A, full_matrices=False)


x_svd = np.dot(VT.T, np.dot(np.linalg.inv(np.diag(S)), np.dot(U.T, b)))

f, f1, f2, v_F, v_F2, v_F1, errores, erroresF1, erroresF2 = gradiente_descendiente(A, b, x, s, s1, s2, x_svd)


iteraciones = range(len(v_F))

plt.figure()
plt.rcParams.update({'font.size': 12})
plt.plot(iteraciones, v_F, label='F')
plt.plot(iteraciones, v_F1, label='F1')
plt.plot(iteraciones, v_F2, label='F2')
# #plt.plot(iteraciones, svd, label='SVD')

plt.xlabel('Numero de iteraciones')
plt.ylabel('Valor de la funcion de costo')
plt.title('Convergencia del algoritmo de gradiente descendiente')
#plt.yscale("log")
plt.legend()
plt.show()

print("V_F:", v_F[-1])
print("\n")
print("V_F2:", v_F2[-1])
print("\n")
print("V_F1:", v_F1[-1])
print("\n")



vector_indices = range(1, len(errores) + 1)

plt.figure()
plt.rcParams.update({'font.size': 12})
plt.plot(vector_indices, errores, label='F - SVD')
plt.plot(vector_indices, erroresF2, label='F2 - SVD')
plt.plot(vector_indices, erroresF1, label='F1 - SVD')

plt.xlabel('Numero de iteraciones')
plt.ylabel('Error')
plt.title('Error entre las soluciones obtenidas con gradiente descendiente y SVD')
plt.legend()
plt.show()

print("E_F:", errores[-1])
print("\n")
print("E_F2:", erroresF2[-1])
print("\n")
print("E_F1:", erroresF1[-1])
print("\n")




