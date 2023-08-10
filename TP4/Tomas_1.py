import random
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

M = 5
N = 100

def generar_matriz(m, n):
    return np.random.randn(m, n)

def generar_x(n):
    return np.random.randn(n, 1)

def generar_b(m):
    return np.random.randn(m, 1)

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

def gradiente_descendiente(A, b, x, s, s1, s2, error):
    errores = [norm(error - x)]
    erroresF1 = [norm(error - x)]
    erroresF2 = [norm(error - x)]

    xf_anterior = x
    xf1_anterior = x
    xf2_anterior = x

    costos_f = []
    costos_f1 = []
    costos_f2 = []

    x_ = []
    x_1 = []
    x_2 = []
    for i in range(1000):
        xf_nueva = xf_anterior - s * grad_F(A, xf_anterior, b)
        xf_anterior = xf_nueva

        xf1_nueva = xf1_anterior - s * grad_F1(A, xf1_anterior, b, s1) #1e-03 * S[0]
        xf1_anterior = xf1_nueva

        xf2_nueva = xf2_anterior - s * grad_F2(A, xf2_anterior, b, s2) #1e-02 * S[0]
        xf2_anterior = xf2_nueva

        costo_f = np.linalg.norm(A @ xf_nueva - b) ** 2
        costos_f.append(costo_f)

        costo_f1 = np.linalg.norm(A @ xf1_nueva - b) ** 2 + 1e-03 * np.linalg.norm(xf1_nueva, ord=1)
        costos_f1.append(costo_f1)

        costo_f2 = np.linalg.norm(A @ xf2_nueva - b) ** 2 + 1e-02 * np.linalg.norm(xf2_nueva) ** 2
        costos_f2.append(costo_f2)

        x_.append(norm(F(A, xf_nueva, b)))
        x_1.append(norm(F1(A, xf_nueva, b, s1)))
        x_2.append(norm(F2(A, xf_nueva, b, s2)))

        errores.append(norm(error - xf_nueva))
        erroresF1.append(norm(error - xf1_nueva))
        erroresF2.append(norm(error - xf2_nueva))
    return xf_nueva, xf1_nueva, xf2_nueva, errores, erroresF1, erroresF2, costos_f, costos_f1, costos_f2, x_, x_1, x_2

A = generar_matriz(M, N)
x = generar_x(N)
b = generar_b(M)
Hessiano = hessiano(A)
U, S, VT = np.linalg.svd(Hessiano)
s = 1 / S[0]
U, S, VT = np.linalg.svd(A, full_matrices=False)
s1 = 1e-03 * S[0]
s2 = 1e-02 * S[0]

x_svd = np.dot(VT.T, np.dot(np.linalg.inv(np.diag(S)), np.dot(U.T, b)))

f1, f2, f3, errores, erroresF1, erroresF2, costos_f, costos_f1, costos_f2, x_, x_1, x_2 = gradiente_descendiente(A, b, x, s, s1, s2, x_svd)

print(s)

# Generar vector para el eje x
vector_indices = range(1, len(errores) + 1)

plt.figure()
plt.plot(vector_indices, errores, label='F - SVD')
plt.plot(vector_indices, erroresF2, label='F2 - SVD')
plt.plot(vector_indices, erroresF1, label='F1 - SVD')

plt.xlabel('Iteracion')
plt.ylabel('Error')
plt.title('Error entre las soluciones obtenidas y SVD')
plt.legend()
plt.show()

iteraciones = range(len(costos_f))

plt.figure()
plt.plot(iteraciones, x_, label='F')
plt.plot(iteraciones, x_1, label='F1')
plt.plot(iteraciones, x_2, label='F2')

plt.xlabel('Iteración')
plt.ylabel('Valor de costo')
plt.title('Convergencia del gradiente descendente con diferentes funciones de costo (escala logarítmica)')
plt.yscale("log")
plt.legend()
plt.show()

print(norm(f1))
print(norm(f2))
print(norm(f3))
print(norm(x_svd))
