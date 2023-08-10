import random
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

M = 100
N = 100
condicion = 38

def generar_matriz(m, n):
    matriz = []
    for i in range (m):
        fila = []
        for j in range (n):
            number = random.randint(1, 100)
            fila.append(number)
        matriz.append(fila)
    return np.array(matriz)

def generar_matriz_condicion(matriz, numero_deseado):
    U, S, VT = np.linalg.svd(matriz)
    for i in range(len(S)):
        if S[i] > numero_deseado:
            S[i] = numero_deseado
        elif S[i] < 1:
            S[i] = 1
    S[-1] = 1
    return U @ np.diag(S) @ VT

def generar_x_random(n):
    x = []
    for i in range(n):
        x.append(random.randint(-100, 100))
    return np.array(x)

def generar_b(m):
    b = []
    for i in range(m):
        b.append(random.randint(1, 100))
    return np.array(b)

def grad_F(A, x, b):
    return 2 * A.T @ A @ x - 2 * A.T @ b

def hessiano(A):
    H = 2 * A.T @ A
    return H

def gradiente_descendiente(A, b, x):
    Hessiano = hessiano(A)
    U, S, VT = np.linalg.svd(Hessiano)
    s = 1 / S[0]
    xf_anterior = x
    num_it = 0
    while True:
        xf_nueva = xf_anterior - s * grad_F(A, xf_anterior, b)
        num_it += 1
        if norm(xf_nueva - xf_anterior) <= 1e-02:
            break
        xf_anterior = xf_nueva
    return xf_nueva, num_it

def predicted_iterations(lambda_max, lambda_min, tol):
    error = 10000
    k = 0
    while error >= tol:
        error = (1-lambda_min/lambda_max)**k
        k += 1
    return k

A = generar_matriz_condicion(generar_matriz(M, N), condicion)
cond_A = np.linalg.cond(A)
x, num_iter = gradiente_descendiente(A, generar_b(M), generar_x_random(N))
H = hessiano(A)
U, S, VT =   U, S, VT = np.linalg.svd(H)
predict = predicted_iterations(S[0], S[-1], 1e-02)
print("Número de iteraciones obtenido:", num_iter)
print("Número de iteraciones predicho:", predict)
print(cond_A)

