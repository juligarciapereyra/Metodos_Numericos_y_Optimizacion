import random
import numpy as np
import matplotlib.pyplot as plt

M = 100
N = 100
condicion = 2
tolerancia = 1e-2

def generar_matriz(m, n):
    matriz = []
    for i in range(m):
        fila = []
        for j in range(n):
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

def gradiente_descendente(A, b, x):
    Hessiano = hessiano(A)
    U, S, VT = np.linalg.svd(Hessiano)
    s = 1 / S[0]
    xf_anterior = x
    errores = []
    num_it = 0
    while True:
        xf_nueva = xf_anterior - s * grad_F(A, xf_anterior, b)
        num_it += 1
        error = np.linalg.norm(A @ xf_nueva - b)
        errores.append(error)
        if error <= tolerancia:
            break
        xf_anterior = xf_nueva
    return xf_nueva, errores, num_it

def predicted_iterations(lambda_max, lambda_min, error_metodo, error_inicial):
    error = error_inicial * (1 - lambda_min / lambda_max)
    k = 2
    while error >= error_metodo:
        error = error_inicial * (1 - lambda_min / lambda_max) ** k
        k += 1
    return k

A = generar_matriz_condicion(generar_matriz(M, N), condicion)
cond_A = np.linalg.cond(A)
x, errores, num_iter = gradiente_descendente(A, generar_b(M), generar_x_random(N))
H = hessiano(A)
U, S, VT = np.linalg.svd(H)
predict = predicted_iterations(S[0], S[-1], tolerancia, errores[0])

# Comparación de la función de costo a cada paso con la prediccion teórica
k = range(len(errores))
prediccion_teorica = [errores[0] * (1 - S[-1] / S[0]) ** i for i in k]

plt.plot(k, errores, label='Función de costo')
plt.plot(k, prediccion_teorica, label='Prediccion teórica')
plt.xlabel('Iteración')
plt.ylabel('Valor de costo')
plt.title('Comparación de la función de costo con la prediccion teórica')
plt.legend()
plt.show()

print("Número de iteraciones obtenido:", num_iter)
print("Número de iteraciones predicho:", predict)
print("Número de condición de A:", cond_A)
