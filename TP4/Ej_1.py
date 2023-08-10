import random
import numpy as np

M = 5
N = 100

def generar_matriz(m, n):
    matriz = []
    for i in range (m):
        fila = []
        for j in range (n):
            number = random.randint(1, 100)
            fila.append(number)
        matriz.append(fila)
    return np.array(matriz)

def generar_x(n):
    x = []
    for i in range(n):
        x.append(0)
    return np.array(x)

def generar_b(m):
    b = []
    for i in range(m):
        b.append(random.randint(1, 100))
    return np.array(b)

def grad_F(A, x, b):
    return 2 * A.T @ A @ x - 2 * A.T @ b

def grad_F2(A, x, b, delta2):
    return grad_F(A, x, b) + 2 * delta2 * x

def grad_F1(A, x, b, delta1):
    return grad_F(A, x, b) + delta1 * np.sign(x)

def hessiano(A):
    H = 2 * A.T @ A
    return H

def gradiente_descendiente(A, b, x):
    Hessiano = hessiano(A)
    U, S, VT = np.linalg.svd(Hessiano)
    s = 1 / S[0]
    xf_anterior = x
    xf1_anterior = x
    xf2_anterior = x

    costos_f = []
    costos_f1 = []
    costos_f2 = []


    for i in range(1000):
        xf_nueva = xf_anterior - s * grad_F(A, xf_anterior, b)
        xf_anterior = xf_nueva
        xf1_nueva = xf1_anterior - s * grad_F1(A, xf1_anterior, b, 47365) #1e-03 * S[0]
        xf1_anterior = xf1_nueva
        xf2_nueva = xf2_anterior - s * grad_F2(A, xf2_anterior, b, 47263) #1e-02 * S[0]
        xf2_anterior = xf2_nueva

        costo_f = np.linalg.norm(A @ xf_nueva - b) ** 2
        costos_f.append(costo_f)

        costo_f1 = np.linalg.norm(A @ xf1_nueva - b) ** 2 + 1e-03 * np.linalg.norm(xf1_nueva, ord=1)
        costos_f1.append(costo_f1)

        costo_f2 = np.linalg.norm(A @ xf2_nueva - b) ** 2 + 1e-02 * np.linalg.norm(xf2_nueva) ** 2
        costos_f2.append(costo_f2)

    return xf_nueva, xf1_nueva, xf2_nueva, 1e-03 * S[1], 1e-02 * S[1], S[0], costos_f, costos_f1, costos_f2

A = generar_matriz(M, N)
x = generar_x(N)
b = generar_b(M)

f1, f2, f3, delta1, delta2 , s, costos_f, costos_f1, costos_f2 = gradiente_descendiente(A, b, x)

U, S, VT = np.linalg.svd(A, full_matrices=False)

x_svd = np.dot(VT.T, np.dot(np.linalg.inv(np.diag(S)), np.dot(U.T, b)))
x_svd_F2 = np.dot(VT.T, np.dot(np.linalg.inv(np.diag(S**2 + delta2)), np.dot(U.T, b)))
x_svd_F1 = np.dot(VT.T, np.dot(np.linalg.inv(np.diag(S + delta1)), np.dot(U.T, b)))

print("Solución minimizando F (gradiente descendente):", f1)
print("\n")
print("Solución minimizando F2 (gradiente descendente):", f3)
print("\n")
print("Solución minimizando F1 (gradiente descendente):", f2)
print("\n")
print("Solución minimizando F (SVD):", x_svd)
print("\n")
print("Solución minimizando F2 (SVD):", x_svd_F2)
print("\n")
print("Solución minimizando F1 (SVD):", x_svd_F1)
print("\n")


print(f1 - x_svd)
print("\n")
print(f3 - x_svd_F2)
print("\n")
print(f2 - x_svd_F1)
print("\n")

print(s)
print("\n")



#Comparacion de las soluciones ------------------------------------------------------

import matplotlib.pyplot as plt

indices = range(len(f1))

plt.figure()
plt.bar(indices, f1, label='Gradiente Descendente - F')
plt.bar(indices, f2, label='Gradiente Descendente - F1')
plt.bar(indices, f3, label='Gradiente Descendente - F2')
plt.xlabel('Componente')
plt.ylabel('Valor')
plt.title('Soluciones con gradiente descendiente')
plt.legend()
plt.show()


plt.bar(indices, x_svd, label='SVD')
plt.bar(indices, x_svd_F1, label='SVD - F1')
plt.bar(indices, x_svd_F2, label='SVD - F2')
plt.xlabel('Componente')
plt.ylabel('Valor')
plt.title('Soluciones con SVD')
plt.legend()
plt.show()

#Error--------------------
indices = np.arange(len(f1))

plt.figure()
plt.plot(indices, f1 - x_svd, label='F - SVD')
plt.plot(indices, f3 - x_svd_F2, label='F2 - SVD')
plt.plot(indices, f2 - x_svd_F1, label='F1 - SVD')

plt.xlabel('Componente')
plt.ylabel('Error')
plt.title('Error entre las soluciones obtenidas y SVD')
plt.legend()
plt.show()


#Convergencia del gradiente descendente-------------------------------------
import matplotlib.pyplot as plt

iteraciones = range(len(costos_f))

plt.figure()
plt.plot(iteraciones, costos_f, label='F')
plt.plot(iteraciones, costos_f1, label='F1')
plt.plot(iteraciones, costos_f2, label='F2')

plt.xlabel('Iteración')
plt.ylabel('Valor de costo')
plt.title('Convergencia del gradiente descendente con diferentes funciones de costo')
plt.legend()
plt.show()



plt.figure()
plt.plot(iteraciones, costos_f, label='F')
plt.plot(iteraciones, costos_f1, label='F1')
plt.plot(iteraciones, costos_f2, label='F2')

plt.xlabel('Iteración')
plt.ylabel('Valor de costo')
plt.title('Convergencia del gradiente descendente con diferentes funciones de costo (escala logarítmica)')
plt.yscale('log')  # Aplica escala logarítmica en el eje y
plt.legend()
plt.show()







