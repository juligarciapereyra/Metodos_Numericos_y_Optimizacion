import numpy as np
from pylab import *
from PIL import Image
import matplotlib.pyplot as plt
import math

def frobenius (matriz, rows, columns):
    suma = 0
    for i in range(0, rows):
        for j in range(0, columns):
            suma += (matriz[i][j])**2
    return math.sqrt(suma)

imagen = Image.open("./dataset_imagenes/img08.jpeg")
comparacion = Image.open("./dataset_imagenes/img09.jpeg")
comparacion= np.array(comparacion)
matriz = np.array(imagen)
print(matriz.shape)

U, S, VT = np.linalg.svd(matriz)
print(U.shape, S.shape, VT.shape)
Sd = np.diag(S)
f_original = frobenius(matriz, 28, 28)
for k in range(1, len(matriz)):
    comprimida = U[:,:k] @ Sd[0:k,:k] @ VT[:k,:]
    if (frobenius(matriz - comprimida,28 ,28))/f_original <= 0.05:
        break
imshow(comprimida)

U2, S2, VT2 = np.linalg.svd(comparacion)
Sd2 = np.diag(S2)
comprimida2 = U2[:,:k] @ Sd2[0:k,:k] @ VT2[:k,:]

print(k)

matriz2 = U[:,:4] @ Sd[0:4,:4] @ VT[:4,:]
plt.imshow(matriz2, cmap="gray")
plt.axis('off')  # Desactivar los ejes
plt.show()


plt.imshow(matriz, cmap="gray")
plt.axis('off')  # Desactivar los ejes
plt.show()

plt.imshow(comprimida)
plt.axis('off')  # Desactivar los ejes
plt.show()


plt.imshow(comprimida2)
plt.axis('off')  # Desactivar los ejes
plt.show()
