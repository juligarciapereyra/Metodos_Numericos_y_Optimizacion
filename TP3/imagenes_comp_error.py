import numpy as np
from pylab import *
from PIL import Image
import matplotlib.pyplot as plt
import math

imagenes = []
imagen1 = Image.open("./dataset_imagenes/img00.jpeg")
imagenes.append(np.array(imagen1))
imagen2 = Image.open("./dataset_imagenes/img01.jpeg")
imagenes.append(np.array(imagen2))
imagen3 = Image.open("./dataset_imagenes/img02.jpeg")
imagenes.append(np.array(imagen3))
imagen4 = Image.open("./dataset_imagenes/img03.jpeg")
imagenes.append(np.array(imagen4))
imagen5 = Image.open("./dataset_imagenes/img04.jpeg")
imagenes.append(np.array(imagen5))
imagen6 = Image.open("./dataset_imagenes/img05.jpeg")
imagenes.append(np.array(imagen6))
imagen7 = Image.open("./dataset_imagenes/img06.jpeg")
imagenes.append(np.array(imagen7))
imagen8 = Image.open("./dataset_imagenes/img07.jpeg")
imagenes.append(np.array(imagen8))
imagen9 = Image.open("./dataset_imagenes/img08.jpeg")
imagenes.append(np.array(imagen9))
imagen10 = Image.open("./dataset_imagenes/img09.jpeg")
imagenes.append(np.array(imagen10))
imagen11 = Image.open("./dataset_imagenes/img10.jpeg")
imagenes.append(np.array(imagen11))
imagen12 = Image.open("./dataset_imagenes/img11.jpeg")
imagenes.append(np.array(imagen12))
imagen13 = Image.open("./dataset_imagenes/img12.jpeg")
imagenes.append(np.array(imagen13))
imagen14 = Image.open("./dataset_imagenes/img13.jpeg")
imagenes.append(np.array(imagen14))
imagen15 = Image.open("./dataset_imagenes/img14.jpeg")
imagenes.append(np.array(imagen15))
imagen16 = Image.open("./dataset_imagenes/img15.jpeg")
imagenes.append(np.array(imagen16))

def frobenius (matriz, rows, columns):
    suma = 0
    for i in range(0, rows):
        for j in range(0, columns):
            suma += (matriz[i][j])**2
    return math.sqrt(suma)

k = 12
errores = []
imagenes_ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
for imagen in imagenes:
    U, S, VT = np.linalg.svd(imagen)
    Sd = np.diag(S)
    f_original = frobenius(imagen, 28, 28)
    comprimida = U[:,:k] @ Sd[0:k,:k] @ VT[:k,:]
    error = ((frobenius(imagen - comprimida, 28 , 28))/f_original) * 100
    errores.append(error)

plt.figure(figsize=(12, 6))  # Ajustar el tamaño del gráfico
plt.bar(np.arange(len(imagenes_)), errores, tick_label=imagenes_)
plt.xlabel('Nro de imagen')
plt.ylabel('Porcentaje de error')
plt.title("Errores tomando 12 autovectores en las imagenes")
plt.xticks(rotation=0)  # Rotar las etiquetas del eje X

# Mostrar el gráfico
plt.show()

#---------------------------------------------------

n_autovectores = []

for imagen in imagenes:
    U, S, VT = np.linalg.svd(imagen)
    Sd = np.diag(S)
    f_original = frobenius(imagen, 28, 28)
    for k in range(1, len(imagen)+1):
        comprimida = U[:,:k] @ Sd[0:k,:k] @ VT[:k,:]
        if (frobenius(imagen - comprimida, 28, 28))/f_original <= 0.05:
            n_autovectores.append(k)
            break

imagenes_ = range(len(imagenes))  # Lista de nombres de imágenes (suponiendo que tienes nombres específicos para cada imagen)

plt.figure(figsize=(12, 6))  # Ajustar el tamaño del gráfico
plt.bar(imagenes_, n_autovectores, tick_label=imagenes_)
plt.xlabel('Imagen')
plt.ylabel('Numero de autovectores')
plt.title("Número de autovectores por imagen para que el error sea <= 5%")
plt.xticks(rotation=0)  # Rotar las etiquetas del eje X

# Mostrar el gráfico
plt.show()
