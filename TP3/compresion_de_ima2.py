import numpy as np
from pylab import *
from PIL import Image
import matplotlib.pyplot as plt
 
imagenes = []
imagen1 = Image.open("./dataset_imagenes/img00.jpeg")
imagen2 = Image.open("./dataset_imagenes/img01.jpeg")
imagen3 = Image.open("./dataset_imagenes/img02.jpeg")
imagen4 = Image.open("./dataset_imagenes/img03.jpeg")
imagen5 = Image.open("./dataset_imagenes/img04.jpeg")
imagen6 = Image.open("./dataset_imagenes/img05.jpeg")
imagen7 = Image.open("./dataset_imagenes/img06.jpeg")
imagen8 = Image.open("./dataset_imagenes/img07.jpeg")
imagen9 = Image.open("./dataset_imagenes/img08.jpeg")
imagen10 = Image.open("./dataset_imagenes/img09.jpeg")
imagen11 = Image.open("./dataset_imagenes/img10.jpeg")
imagen12 = Image.open("./dataset_imagenes/img11.jpeg")
imagen13 = Image.open("./dataset_imagenes/img12.jpeg")
imagen14 = Image.open("./dataset_imagenes/img13.jpeg")
imagen15 = Image.open("./dataset_imagenes/img14.jpeg")
imagen16 = Image.open("./dataset_imagenes/img15.jpeg")

matriz = np.column_stack([np.array(imagen1).flatten(), np.array(imagen2).flatten(), np.array(imagen3).flatten(), np.array(imagen4).flatten(), np.array(imagen5).flatten(), np.array(imagen6).flatten(), np.array(imagen7).flatten(), np.array(imagen8).flatten(), np.array(imagen9).flatten(), np.array(imagen10).flatten(), np.array(imagen11).flatten(), np.array(imagen12).flatten(), np.array(imagen13).flatten(), np.array(imagen14).flatten(), np.array(imagen15).flatten(), np.array(imagen16).flatten()])
print(matriz.shape)
U, S, VT = np.linalg.svd(matriz)
print(len(S))
Sd = np.diag(S)
imagenes = []
print(U.shape, Sd.shape)
for k in range(0, 20):
    columnas = U.T
    imagen_array = columnas[k]
    imagenes.append(imagen_array.reshape((28, 28)))
# Configuración de los subplots
num_imagenes = len(imagenes)
num_filas = int(np.ceil(np.sqrt(num_imagenes)))
num_columnas = int(np.ceil(num_imagenes / num_filas))

# Crear los subplots
fig, axs = plt.subplots(num_filas, num_columnas, figsize=(12, 12))  # Ajusta el tamaño de la figura según tus necesidades

# Mostrar las imágenes en los subplots
for i, imagen in enumerate(imagenes):
    fila = i // num_columnas
    columna = i % num_columnas
    axs[fila, columna].imshow(imagen, cmap='gray')
    axs[fila, columna].axis('off')
    axs[fila, columna].text(0.5, -0.1, f'Autovector={i}', transform=axs[fila, columna].transAxes, ha='center', va='top')

# Ajustar el espaciado entre subplots y mostrar la figura
plt.tight_layout()
plt.show()

print(S)
plt.plot(S, marker="o", markersize=5)
plt.xlabel("Índice")
plt.ylabel("Valor")
plt.legend(["Valores singulares"])
plt.title("Valores singulares de la matriz")
plt.show()