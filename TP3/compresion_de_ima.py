import numpy as np
from pylab import *
from PIL import Image
import matplotlib.pyplot as plt

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

fig, axs = plt.subplots(4, 4, figsize=(10, 10))

# Mostrar las imágenes en los subplots correspondientes
axs[0, 0].imshow(imagen1, cmap="gray")
axs[0, 1].imshow(imagen2, cmap="gray")
axs[0, 2].imshow(imagen3, cmap="gray")
axs[0, 3].imshow(imagen4, cmap="gray")
axs[1, 0].imshow(imagen5, cmap="gray")
axs[1, 1].imshow(imagen6, cmap="gray")
axs[1, 2].imshow(imagen7, cmap="gray")
axs[1, 3].imshow(imagen8, cmap="gray")
axs[2, 0].imshow(imagen9, cmap="gray")
axs[2, 1].imshow(imagen10, cmap="gray")
axs[2, 2].imshow(imagen11, cmap="gray")
axs[2, 3].imshow(imagen12, cmap="gray")
axs[3, 0].imshow(imagen13, cmap="gray")
axs[3, 1].imshow(imagen14, cmap="gray")
axs[3, 2].imshow(imagen15, cmap="gray")
axs[3, 3].imshow(imagen16, cmap="gray")

# Eliminar los ejes de los subplots
for ax in axs.flat:
    ax.axis('off')

# Agregar el número de imagen debajo de cada imagen
imagen_numbers = ["Imagen 0", "Imagen 1", "Imagen 2", "Imagen 3",
                  "Imagen 4", "Imagen 5", "Imagen 6", "Imagen 7",
                  "Imagen 8", "Imagen 9", "Imagen 10", "Imagen 11",
                  "Imagen 12", "Imagen 13", "Imagen 14", "Imagen 15"]

for i, ax in enumerate(axs.flat):
    ax.text(0.5, -0.1, imagen_numbers[i], transform=ax.transAxes,
            fontsize=13, ha='center')

# Mostrar la figura
plt.tight_layout()
plt.show()

ima= np.array(imagen2)
#print(ima.shape)

matriz = np.column_stack([np.array(imagen1).flatten(), np.array(imagen2).flatten(), np.array(imagen3).flatten(), np.array(imagen4).flatten(), np.array(imagen5).flatten(), np.array(imagen6).flatten(), np.array(imagen7).flatten(), np.array(imagen8).flatten(), np.array(imagen9).flatten(), np.array(imagen10).flatten(), np.array(imagen11).flatten(), np.array(imagen12).flatten(), np.array(imagen13).flatten(), np.array(imagen14).flatten(), np.array(imagen15).flatten(), np.array(imagen16).flatten()])
#print(matriz.shape)
# plt.imshow(matriz, cmap="gray")
# plt.axis('off')
# plt.show()

U, S, VT = np.linalg.svd(matriz)
Sd = np.diag(S)
print(Sd.shape)
print(Sd)
imagenes = []
for k in range(0, 17):
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

