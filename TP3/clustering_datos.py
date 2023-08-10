import pandas as pd
import math
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(10000)
nombre_archivo = "dataset_clusters.csv"

def K1(fila1, fila2, sigma):
    fila_restada = fila1 - fila2
    suma = 0
    for element in fila_restada:
        suma += (element ** 2)       
    return np.exp(- 1 * ((math.sqrt(suma)) / (2 * (sigma**2))))

df = pd.read_csv("dataset_clusters.csv")
filas = []
for index, fila in df.iterrows():
    fila_arr = np.array(fila)
    filas.append(fila_arr)  


def pca(n): 
    U, S, V = np.linalg.svd(filas)
    componentes = V[:n]
    reducidos = np.dot(filas, componentes.T)
    return reducidos
    
datos_reducidos = pca(2)
print(datos_reducidos)


# pca = PCA(n_components=2)
# datos_reducidos = pca.fit_transform(filas)

matriz_s = []
varianza = np.var(datos_reducidos)
for fila in datos_reducidos:
    fila_similaridad = []
    for fila2 in datos_reducidos:
        similaridad = K1(fila, fila2, varianza) #varianza de los datos_reducidos 3.1061116247596496
        fila_similaridad.append(similaridad)
    matriz_s.append(fila_similaridad)

x = datos_reducidos[:, 0]
y = datos_reducidos[:, 1]

plt.scatter(x, y)
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.title('Gráfico de datos reducidos en dos dimensiones')
plt.show()

def create_cluster(datos_, utilizables, matriz_similaridad, tolerancia, i):
    new_cluster = [i]
    cercanos = []
    for puede_utilizar in utilizables:
        if matriz_similaridad[i][puede_utilizar] >= tolerancia:
            cercanos.append(puede_utilizar)
            utilizables.remove(puede_utilizar)
    for cercano in cercanos:
        continuacion, utilizables = create_cluster(datos_, utilizables, matriz_similaridad, tolerancia, cercano)
        for elements in continuacion:
            new_cluster.append(elements)
    return new_cluster, utilizables

    
def make_clusters(por_utilizar, matriz_similitud, datos, tolerancia):
    clusters = []
    for no_usado in por_utilizar:
        i = no_usado
        por_utilizar.remove(no_usado)
        cluster, usar = create_cluster(datos, por_utilizar, matriz_similitud, tolerancia, i)
        clusters.append(cluster)
        otros_cluster = make_clusters(usar, matriz_similitud, datos, tolerancia)
        for nuevos_clusters in otros_cluster:
            clusters.append(nuevos_clusters)
        if len(por_utilizar) == 0:
            break
    return clusters
    
sin_usar = []
for i in range (0, 1999):
    sin_usar.append(i)

def centroide(cluster_obtenidos, datos_reducidos):
    promedios = []
    for cluster in cluster_obtenidos: 
        values = [] 
        for i in cluster: 
            values.append(datos_reducidos[i])
        promedio = np.mean(values,axis=0 )
        promedios.append(promedio)
    print(promedios)
    x, y =list(zip(*promedios))
    plt.scatter(x,y, color = "red")


def plot_clusters(clusters_index, datos):
    print(len(clusters_index))
    colors = ['blue', 'yellow', "black", "purple", "green"]
    fig, ax = plt.subplots()

    i = 0
    for cluster in clusters_index:
        x = []
        y = []
        for index in cluster:
            x.append(datos[index][0])
            y.append(datos[index][1])
            color = colors[i % len(colors)]
            ax.scatter(x, y, color=color)
        i = i + 1
    p = centroide(clusters_index, datos_reducidos)
    print(p)
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.title('Gráfico de clusters en dos dimensiones')
    plt.show()
cluster_obtenidos = make_clusters(sin_usar, matriz_s, datos_reducidos, 0.97877) #788


def juntar_1(cluster_obtenidos):
    nuevos_clusters = []
    index_no_usados = []
    for cluster in cluster_obtenidos:
        if len(cluster) <= 7:
            for indice in cluster:
                max = 0
                index = 0
                i = 0
                for numero in matriz_s[indice]:
                    if numero > max:
                        max = numero
                        index = i
                    i = i + 1
                for cluster_grande in cluster_obtenidos:
                    if index in cluster_grande and index not in cluster and len(cluster_grande) > 7:
                            cluster_grande.append(indice)
                            cluster.remove(indice)
                    elif index in cluster_grande:
                        index_no_usados.append(index) 
        else:
            nuevos_clusters.append(cluster)
    while(index_no_usados != []):
        for dato in index_no_usados:
            maximo = 0
            inde = 0
            ind = 0
            for s in matriz_s[dato]:
                if s > maximo and ind not in index_no_usados:
                    for cluster__ in nuevos_clusters:
                        if ind in cluster__:
                            maximo = s
                            inde = ind 
                ind = ind + 1
            for cluster_ in nuevos_clusters:
                if inde in cluster_ and dato not in cluster_:
                    cluster_.append(dato)
                    index_no_usados.remove(dato)     
    return nuevos_clusters

clusters_finales = juntar_1(cluster_obtenidos)
plot_clusters(clusters_finales, datos_reducidos)

for cluster in clusters_finales: 
    print(len(cluster))
    print(cluster)




