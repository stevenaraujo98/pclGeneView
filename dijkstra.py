import heapq

#Codigo Grafo
class Graph:
    def __init__(self):
        self.graph = {}
    def add_vertex(self, vertex):
        if vertex not in self.graph:
            self.graph[vertex] = {}
    def add_edge(self, vertex1, vertex2, weight=None, directed=False):
        if vertex1 not in self.graph:
            self.add_vertex(vertex1)
        if vertex2 not in self.graph:
            self.add_vertex(vertex2)
        self.graph[vertex1][vertex2] = weight
        if not directed:
            self.graph[vertex2][vertex1] = weight

    def dijkstra(self, start):
        distances = {vertex: float('infinity') for vertex in self.graph}
        distances[start] = 0
        queue = [(0, start)]
        # Comienza el bucle principal mientras haya elementos en la cola de prioridad.
        while queue:
            # Extrae el vértice con la distancia mínima actual desde el vértice de inicio.
            current_distance, current_vertex = heapq.heappop(queue)
            # Comprueba si la distancia actual extraída es mayor que la distancia almacenada en el diccionario de distancias.
            # Si es mayor, esto significa que hemos encontrado una distancia más corta a este vértice en un paso anterior, por lo que lo ignoramos.
            if current_distance > distances[current_vertex]:
                continue
            # Explora los vértices adyacentes al vértice actual.
            for neighbor, weight in self.graph[current_vertex].items():
                # Calcula la distancia acumulada hasta el vecino desde el vértice de inicio.
                distance = current_distance + weight
                # Si la distancia acumulada es menor que la distancia almacenada en el diccionario de distancias para este vecino,
                # actualiza la distancia almacenada y agrega el vecino a la cola de prioridad para explorar sus vecinos.
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(queue, (distance, neighbor))
        return distances

"""
#Creamos el Grafo
g = Graph()
# Agregar vértices
g.add_vertex('A')
g.add_vertex('B')
g.add_vertex('C')
g.add_vertex('D')
# Agregar conexiones
g.add_edge('A', 'B', 3)
g.add_edge('A', 'C', 2)
g.add_edge('B', 'C', 1)
g.add_edge('B', 'D', 5)
g.add_edge('C', 'D', 4)
# Ejecutar algoritmo de Dijkstra
start_vertex = 'A'
distances = g.dijkstra(start_vertex)
# Mostrar resultados
print("Distancias mínimas desde el vértice", start_vertex + ":")
for vertex, distance in distances.items():
    print("Distancia al vértice", vertex + ":", distance)
"""

"""
import networkx as nx 
import random

G = nx.Graph()
G.add_edge(1, 2, weight = 450)
G.add_edge(2, 1, weight = 450)
G.add_edge(1, 3, weight = 390)
G.add_edge(3, 1, weight = 390)
G.add_edge(1, 4, weight = 550)
G.add_edge(4, 1, weight = 550)
G.add_edge(1, 8, weight = 1310)
G.add_edge(8, 1, weight = 1310)
G.add_edge(2, 3, weight = 300)

node_1 = 2
node_2 = 8


# shortest path
print("Le plus court chemin de ",node_1," a ",node_2," est :")
liste=nx.dijkstra_path(G,node_1,node_2)
print (liste) 

# length of the shortest path 
print("la longueur du plus court chemin est ")
print (nx.shortest_path_length(G,source=node_1,target=node_2))
"""