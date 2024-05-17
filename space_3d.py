import numpy as np
# from dijkstra import Graph
import networkx as nx

def calcular_centroide(lista_de_listas):
    # Inicializa las sumas de las coordenadas x, y, z
    sum_x = sum_y = sum_z = 0
    
    # Itera sobre cada lista en la lista de listas
    for punto in lista_de_listas:
        # Suma las coordenadas x, y, z de cada punto
        sum_x += punto[0]
        sum_y += punto[1]
        sum_z += punto[2]
    
    # Calcula el promedio de las coordenadas x, y, z
    total_puntos = len(lista_de_listas)
    centroide_x = sum_x / total_puntos
    centroide_y = sum_y / total_puntos
    centroide_z = sum_z / total_puntos
    
    return (centroide_x, centroide_y, centroide_z)

def show_each_point_of_person(left_kpts, right_kpts, baseline, f_px, center, list_color_to_paint, ax, plot_3d, body_3d, list_points_persons):
    # Ilustrar cada punto en 3D
    for left_k, right_k, color in zip(left_kpts, right_kpts, list_color_to_paint):
        points = body_3d(left_k, right_k, baseline, f_px, center)

        # Filtro de puntos menores a 1000 y mayores a 10000
        filtered_points = [[a, b, c] for a, b, c in zip(*points) if 1000 < c <= 10000]
        for point in filtered_points:
            plot_3d(point[0], point[1], point[2], ax, color)
        list_points_persons.append(filtered_points)

def get_vector_normal_to_plane(person, centroide, ax, color):
    # Vector perpendicular al plano
    # Puntos que definen el plano en el espacio tridimensional
    p1 = np.array(person[0])
    p2 = np.array(person[1])
    p3 = np.array(person[2])
    # p4 = np.array(person[3])

    # Calcular el vector normal al plano
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)

    # Definir un punto en la línea
    point_on_line = centroide
    # Definir el vector director de la línea (el vector normal al plano)
    line_direction = normal

    # Parámetros para la ecuación paramétrica de la línea
    # Definir el rango para los valores de parámetro t
    t_min = -5
    t_max = 5
    num_points = 100
    t_values = np.linspace(t_min, t_max, num_points)

    # Calcular el punto de intersección entre la línea y el plano
    intersection_point = point_on_line + line_direction * t_values[:, np.newaxis]
    intersection_point_on_plane = intersection_point[
        np.abs(intersection_point - point_on_line).sum(axis=1).argmin()
    ]

    # Graficar la línea perpendicular al plano
    ax.plot(
        [point_on_line[0], intersection_point_on_plane[0]],
        [point_on_line[1], intersection_point_on_plane[1]],
        [point_on_line[2], intersection_point_on_plane[2]],
        color=color
    )

def show_centroid_and_normal(list_points_persons, list_color_to_paint, ax, list_centroides, plot_3d):
    # Ilustrar los centroides de cada persona
    index=0
    for person, color in zip(list_points_persons, list_color_to_paint):
        # Puede ocasionar que person no pase el filtro por lo que se debe validar
        if len(person) < 4:
            continue
        elif (len(person) == 4):
            points_match_body = [(1,2), (1,3), (2,3)]
        else:
            points_match_body = [(1,2), (1,3), (2,4), (3,4)]

        for point in points_match_body:
            ax.plot([person[point[0]][0], person[point[1]][0]], 
                    [person[point[0]][1], person[point[1]][1]], 
                    [person[point[0]][2], person[point[1]][2]], color)
        
        centroide = calcular_centroide(person)
        get_vector_normal_to_plane(person, centroide, ax, color)        

        plot_3d(centroide[0], centroide[1], centroide[2], ax, color, s=400, marker='o', label="C"+str(index))

        # Centroide a la nariz
        plot_3d(centroide[0], person[0][1], centroide[2], ax, color, s=100, marker='o', label="C"+str(index))
        # unir con una linea 2 puntos
        ax.plot([centroide[0], person[0][0]], [person[0][1], person[0][1]], [centroide[2], person[0][2]], color)

        # plot_3d(centroide[0], centroide[1], centroide[2], ax, color, s=400, marker='o', label="C"+str(index))
        list_centroides.append(centroide)
        index+=1

def show_connection_points(list_centroides, ax):
    # Calcular la distancia entre puntos en un plano 3D
    # g = Graph()
    G = nx.Graph() # G.clear()

    """
    # Agregar vértices
    for i in range(len(list_centroides)):
        g.add_vertex(i)
    """

    # Agregar conexiones y distancias entre los centroides
    for i in range(len(list_centroides)):
        for j in range(i+1, len(list_centroides)):
            point1 = list_centroides[i]
            point2 = list_centroides[j]

            distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)
            # g.add_edge(i, j, distance)
            G.add_edge(i, j, weight = distance)

    """
    # la distancia minima de cada punto asia cada punto, pero se puede encontrar una sola linea porque la minima de un punto a otro es la misma
    # res_sorted = []
    for start_vertex in range(len(list_centroides)-1):
        distances = g.dijkstra(start_vertex)
        print("Distancias mínimas desde el vértice", start_vertex, distances)
        # Ordenar el diccionario por los valores de los items
        sorted_dict = dict(sorted(distances.items(), key=lambda item: item[1]))
        # res_sorted.append((start_vertex, list(sorted_dict.items())[1][0]))
        print((start_vertex, list(sorted_dict.items())[1][0]))
        print(nx.dijkstra_path(G,start_vertex, list(sorted_dict.items())[1][0]))
    """

    # Escoger el camino más corto entre cada centroide sin repetir
    res_sorted = []
    for start_vertex in range(len(list_centroides)-1):
        tmp_res = {}
        for end_vertex in range(start_vertex, len(list_centroides)):
            if (start_vertex == end_vertex):
                continue
            # nodos para la distancia minima
            # print(nx.dijkstra_path(G, start_vertex, end_vertex))
            # la distancia minima
            # print(nx.shortest_path_length(G, source=start_vertex, target=end_vertex, weight='weight'))
            tmp_res[(start_vertex, end_vertex)] = nx.shortest_path_length(G, source=start_vertex, target=end_vertex, weight='weight')
        sorted_dict = dict(sorted(tmp_res.items(), key=lambda item: item[1]))
        res_sorted.append(list(sorted_dict.items())[0][0])

    # Unir los centroides con líneas
    for i, j in res_sorted:
        ax.plot([list_centroides[i][0], list_centroides[j][0]],
                [list_centroides[i][1], list_centroides[j][1]],
                [list_centroides[i][2], list_centroides[j][2]], color='black')
