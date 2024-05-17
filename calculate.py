import numpy as np

# distancia entre dos puntos en el plano
def get_distance(x1, y1, x2, y2):
    # Calcula la distancia euclidiana entre los dos puntos
    return np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
    

# calcula el match entre persona del frame izquierdo y derecho
def get_resolution(keypointsL, keypointsR):
    dict_pts0_p = {}
    for index, element in enumerate(keypointsL):
        dict_pts0_p[index] = element

    dict_pts1_p = {}
    for index, element in enumerate(keypointsR):
        dict_pts1_p[index] = element
    dict_pts1_p_cp = dict_pts1_p.copy()
    
    """
    np_pts0_p = np.asarray(get_kps(pts0_p))
    np_pts1_p = np.asarray(get_kps(pts1_p))
    np_pts1_p_cp = np.copy(np_pts1_p)
    print(type(np_pts0_p), type(pts0_p))
    """

    result_person = {}
    person_ind = 0

    for person_left in dict_pts0_p:
        # print(person_left)
        list_dist_first_person = []
        for person_right in dict_pts1_p_cp:
            # print(person_right)
            list_result_distance = []
            for num_point in range(17):
                list_result_distance.append(get_distance(dict_pts0_p[person_left][num_point][0], dict_pts0_p[person_left][num_point][1], dict_pts1_p[person_right][num_point][0], dict_pts1_p[person_right][num_point][1]))
            list_dist_first_person.append(list_result_distance)
        # distancia minima entre puntos el promedio de las distancia en cada punto.
        min_indices = np.argmin([np.mean(distances, axis=0) for distances in list_dist_first_person])
        # con el indice del valor minimo que cuadra obtengo al key(index) del frame derecho
        result_person[person_ind] = list(dict_pts1_p_cp.keys())[int(min_indices)]
        # elimino ese key de la copia y me quedo con los que no han coincidido
        dict_pts1_p_cp.pop(result_person[person_ind])
        # print(dict_pts1_p_cp.keys())
        person_ind+=1


    return result_person
