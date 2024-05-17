import open3d as o3d
import numpy as np
from getKeypointsFile import getKeypoints
import cv2

figure = None
lista_colores = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
color_map = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 0]]
list_colors = [(255,0,255), (0, 255, 255), (255, 0, 0), (0, 0, 0), (255, 255, 0), (205, 92, 92), (255, 0, 255), (0, 128, 128), (128, 0, 0), (128, 128, 0), (128, 128, 128)]

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

def graph_circles(array_pts, w, h, frame):
    num_person = 0
    for person in array_pts:
        for pt in person:
            pos_x = int(pt[0])
            pos_y = int(pt[1])
            if (pos_x != 0 and pos_y != 0):
                # print(pos_x, pos_y)
                cv2.circle(frame, (pos_x, pos_y), 3, list_colors[num_person], 3)
        num_person+=1


def live_plot_3d(left_kpts, right_kpts, baseline, f_px, center):
    list_points = []
    list_color_to_paint = []
    list_np = []
    # Agregar a una lista de colores para pintar los puntos de cada persona en caso de ser mas de len(lista_colores)
    for i in range(len(left_kpts)):
        indice_color = i % len(lista_colores)
        list_color_to_paint.append(lista_colores[indice_color])

    for left_k, right_k, color in zip(left_kpts, right_kpts, list_color_to_paint):
        points = body_3d(left_k, right_k, baseline, f_px, center)
        data_np = np.empty((0, 3))

        # Filtro de puntos menores a 1000 y mayores a 10000 
        filtered_points = [[a, b, c] for a, b, c in zip(*points) if 1000 < c <= 10000]
        for point in filtered_points:
            #plot_3d(point[0], point[1], point[2], ax, color)
            data_np = np.append(data_np, np.array([[point[0], point[1], point[2]]]), axis=0)
        list_points.append(filtered_points)
        print("filtered_points", len(filtered_points))
        list_np.append(data_np)

    for person, color in zip(list_points, list_color_to_paint):
        # Puede ocasionar que person no pase el filtro por lo que se debe validar
        if len(person) == 0:
            continue
        centroide = calcular_centroide(person)
        # plot_3d(centroide[0], centroide[1], centroide[2], ax, color, s=100, marker='o')
        # data_np = np.append(data_np, np.array([[centroide[0], centroide[1], centroide[2]]]), axis=0)
        print("centroide", centroide)
    
    return list_points, list_np

def find_depth_from_disparities(right_points, left_points, baseline, f_pixel):
    x_right = np.array(right_points)
    x_left = np.array(left_points)
    disparity = np.abs(x_left - x_right)
    z_depth = (baseline * f_pixel) / disparity
    return np.mean(z_depth)

def body_3d(face_left, face_right, baseline, f_px, center_left):
    assert len(face_left) == len(face_right)

    z = [find_depth_from_disparities(
        [x1[0]], [x2[0]], baseline, f_px) for x1, x2 in zip(face_left, face_right)]

    x = (face_left[:, 0] - center_left[0]) * z / f_px
    y = (face_left[:, 1] - center_left[1]) * z / f_px

    return x, y, z

def point_cloud(left_kpts, right_kpts, baseline, f_px, center):
    return live_plot_3d(left_kpts, right_kpts, baseline, f_px, center)

def getStereoRectifier(calib_file):
    """Build rectifier from stereo map file
    Parameters:
        calib_file (str): file name of the stereo map file generated with calibration procedure
    Returns:
        (np.ndarray, np.ndarray)->(np.ndarray,np.ndarray) rectify function takes 2 unrectified images and returns those images calibrated
    """

    # Camera parameters to undistort and rectify images
    cv_file = cv2.FileStorage()
    cv_file.open(calib_file, cv2.FileStorage_READ)

    stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
    stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
    stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
    stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

    def undistortRectify(frameL, frameR):

        # Undistort and rectify images
        undistortedL = cv2.remap(frameL, stereoMapL_x, stereoMapL_y,
                                 cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        undistortedR = cv2.remap(frameR, stereoMapR_x, stereoMapR_y,
                                 cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

        return undistortedL, undistortedR

    return undistortRectify

# Ejemplo de uso:
baseline = 58
f_px = 3098.3472392388794
center = (777.6339950561523,
            539.533634185791)



# YOLO ------------------------------------------------------------------------------------------------------------
capL=cv2.VideoCapture('./predicts/YOLO/newcalibration/15_58_03_05_04_2024_VIDEO_LEFT.avi')
path_file_left = './predicts/YOLO/newcalibration/15_58_03_05_04_2024_VIDEO_LEFT/15_58_03_05_04_2024_VIDEO_LEFT_'
capR=cv2.VideoCapture('./predicts/YOLO/newcalibration/15_58_03_05_04_2024_VIDEO_RIGHT.avi')
path_file_right = './predicts/YOLO/newcalibration/15_58_03_05_04_2024_VIDEO_RIGHT/15_58_03_05_04_2024_VIDEO_RIGHT_'

"""
# LightGlue ------------------------------------------------------------------------------------------------------------
capL=cv2.VideoCapture('./predicts/LightGlue/16_35_42_26_02_2024_VID_LEFT_calibrated.avi')
path_file_left = './predicts/LightGlue/16_35_42_26_02_2024_VID_LEFT/16_35_42_26_02_2024_VID_LEFT_'
capR=cv2.VideoCapture('./predicts/LightGlue/16_35_42_26_02_2024_VID_RIGHT_calibrated.avi')
path_file_right = './predicts/LightGlue/16_35_42_26_02_2024_VID_RIGHT/16_35_42_26_02_2024_VID_RIGHT_'
"""

frame_num = 1
step_frames = 256
while(capR.isOpened() and capL.isOpened()):
    if step_frames > frame_num:
        capL.set(cv2.CAP_PROP_POS_FRAMES, step_frames)
        capR.set(cv2.CAP_PROP_POS_FRAMES, step_frames)
        frame_num = step_frames

    ret,frameL = capL.read()
    retR,frameR = capR.read()

    h = frameR.shape[0]
    w = frameR.shape[1]

    if(not ret or not retR):
        print("Failed to read frames")
        break
    
    cv2.imshow('LEFT',frameL)
    # cv2.imshow('RIGHT',frameR)

    
    keypointsL = np.array(getKeypoints(path_file_left + str(frame_num) + '.txt'))
    keypointsR_sorted = np.array(getKeypoints(path_file_right + str(frame_num) + '.txt'))

    graph_circles(keypointsL, 0, 0, frameL)
    cv2.imshow('LEFT KP Light',frameL)

    key = cv2.waitKey(0)
    if key == ord('q'):
        # Close
        break

    list_points, list_np = point_cloud(keypointsL, keypointsR_sorted, baseline, f_px, center)

    # Crear un objeto PointCloud en Open3D
    point_cloud_view = o3d.geometry.PointCloud()
    # Listas para almacenar todos los puntos y colores
    all_points = []
    all_colors = []

    # Iterar sobre las listas de puntos y colores
    for i in range(len(list_np)):
        # Agregar los puntos y colores de la iteración actual a las listas
        all_points.extend(list_np[i])
        # Obtener el color RGB correspondiente al código de color
        color_rgb = color_map[i % len(lista_colores)]
        
        # Agregar el color correspondiente a cada punto de la iteración actual
        all_colors.extend([color_rgb] * len(list_np[i]))

    # Asignar las listas acumuladas al objeto PointCloud
    point_cloud_view.points = o3d.utility.Vector3dVector(all_points)
    point_cloud_view.colors = o3d.utility.Vector3dVector(all_colors)


    aabb = point_cloud_view.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    obb = point_cloud_view.get_oriented_bounding_box()
    obb.color = (0, 1, 0)
    # Visualizar la nube de puntos
    o3d.visualization.draw_geometries([point_cloud_view, aabb, obb])
    frame_num += 1
    print("Frame", frame_num)

capL.release()
capR.release()
cv2.destroyAllWindows()