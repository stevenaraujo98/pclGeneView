import cv2
import numpy as np
import matplotlib.pyplot as plt
from getKeypointsFile import getKeypoints

lista_colores = ['r', 'g', 'b', 'c', 'm', 'y', 'k']


figure = None
def setup_plot():
    global figure
    if figure is not None:
        return figure

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_ylim(-1000, 1000)
    ax.set_xlim(-2000, 2000)
    ax.set_zlim(-3000, 3000)
    figure = fig, ax
    return fig, ax

def plot_3d(x, y, z, ax, color, s=None, marker="o"):
    if (s):
        ax.scatter(x, y, z, color=color, marker=marker, s=s)
    else:
        ax.scatter(x, y, z, color=color, marker=marker)

def clean_plot(ax):
    ax.cla()
    ax.set_ylim(-1000, 2000)
    ax.set_xlim(-3000, 3000)
    ax.set_zlim(0, 10000)

def live_plot_3d(left_kpts, right_kpts, baseline, f_px, center):
    fig, ax = setup_plot()
    clean_plot(ax)
    list_points = []
    list_color_to_paint = []

    # Agregar a una lista de colores para pintar los puntos de cada persona en caso de ser mas de len(lista_colores)
    for i in range(len(left_kpts)):
        indice_color = i % len(lista_colores)
        list_color_to_paint.append(lista_colores[indice_color])

    for left_k, right_k, color in zip(left_kpts, right_kpts, list_color_to_paint):
        points = body_3d(left_k, right_k, baseline, f_px, center)

        # Filtro de puntos menores a 1000 y mayores a 10000 
        filtered_points = [[a, b, c] for a, b, c in zip(*points) if 1000 < c <= 10000]
        for point in filtered_points:
            plot_3d(point[0], point[1], point[2], ax, color)
        list_points.append(filtered_points)

    for person, color in zip(list_points, list_color_to_paint):
        # Puede ocasionar que person no pase el filtro por lo que se debe validar
        if len(person) == 0:
            continue
        centroide = calcular_centroide(person)
        plot_3d(centroide[0], centroide[1], centroide[2], ax, color, s=100, marker='o')
    
    plt.draw()
    plt.pause(0.0001)
    return list_points

def find_depth_from_disparities(right_points, left_points, baseline, f_pixel):
    x_right = np.array(right_points)
    x_left = np.array(left_points)
    disparity = np.abs(x_left - x_right)
    z_depth = (baseline * f_pixel) / disparity
    return np.mean(z_depth)

def body_3d(face_left, face_right, baseline, f_px, center_left):
    assert len(face_left) == len(face_right)

    z = [find_depth_from_disparities(
            [x1[0]], 
            [x2[0]], 
            baseline, 
            f_px
        ) for x1, x2 in zip(face_left, face_right)]

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
# baseline = 58
baseline = 29.659969789679785
# f_px = 3098.3472392388794
f_px = (1430.012149825778 + 1430.9237520924735 + 1434.1823778243315 + 1435.2411841383973) / 4 
center = ((929.8117736131715 + 936.7865692255547) / 2,
          (506.4104424162777 + 520.0461674300153) / 2)

# center = (777.6339950561523,539.533634185791)

list_colors = [(255,0,255), (0, 255, 255), (255, 0, 0), (0, 0, 0), (255, 255, 0), (205, 92, 92), (255, 0, 255), (0, 128, 128), (128, 0, 0), (128, 128, 0), (128, 128, 128)]

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

# YOLO ------------------------------------------------------------------------------------------------------------
frameL=cv2.imread('./predicts/YOLO/rosmasterx3plus/200/integradora/14_04_56_13_05_2024_IMG_LEFT.jpg')
path_file_left = './predicts/YOLO/rosmasterx3plus/200/integradora/14_04_56_13_05_2024_IMG_LEFT/14_04_56_13_05_2024_IMG_LEFT'
frameR=cv2.imread('./predicts/YOLO/rosmasterx3plus/200/integradora/14_04_56_13_05_2024_IMG_RIGHT.jpg')
path_file_right = './predicts/YOLO/rosmasterx3plus/200/integradora/14_04_56_13_05_2024_IMG_RIGHT/14_04_56_13_05_2024_IMG_RIGHT'

# frameL=cv2.imread('./predicts/YOLO/rosmasterx3plus/200/re_calibration2/14_04_56_13_05_2024_IMG_LEFT.jpg')
# path_file_left = './predicts/YOLO/rosmasterx3plus/200/re_calibration2/14_04_56_13_05_2024_IMG_LEFT/14_04_56_13_05_2024_IMG_LEFT'
# frameR=cv2.imread('./predicts/YOLO/rosmasterx3plus/200/re_calibration2/14_04_56_13_05_2024_IMG_RIGHT.jpg')
# path_file_right = './predicts/YOLO/rosmasterx3plus/200/re_calibration2/14_04_56_13_05_2024_IMG_RIGHT/14_04_56_13_05_2024_IMG_RIGHT'

# LightGlue ------------------------------------------------------------------------------------------------------------
# frameL=cv2.imread('./predicts/LightGlue/rosmasterx3plus/200/integradora/14_04_56_13_05_2024_IMG_LEFT.jpg')
# path_file_left = './predicts/LightGlue/rosmasterx3plus/200/integradora/14_04_56_13_05_2024_IMG_LEFT/14_04_56_13_05_2024_IMG_LEFT'
# frameR=cv2.imread('./predicts/LightGlue/rosmasterx3plus/200/integradora/14_04_56_13_05_2024_IMG_RIGHT.jpg')
# path_file_right = './predicts/LightGlue/rosmasterx3plus/200/integradora/14_04_56_13_05_2024_IMG_RIGHT/14_04_56_13_05_2024_IMG_RIGHT'

# frameL=cv2.imread('./predicts/LightGlue/rosmasterx3plus/200/re_calibration2/14_04_56_13_05_2024_IMG_LEFT.jpg')
# path_file_left = './predicts/LightGlue/rosmasterx3plus/200/re_calibration2/14_04_56_13_05_2024_IMG_LEFT/14_04_56_13_05_2024_IMG_LEFT'
# frameR=cv2.imread('./predicts/LightGlue/rosmasterx3plus/200/re_calibration2/14_04_56_13_05_2024_IMG_RIGHT.jpg')
# path_file_right = './predicts/LightGlue/rosmasterx3plus/200/re_calibration2/14_04_56_13_05_2024_IMG_RIGHT/14_04_56_13_05_2024_IMG_RIGHT'


h = frameR.shape[0]
w = frameR.shape[1]

cv2.imshow('LEFT',frameL)
cv2.imshow('RIGHT',frameR)

keypointsL = np.array(getKeypoints(path_file_left + '_1.txt'))
keypointsR_sorted = np.array(getKeypoints(path_file_right + '_1.txt'))

graph_circles(keypointsL, 0, 0, frameL)
# cv2.imshow('LEFT KP',frameL)

# keypointsL_copy = keypointsL.copy()
# keypointsL_copy[:, :, 0] += 8.5
# lists_points_3d = point_cloud(keypointsL, keypointsL_copy, baseline, f_px, center)

lists_points_3d = point_cloud(keypointsL, keypointsR_sorted, baseline, f_px, center)


key = cv2.waitKey(0)
cv2.destroyAllWindows()
