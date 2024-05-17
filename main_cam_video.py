from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from calculate import get_resolution
from utils.rectification import getStereoRectifier

# Create a new YOLO model from scratch (crear desde cero)
# model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
# model = YOLO('yolov8n.pt')
model = YOLO('yolov8x-pose-p6.pt')

lista_colores = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

figure = None
def setup_plot():
    global figure
    if figure is not None:
        return figure
    print("setup plot")
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
    print("live plot")
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
        [x1[0]], [x2[0]], baseline, f_px) for x1, x2 in zip(face_left, face_right)]

    x = (face_left[:, 0] - center_left[0]) * z / f_px
    y = (face_left[:, 1] - center_left[1]) * z / f_px

    return x, y, z

def point_cloud(left_kpts, right_kpts, baseline, f_px, center):
    print("point cloud")
    return live_plot_3d(left_kpts, right_kpts, baseline, f_px, center)


baseline = 58
f_px = 3098.3472392388794
center = (777.6339950561523,
            539.533634185791)


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



# Cámara
CONST_WIDTH_BOTH_LENS, CONST_HEIGHT = 1920*2, 1080

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Configurando la camara")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONST_WIDTH_BOTH_LENS)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONST_HEIGHT)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, 10)
print("Configuracin finalizada.")

# rectifier = getStereoRectifier("./calibration/re_calibration2/stereoMap.xml")
rectifier = getStereoRectifier("./calibration/integradora/newStereoMap.xml")

# while(capR.isOpened() and capL.isOpened()):
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame = cv2.rotate(frame, cv2.ROTATE_180)
    frameL = frame[:, :CONST_WIDTH_BOTH_LENS//2]
    frameR = frame[:, CONST_WIDTH_BOTH_LENS//2:]

    frame_left_calib, frame_right_calib = rectifier(frameL, frameR)

    print("frameL", frameL.shape, "frameR", frameR.shape)
    print("left calib", frame_left_calib.shape, "right calib", frame_right_calib.shape)
    cv2.imshow("left calib", frame_left_calib)
    cv2.imshow("right calib", frame_right_calib)
    # cv2.imshow('frame', frame)
    """
    tmp_frame = frame.copy()
    # frame_combined = frameL/2 + frameR/2
    # frame_combined = frame_combined.astype(np.uint8)
    # cv2.imshow("comnb", frame_combined)
    h = frameR.shape[0]
    w = frameR.shape[1]

    if(not ret or not retR):
        print("Failed to read frames")
    
    # Predict
    resultL = model.predict(frameL, imgsz=(1920*4,1080*4),stream_buffer=True, conf=0.5)
    resultR = model.predict(frameR, imgsz=(1920*4,1080*4),stream_buffer=True, conf=0.5)

    keypointsL = np.array(resultL[0].keypoints.xy.cpu())
    keypointsR = np.array(resultR[0].keypoints.xy.cpu())
    
    print("LEN", len(keypointsL))

    if (len(keypointsL)<=1 or len(keypointsR)<=1):continue
    # keypointsL = list(keypointsL).sort(key=lambda kpts:np.mean(kpts[:,0]))
    # keypointsR = list(keypointsR).sort(key=lambda kpts:np.mean(kpts[:,0]))

    # keypointsL = np.array(keypointsL)
    # keypointsR = np.array(keypointsR)

    # print(result[0].keypoints.xy)
    # print(result[0].keypoints.xy.shape)
    try:
        graph_circles(keypointsL, w, h, frameL)
        graph_circles(keypointsR, w, h, frameR)
    except:
        print("Error")
        continue

    cv2.imshow('LEFT',frameL)
    cv2.imshow('RIGHT',frameR)
    """
    key = cv2.waitKey(1)
    if key == ord('q'):
        # Close
        break


    """
    # pointCloud(keypointsL,keypointsR)
    #keypointsL_sorted = sorted()
    keypointsR_sorted = [keypointsR[index] for index in get_resolution(keypointsL, keypointsR).values()]
    print("sorted")
    lists_points_3d = point_cloud(keypointsL, keypointsR_sorted, baseline, f_px, center)
    """


cap.release()
cv2.destroyAllWindows()
