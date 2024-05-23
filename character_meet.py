import numpy as np
import cv2

# Normalizar y escalar puntos
def normalize_and_scale(point, min_x, max_x, min_y, max_y, width, height):
    x = (point[0] - min_x) / (max_x - min_x) * width
    y = (point[1] - min_y) / (max_y - min_y) * height
    return int(x), int(height - y)  # Invertir y para que el origen esté en la esquina inferior izquierda

def get_img_shape_meet(list_centroides):
    image_back = np.zeros((360, 640, 3), dtype=np.uint8)

    # unir los puntos con una línea
    print("Centroides", len(list_centroides))
    list_xz = []
    for i in list_centroides:
        list_xz.append((i[0], i[2]))
    
    # Encontrar los mínimos y máximos de las coordenadas x y y
    min_x = min(point[0] for point in list_xz)
    max_x = max(point[0] for point in list_xz)
    min_y = min(point[1] for point in list_xz)
    max_y = max(point[1] for point in list_xz)

    # Convertir los puntos a coordenadas de imagen
    scaled_points = [normalize_and_scale(point, min_x, max_x, min_y, max_y, 640, 360) for point in list_xz]

    # Dibujar líneas entre los puntos
    for i in range(len(scaled_points) - 1):
        cv2.line(image_back, scaled_points[i], scaled_points[i + 1], (255, 0, 0), 2)

    
    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(image_back, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow("Grayscale Image", gray_image)
    cv2.waitKey(0)
    
    return gray_image