import open3d as o3d
import numpy as np

print("Testing IO for meshes ...")
mesh = o3d.io.read_point_cloud("./data/output.ply")
print(mesh)

# Obtener las coordenadas (x, y, z) de los puntos
points = np.asarray(mesh.points)
print("Puntos:", points)
print("Forma de los puntos:", points.shape)

# Filtrar los valores del tercer índice mayores a 4 y menores a 5
filtered_points = points[(points[:, 2] > 4) & (points[:, 2] < 5)]

# Obtener las coordenadas filtradas en x, y, z
filtered_x = filtered_points[:, 0]
filtered_y = filtered_points[:, 1]
filtered_z = filtered_points[:, 2]

"""
# Ahora puedes utilizar las coordenadas filtradas (filtered_x, filtered_y, filtered_z) como desees
print("Coordenadas filtradas x:", filtered_x)
print("Coordenadas filtradas y:", filtered_y)
print("Coordenadas filtradas z:", filtered_z)
"""

# Separar las coordenadas en x, y, z
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

# Ahora puedes utilizar las coordenadas (x, y, z) como desees
print("Coordenadas x:", x)
print("Coordenadas y:", y)
print("Coordenadas z:", z)

# o3d.visualization.draw_geometries([mesh])


# Crear una nube de puntos a partir de las coordenadas (x, y, z)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(np.column_stack((filtered_x, filtered_y, filtered_z)))

# Visualizar la nube de puntos
o3d.visualization.draw_geometries([point_cloud])