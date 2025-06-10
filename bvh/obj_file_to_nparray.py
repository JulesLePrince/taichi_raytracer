import numpy as np
from bvh.classes import Triangle

def read_obj_file(file_path):
    vertices = []
    triangles = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                # Parse vertex
                _, x, y, z = line.split()
                vertices.append((float(x), float(y), float(z)))

            n = len(vertices)
            if line.startswith('f '):
                    a, b, c = parse_face_line(line)
                    v1 = a - (a > 1)
                    v2 = b - (b > 1)
                    v3 = c - (c > 1)
                    triangles.append([vertices[v1], vertices[v2], vertices[v3]])

    return np.array(vertices), np.array(triangles)

def parse_face_line(face_line):
    # Split the face line by spaces, ignoring the first 'f'
    vertices = face_line.split()[1:]

    parsed_faces = []

    # Loop through each vertex definition in the face line
    for vertex in vertices:
        # Split by the '/' to get vertex, texture, and normal indices
        v, _, _ = vertex.split('/')

        # Convert the indices to integers (note: OBJ uses 1-based indices)
        # Negative indices refer to the end of the list, which is why they can stay negative
        v = int(v)

        # Store the parsed indices in a tuple
        parsed_faces.append(v)

    return parsed_faces


def normalize_mesh(mesh, box_size=400):
    # Flatten the mesh to get all points
    points = mesh.reshape(-1, 3)

    # Find the current min and max values
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)

    # Calculate the current center and scale
    current_center = (min_vals + max_vals) / 2
    current_scale = np.max(max_vals - min_vals)

    # Calculate the scaling factor to fit in a box_size x box_size x box_size cube
    scale_factor = box_size / current_scale

    # Apply translation to center and scaling to fit in box_size x box_size x box_size cube
    normalized_points = (points - current_center) * scale_factor

    # Reshape back to original mesh shape
    normalized_mesh = normalized_points.reshape(mesh.shape)

    return normalized_mesh


def refactor_triangle_array(tri_array):
    triangles_refactor = np.full((tri_array.shape[0],), Triangle(v0=[0., 0., 0.], v1=[0., 0., 0.], v2=[0., 0., 0.]))

    for i in range(tri_array.shape[0]):
        v0 = tri_array[i][0]
        v1 = tri_array[i][1]
        v2 = tri_array[i][2]
        tri = Triangle(v0, v1, v2)
        triangles_refactor[i] = tri

    return triangles_refactor
