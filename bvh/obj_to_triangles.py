import numpy as np
import pyvista as pv
import plotly.graph_objects as go

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

# Example usage
obj_file_path = '/Users/julesleprince/Downloads/model-8/model.obj'
vertices, triangles = read_obj_file(obj_file_path)
normalized_triangles = normalize_mesh(triangles)


####################################### BVH

class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = np.array(direction)


class Triangle:
    def __init__(self, v0, v1, v2):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2

    @property
    def center(self):
        cx = (self.v0[0] + self.v1[0] + self.v2[0]) / 3
        cy = (self.v0[1] + self.v1[1] + self.v2[1]) / 3
        cz = (self.v0[2] + self.v1[2] + self.v2[2]) / 3

        return np.array([cx, cy, cz])

    def hit(self, ray):
        v0 = self.v0
        v1 = self.v1
        v2 = self.v2
        e1 = v1 - v0
        e2 = v2 - v0
        T = ray.origin - v0
        D = ray.direction
        t = 0

        # Precompute cross products
        P = np.cross(D, e2)
        det = np.dot(P, e1)

        # Early exit for determinant close to zero
        if np.abs(det) > 0.0001:
            inv_det = 1.0 / det

            u = np.dot(P, T) * inv_det

            # Early exit if u is out of range
            if u >= 0 and u <= 1:
                Q = np.cross(T, e1)
                v = np.dot(Q, D) * inv_det

                # Early exit if v is out of range
                if v >= 0 and (u + v) <= 1:
                    t = np.dot(Q, e2) * inv_det

                    if t > 0.0001:
                        return True, t

        return False, t


normalized_triangles_refactor = np.full((normalized_triangles.shape[0],), Triangle(v0=[0., 0., 0.], v1=[0., 0., 0.], v2=[0., 0., 0.]))

for i in range(normalized_triangles.shape[0]):
    v0 = normalized_triangles[i][0]
    v1 = normalized_triangles[i][1]
    v2 = normalized_triangles[i][2]
    tri = Triangle(v0, v1, v2)
    normalized_triangles_refactor[i] = tri


class BoundingBox:
    def __init__(self):
        neg_inf = float('-inf')
        inf = float('inf')
        self.min= np.array([inf, inf, inf])
        self.max = np.array([neg_inf, neg_inf, neg_inf])

    def hit(self, ray):
        """
        Check if the ray intersects with the bounding box.
        Returns True if there's an intersection, False otherwise.
        """
        # Small value to prevent division by zero
        epsilon = 1e-10

        t_min = float('-inf')
        t_max = float('inf')

        for i in range(3):
            # Handle case where ray direction component is close to 0
            if abs(ray.direction[i]) < epsilon:
                if ray.origin[i] < self.min[i] or ray.origin[i] > self.max[i]:
                    return False
            else:
                # Calculate intersection distances
                t1 = (self.min[i] - ray.origin[i]) / ray.direction[i]
                t2 = (self.max[i] - ray.origin[i]) / ray.direction[i]

                # Ensure t1 is the smaller value
                if t1 > t2:
                    t1, t2 = t2, t1

                t_min = max(t_min, t1)
                t_max = min(t_max, t2)

                if t_min > t_max:
                    return False

        return True


    @property
    def center(self):
        return (self.min + self.max) * 0.5

    @property
    def size(self):
        return self.max - self.min

    def growToIncludePoint(self, point):
        self.min = np.minimum(self.min, point)
        self.max = np.maximum(self.max, point)

    def growToIncludeTriangle(self, triangle):
        self.growToIncludePoint(triangle.v0)
        self.growToIncludePoint(triangle.v1)
        self.growToIncludePoint(triangle.v2)

class BVHNode:
    def __init__(self):
        self.boundingBox = BoundingBox()
        self.triangleIndex = 0
        self.triangleCount = 0
        self.childIndex = 0



allNodes = np.array([])
allTriangles = normalized_triangles_refactor

def createBVH():
    global allNodes
    global allTriangles
    root = BVHNode()
    root.triangleCount = allTriangles.shape[0]
    allNodes = np.append(allNodes, [root], axis=0)
    for triangle in allTriangles:
        root.boundingBox.growToIncludeTriangle(triangle)
    split(root, 0)
    return root


def split(parent, depth):
    global allNodes
    global allTriangles

    if (depth >= 7):
        return

    size = parent.boundingBox.size
    if size[0] > max(size[1], size[2]):
        splitAxis = 0
    elif size[1] > size[2]:
        splitAxis = 1
    else:
        splitAxis = 2
    splitPos = parent.boundingBox.center[splitAxis]


    parent.childIndex = allNodes.shape[0]
    allNodes = np.append(allNodes, [BVHNode(), BVHNode()], axis=0)
    allNodes[parent.childIndex].triangleIndex = parent.triangleIndex
    allNodes[parent.childIndex+1].triangleIndex = parent.triangleIndex

    for i in range(parent.triangleIndex, parent.triangleIndex+parent.triangleCount):
        isSideA = allTriangles[i].center[splitAxis] < splitPos
        if isSideA:
            allNodes[parent.childIndex].boundingBox.growToIncludeTriangle(allTriangles[i])
            allNodes[parent.childIndex].triangleCount += 1
            swap = allNodes[parent.childIndex].triangleIndex + allNodes[parent.childIndex].triangleCount - 1
            allTriangles[i], allTriangles[swap] = allTriangles[swap], allTriangles[i]
            allNodes[parent.childIndex+1].triangleIndex += 1
        else:
            allNodes[parent.childIndex+1].boundingBox.growToIncludeTriangle(allTriangles[i])
            allNodes[parent.childIndex+1].triangleCount += 1


    if allNodes[parent.childIndex].triangleCount >= 10:
        split(allNodes[parent.childIndex], depth + 1)
    if allNodes[parent.childIndex+1].triangleCount >= 10:
        split(allNodes[parent.childIndex+1], depth + 1)


BVHroot = createBVH()

ray = Ray(origin=np.array([-50, 0., 200.]), direction=np.array([0, -0.5, -1.]))
ray_end = ray.origin + 400*ray.direction

def triangle_intersection(ray, tri):
    v0 = tri[0]
    v1 = tri[1]
    v2 = tri[2]
    e1 = v1 - v0
    e2 = v2 - v0
    T = ray.origin - v0
    D = ray.direction
    t = 0

    # Precompute cross products
    P = np.cross(D, e2)
    det = np.dot(P, e1)

    # Early exit for determinant close to zero
    if np.abs(det) > 0.0001:
        inv_det = 1.0 / det

        u = np.dot(P, T) * inv_det

        # Early exit if u is out of range
        if u >= 0 and u <= 1:
            Q = np.cross(T, e1)
            v = np.dot(Q, D) * inv_det

            # Early exit if v is out of range
            if v >= 0 and (u + v) <= 1:
                t = np.dot(Q, e2) * inv_det

                if t > 0.0001:
                    return True, t

    return False, t

def dumb_intersection(ray):
    global allTriangles
    k = -1
    t_min = float('inf')
    for i in range(allTriangles.shape[0]):
        intersect, t = allTriangles[i].hit(ray)
        if intersect and t < t_min:
            k = i
            t_min = t
    return k, t_min

def BVH_intersection(ray):
    global allNodes
    global allTriangles
    nodeStack = [allNodes[0]]

    k = -1
    t_min = float('inf')

    while len(nodeStack) > 0:
        node = nodeStack.pop()
        if node.childIndex == 0:
            for i in range(node.triangleIndex, node.triangleCount+node.triangleIndex):
                didHit, t = allTriangles[i].hit(ray)
                if didHit and t < t_min:
                    k = i
                    t_min = t
        else:
            nodeStack.append(allNodes[node.childIndex])
            nodeStack.append(allNodes[node.childIndex+1])

    return k, t_min


k,t_min = BVH_intersection(ray)



def draw_box(box_min, box_max, color='red', opacity=1.0, style='wireframe'):
    """
    Draw a box defined by its minimum and maximum corners.

    Parameters:
    - plotter: PyVista plotter instance
    - box_min: np.array([x_min, y_min, z_min])
    - box_max: np.array([x_max, y_max, z_max])
    - color: color of the box
    - opacity: transparency (0.0 to 1.0)
    - style: 'wireframe' or 'surface'
    """
    # Create the 8 vertices of the box
    vertices = np.array([
        [box_min[0], box_min[1], box_min[2]],  # 0
        [box_max[0], box_min[1], box_min[2]],  # 1
        [box_max[0], box_max[1], box_min[2]],  # 2
        [box_min[0], box_max[1], box_min[2]],  # 3
        [box_min[0], box_min[1], box_max[2]],  # 4
        [box_max[0], box_min[1], box_max[2]],  # 5
        [box_max[0], box_max[1], box_max[2]],  # 6
        [box_min[0], box_max[1], box_max[2]]   # 7
    ])

    # Define the faces of the box
    faces = np.array([
        # Bottom face
        [4, 0, 1, 2, 3],
        # Top face
        [4, 4, 5, 6, 7],
        # Front face
        [4, 0, 1, 5, 4],
        # Right face
        [4, 1, 2, 6, 5],
        # Back face
        [4, 2, 3, 7, 6],
        # Left face
        [4, 3, 0, 4, 7]
    ]).flatten()

    # Create the box mesh
    box_mesh = pv.PolyData(vertices, faces)

    # Add the box to the plotter
    plotter.add_mesh(box_mesh,
                    color=color,
                    opacity=opacity,
                    style=style,
                    line_width=2)

def draw_box_plotly(box_min, box_max, color='red', opacity=1.0, style='wireframe'):
    """
    Draw a box defined by its minimum and maximum corners in Plotly.

    Parameters:
    - fig: Plotly figure instance
    - box_min: np.array([x_min, y_min, z_min])
    - box_max: np.array([x_max, y_max, z_max])
    - color: color of the box
    - opacity: transparency (0.0 to 1.0)
    - style: 'wireframe' or 'surface'
    """
    # Define the vertices of the box
    vertices = np.array([
        [box_min[0], box_min[1], box_min[2]],  # 0
        [box_max[0], box_min[1], box_min[2]],  # 1
        [box_max[0], box_max[1], box_min[2]],  # 2
        [box_min[0], box_max[1], box_min[2]],  # 3
        [box_min[0], box_min[1], box_max[2]],  # 4
        [box_max[0], box_min[1], box_max[2]],  # 5
        [box_max[0], box_max[1], box_max[2]],  # 6
        [box_min[0], box_max[1], box_max[2]]   # 7
    ])

    # Define the edges of the box for wireframe
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]

    # Plotting the box as a surface
    if style == 'surface':
        i_faces = [0, 0, 0, 1, 2, 3]
        j_faces = [1, 2, 3, 2, 3, 0]
        k_faces = [3, 3, 1, 5, 6, 7]

        fig.add_trace(
            go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=i_faces, j=j_faces, k=k_faces,
                color=color,
                opacity=opacity,
                flatshading=True
            )
        )
    elif style == 'wireframe':
        # Plotting the box as a wireframe
        for edge in edges:
            fig.add_trace(
                go.Scatter3d(
                    x=[vertices[edge[0], 0], vertices[edge[1], 0]],
                    y=[vertices[edge[0], 1], vertices[edge[1], 1]],
                    z=[vertices[edge[0], 2], vertices[edge[1], 2]],
                    mode='lines',
                    line=dict(color=color, width=2)
                )
            )

def draw_point(point, color='red', point_size=20, label=None):
    """
    Draw a single point in 3D space.

    Parameters:
    - point: np.array([x, y, z])
    - color: color of the point
    - point_size: size of the point
    - label: optional text label for the point
    """
    # Create a point cloud with a single point
    point_cloud = pv.PolyData(np.array([point]))

    # Add the point to the plotter
    plotter.add_mesh(point_cloud,
                    color=color,
                    point_size=point_size,
                    render_points_as_spheres=True)

    # Add label if specified
    if label:
        plotter.add_point_labels(point_cloud, [label])

def draw_point_plotly(point, color='red', point_size=3, label=None):
    fig.add_trace(
        go.Scatter3d(
            x=[point[0]], y=[point[1]], z=[point[2]],
            mode='markers+text' if label else 'markers',
            marker=dict(size=point_size, color=color),
            text=label,
            textposition="top center"
        )
    )

def draw_line(start_point, end_point, color='red', width=5, label=None):
    """
    Draw a line between two points in 3D space.

    Parameters:
    - plotter: PyVista plotter instance
    - start_point: np.array([x, y, z])
    - end_point: np.array([x, y, z])
    - color: color of the line
    - width: line width
    - label: optional text label for the line
    """
    # Create line points
    points = np.array([start_point, end_point])

    # Create line
    line = pv.Line(start_point, end_point)

    # Add the line to the plotter
    plotter.add_mesh(line,
                    color=color,
                    line_width=width,
                    render_lines_as_tubes=True)

    # Add label if specified
    if label:
        # Place label at midpoint of line
        mid_point = (start_point + end_point) / 2
        plotter.add_point_labels(np.array([mid_point]), [label])

def draw_line_plotly(start_point, end_point, color='red', width=5, label=None):
    """
    Draw a line between two points in 3D space using Plotly.

    Parameters:
    - fig: Plotly figure instance
    - start_point: np.array([x, y, z]) - start point of the line
    - end_point: np.array([x, y, z]) - end point of the line
    - color: color of the line
    - width: line width
    - label: optional text label for the line
    """
    # Extract coordinates for the line
    x_line = [start_point[0], end_point[0]]
    y_line = [start_point[1], end_point[1]]
    z_line = [start_point[2], end_point[2]]

    # Add the line as a Scatter3d trace
    fig.add_trace(
        go.Scatter3d(
            x=x_line, y=y_line, z=z_line,
            mode='lines+text' if label else 'lines',
            line=dict(color=color, width=width),
            text=[label] if label else None,
            textposition="middle center" if label else None,
            hoverinfo='none'
        )
    )

    # Add label at midpoint if specified
    if label:
        mid_point = (start_point + end_point) / 2
        fig.add_trace(
            go.Scatter3d(
                x=[mid_point[0]], y=[mid_point[1]], z=[mid_point[2]],
                mode='text',
                text=[label],
                textposition="top center"
            )
        )

def draw_BVH(allNodes):
    for node in allNodes:
        if node.childIndex == 0:
            box_min = node.boundingBox.min
            box_max = node.boundingBox.max
            draw_box(box_min, box_max)
            draw_box_plotly(box_min, box_max)




def plot_triangles(triangles, highlight_index=None):
    # Convert triangle array to vertices and faces arrays
    vertices = []
    faces = []

    # Collect all unique vertices and create faces indices
    for i, tri in enumerate(triangles):
        # Add vertices
        vertices.extend([tri.v0, tri.v1, tri.v2])
        # Add face (3 indicates it's a triangle, followed by the 3 vertex indices)
        faces.extend([3, 3*i, 3*i+1, 3*i+2])

    # Convert to numpy arrays
    vertices = np.array(vertices)
    faces = np.array(faces)

    # Create mesh
    mesh = pv.PolyData(vertices, faces)

    # Create colors array (default color for all triangles)
    colors = np.ones((len(triangles), 3)) * [0.623, 0.80, 0.84]  # Default gray color

    # Set color for specific triangle if highlight_index is provided
    if highlight_index is not None and highlight_index >= 0:
        colors[highlight_index] = [1, 0, 0]  # Red color for highlighted triangle

    # Add colors to mesh
    mesh.cell_data['colors'] = colors

    # Create plotter
    plotter.add_mesh(mesh, scalars='colors', opacity=0.8, rgb=True)

def plot_triangles_plotly(triangles, highlight_index=None):
    # Convert triangle array to vertices and faces arrays
    vertices = []
    i_faces = []
    j_faces = []
    k_faces = []

    # Collect all unique vertices and create faces indices
    for i, tri in enumerate(triangles):
        # Add vertices
        vertices.extend([tri.v0, tri.v1, tri.v2])
        i_faces.append(3 * i)
        j_faces.append(3 * i + 1)
        k_faces.append(3 * i + 2)

    # Convert to numpy array for easier manipulation
    vertices = np.array(vertices)
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    # Create colors array (default color for all triangles)
    face_colors = ['rgb(159, 204, 214)'] * len(triangles)  # Default gray color

    # Set color for specific triangle if highlight_index is provided
    if highlight_index is not None and highlight_index >= 0:
        face_colors[highlight_index] = 'rgb(255, 0, 0)'  # Red color for highlighted triangle

    # Add mesh to the existing figure
    fig.add_trace(
        go.Mesh3d(
            x=x, y=y, z=z,
            i=i_faces, j=j_faces, k=k_faces,
            facecolor=face_colors,
            opacity=0.8,
            flatshading=True
        )
    )

plotter = pv.Plotter()
fig = go.Figure()

# Define the corner points of the box (min and max values in 3D)
#draw_BVH(allNodes)
plot_triangles(allTriangles, k)
plot_triangles_plotly(allTriangles, k)

"""draw_point(point=ray.origin)
draw_point_plotly(point=ray.origin)

draw_line(ray.origin, ray.origin+500*ray.direction, 'green', 2)
draw_line_plotly(ray.origin, ray.origin+500*ray.direction, 'green', 2)"""


if k >= 0:
    draw_point(ray.origin+t_min*ray.direction, 'blue', 10)

#plotter.show()
# Update layout and show the figure
fig.update_layout(scene=dict(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    zaxis=dict(visible=False),
    aspectmode='data'
))
fig.show()
