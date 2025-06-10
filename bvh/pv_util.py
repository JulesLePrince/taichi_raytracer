import pyvista as pv
import numpy as np
from bvh.classes import BVHNode

def draw_box(plotter, box_min, box_max, color='red', opacity=1.0, style='wireframe'):
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


def draw_boxes(plotter, boxes, color='red', opacity=1.0, style='wireframe'):
    """
    Draw multiple boxes using PyVista's box creation.

    Parameters:
    - plotter: PyVista plotter instance
    - boxes: List of dictionaries with 'box_min' and 'box_max' for each box.
    - color: color of the boxes
    - opacity: transparency (0.0 to 1.0)
    - style: 'wireframe' or 'surface'
    """
    # List to hold each individual box mesh
    box_meshes = []

    # Loop through each box and create a PyVista Box
    for box in boxes:
        box_min = box['box_min']
        box_max = box['box_max']

        # Create the box using PyVista's Box method
        box_mesh = pv.Box(bounds=(box_min[0], box_max[0],
                                  box_min[1], box_max[1],
                                  box_min[2], box_max[2]))

        box_meshes.append(box_mesh)

    # Combine all boxes into a single mesh for efficient plotting
    combined_mesh = pv.MultiBlock(box_meshes).combine()

    # Add the combined mesh to the plotter
    if style == 'wireframe':
        plotter.add_mesh(combined_mesh, style='wireframe', line_width=1, color=color, opacity=opacity)
    else:
        plotter.add_mesh(combined_mesh, color=color, opacity=opacity)



def draw_point(plotter, point, color='red', point_size=20, label=None):
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

def draw_line(plotter, start_point, end_point, color='red', width=5, label=None):
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

def plot_triangles(plotter, triangles, highlight_index=None):
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

def draw_BVH(plotter, allNodes):
    for node in allNodes:
        if node.childIndex == 0:
            box_min = node.boundingBox.min
            box_max = node.boundingBox.max
            draw_box(plotter, box_min, box_max)

def split_into_chunks(array, chunk_size):
    # Split array into chunks of the specified chunk_size
    return [array[i:i + chunk_size] for i in range(0, len(array), chunk_size)]

def draw_BVH_efficient(plotter, allNodes):
    boxes = []
    for node in allNodes:
        if node.childIndex == 0:
            box_min = node.boundingBox.min
            box_max = node.boundingBox.max
            boxes.append({'box_min':box_min, 'box_max':box_max})
    draw_boxes(plotter, boxes)
