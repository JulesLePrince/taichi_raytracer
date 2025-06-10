import numpy as np

def saveMeshAndBVH(allTriangles, allNodes, name):
    triangle_number = allTriangles.shape[0]
    triangle_refactor = triangles = np.zeros((triangle_number, 3, 3))
    for i in range(triangle_number):
        triangle_refactor[i] = [allTriangles[i].v0, allTriangles[i].v1, allTriangles[i].v2]

    dt = np.dtype([
        ('boundingBox', [('min', float, 3), ('max', float, 3)]),  # specify shape (3,) for 3D vectors
        ('triangleIndex', int),
        ('triangleCount', int),
        ('childIndex', int)
    ])

    node_number = allNodes.shape[0]
    node_refactor = np.zeros(node_number, dtype=dt)

    for i in range(node_number):
        node = allNodes[i]
        node_refactor[i]['boundingBox']['min'] = node.boundingBox.min  # [0., 0., 0.]
        node_refactor[i]['boundingBox']['max'] = node.boundingBox.max  # also a 3D vector
        node_refactor[i]['triangleIndex'] = node.triangleIndex
        node_refactor[i]['triangleCount'] = node.triangleCount
        node_refactor[i]['childIndex'] = node.childIndex

    np.save(f"{name}_triangle.npy", triangle_refactor)
    np.save(f"{name}_nodes.npy", node_refactor)
