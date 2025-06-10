from numpy.linalg import norm
import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from models.triangle import Triangle, createTriangle
from models.vector import MaterialCoord, Point, Vec3
from models.mesh import BVHNode, BoundingBox

ti.init()

@ti.kernel
def fill_triangle_buffer(
    v0s: ti.types.ndarray(),
    v1s: ti.types.ndarray(),
    v2s: ti.types.ndarray(),
    triangle_buffer: ti.template()
):
    for i in triangle_buffer:
        v0, v1, v2 = Point(v0s[i, 0], v0s[i, 1], v0s[i, 2]), Point(v1s[i, 0], v1s[i, 1], v1s[i, 2]), Point(v2s[i, 0], v2s[i, 1], v2s[i, 2])
        e1 = (v1-v0)
        e2 = (v2-v0)
        normal = ti.math.normalize(ti.math.cross(e1, e2))
        triangle_buffer[i] = Triangle(
            v0=v0,  # Assuming each point has x,y,z coordinates
            v1=v1,
            v2=v2,
            e1=e1,
            e2=e2,
            normal = normal,
            mat_coord=MaterialCoord(0, 0)
        )


def createTriangleBuffer(triangles_array, filename):
    triangles = triangles_array
    triangle_number = triangles.shape[0]
    triangleBuffer = Triangle.field(shape=triangle_number)

    # Create separate arrays for vertices
    v0s = np.array([t.v0 for t in triangles])  # This should be a Nx3 array
    v1s = np.array([t.v1 for t in triangles])
    v2s = np.array([t.v2 for t in triangles])
    # Fill buffer using parallel kernel
    fill_triangle_buffer(v0s, v1s, v2s, triangleBuffer)
    triangle_dict = triangleBuffer.to_numpy()
    #filename = triangles_file_path.split('/')[-1]
    np.save(f"meshes/{filename}", triangle_dict, allow_pickle=True)



@ti.kernel
def fill_node_buffer(
    node_buffer: ti.template(),
    bboxMins: ti.types.ndarray(),
    bboxMaxs: ti.types.ndarray(),
    triangleIndexes: ti.types.ndarray(),
    triangleCounts: ti.types.ndarray(),
    childIndexes: ti.types.ndarray()
):
    for i in node_buffer:
        bbox = BoundingBox(min=Vec3(bboxMins[i, 0], bboxMins[i, 1], bboxMins[i, 2]), max=Vec3(bboxMaxs[i, 0], bboxMaxs[i, 1], bboxMaxs[i, 2]))
        node_buffer[i] = BVHNode(boundingBox=bbox, childIndex=childIndexes[i], triangleIndex=triangleIndexes[i], triangleCount=triangleCounts[i])

def createNodeBuffer(nodes_array, filename):
    nodes = nodes_array
    node_number = nodes.shape[0]
    nodeBuffer = BVHNode.field(shape=node_number)
    bboxMins = np.array([node.boundingBox.min for node in nodes])
    bboxMaxs = np.array([node.boundingBox.max for node in nodes])
    triangleIndexes = np.array([node.triangleIndex for node in nodes])
    triangleCounts = np.array([node.triangleCount for node in nodes])
    childIndexes = np.array([node.childIndex for node in nodes])

    fill_node_buffer(nodeBuffer, bboxMins, bboxMaxs, triangleIndexes, triangleCounts, childIndexes)

    # for i in range(node_number):
    #     print(f"Nodes : {i+1}/{node_number}", end='\r')
    #     node = nodes[i]
    #     bound = BoundingBox(min=Vec3(node.boundingBox.min), max=Vec3(node.boundingBox.max))
    #     triangleIndex = node.triangleIndex
    #     triangleCount = node.triangleCount
    #     childIndex = node.childIndex
    #     nodeBuffer[i] = BVHNode(boundingBox=bound, childIndex=childIndex, triangleIndex=triangleIndex, triangleCount=triangleCount)
    # print("")
    #filename = nodes_file_path.split('/')[-1]
    node_dict = nodeBuffer.to_numpy()
    np.save(f"meshes/{filename}", node_dict, allow_pickle=True)

def createTriangleBuffer2(triangles_array, filename):
    triangles = triangles_array
    triangle_number = triangles.shape[0]

    triangleBuffer = Triangle.field(shape=triangle_number)

    for i in range(triangle_number):
        print(f"Triangles : {i+1}/{triangle_number}", end='\r')
        numpy_tri = triangles[i]
        triangleBuffer[i] = createTriangle(v0=Point(numpy_tri[0]), v1=Point(numpy_tri[1]), v2=Point(numpy_tri[2]), mat_coord=MaterialCoord(0, 0))

    triangle_dict = triangleBuffer.to_numpy()
    #filename = triangles_file_path.split('/')[-1]
    np.save(f"meshes/{filename}", triangle_dict, allow_pickle=True)

def createNodeBuffer2(nodes_array, filename):
    nodes = nodes_array
    node_number = nodes.shape[0]
    nodeBuffer = BVHNode.field(shape=node_number)
    for i in range(node_number):
        print(f"Nodes : {i+1}/{node_number}", end='\r')
        node = nodes[i]
        bound = BoundingBox(min=Vec3(node[0][0]), max=Vec3(node[0][1]))
        triangleIndex = node[1]
        triangleCount = node[2]
        childIndex = node[3]
        nodeBuffer[i] = BVHNode(boundingBox=bound, childIndex=childIndex, triangleIndex=triangleIndex, triangleCount=triangleCount)
    print("")
    #filename = nodes_file_path.split('/')[-1]
    node_dict = nodeBuffer.to_numpy()
    np.save(f"meshes/{filename}", node_dict, allow_pickle=True)

if __name__ == '__main__':
    triangles = np.load('jaguar_triangle.npy')
    nodes = np.load('jaguar_nodes.npy')
    createTriangleBuffer2(triangles, 'jaguar_triangle')
    createNodeBuffer2(nodes, 'jaguar2_nodes')
