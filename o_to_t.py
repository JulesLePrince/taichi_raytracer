import numpy as np
from bvh.obj_file_to_nparray import read_obj_file, normalize_mesh, refactor_triangle_array
from bvh.classes import BVHNode, Triangle, Ray, BoundingBox
import bvh.pv_util as pv_util
import pyvista as pv

from bvh.save_to_numpy import saveMeshAndBVH
from tqdm import tqdm


obj_file_path = '/Users/julesleprince/Downloads/easter/easter.obj'
name = "easter"

vertices, triangles = read_obj_file(obj_file_path)
normalized_triangles = normalize_mesh(triangles)
allTriangles = refactor_triangle_array(normalized_triangles)

MAX_DEPTH = 32
emptyNode = BVHNode()
nodeIndex = 0
allNodes = np.repeat(emptyNode, len(allTriangles)*2)




def bestSplit(BoundingBox):
    split_axis = np.argmax(BoundingBox.size)
    split_pos = BoundingBox.center[split_axis]
    return split_axis, split_pos


def evaluate(node, splitAxis, splitPos):
    global allTriangles

    boundsA = BoundingBox()
    boundsB = BoundingBox()
    numInA = 0
    numInB = 0

    for i in range(node.triangleIndex, node.triangleIndex+node.triangleCount):
        tri = allTriangles[i]
        if tri.center[splitAxis] < splitPos:
            boundsA.growToIncludeTriangle(tri)
            numInA += 1
        else:
            boundsB.growToIncludeTriangle(tri)
            numInB += 1

    cost = numInA*(boundsA.surfaceArea) + numInB*(boundsB.surfaceArea)
    return cost

def bestSplitSAH(node):
    numTestPerAxis = 5
    bestCost = float("inf")
    bestPos = 0
    bestAxis = 0

    for axis in range(3):
        boundStart = node.boundingBox.min[axis]
        boundEnd = node.boundingBox.max[axis]
        for i in range(numTestPerAxis):
            splitT = (i+1)/(numTestPerAxis+1)
            pos = boundStart + (boundEnd - boundStart)*splitT
            cost = evaluate(node, axis, pos)
            if cost < bestCost:
                bestCost = cost
                bestPos = pos
                bestAxis = axis
    return bestAxis, bestPos, bestCost

def splitSAH(parent, depth, progress_bar=None):
    global allNodes, allTriangles, nodeIndex

    if depth >= MAX_DEPTH or parent.triangleCount < 2:
        return

    parentCost = parent.triangleCount*(parent.boundingBox.surfaceArea)
    splitAxis, splitPos, cost = bestSplitSAH(parent)

    if cost >= parentCost:
        return


    parent.childIndex = nodeIndex
    childA = BVHNode()
    childB = BVHNode()
    childA.triangleIndex = parent.triangleIndex
    childB.triangleIndex = parent.triangleIndex

    for i in range(parent.triangleIndex, parent.triangleIndex + parent.triangleCount):
        isSideA = allTriangles[i].center[splitAxis] < splitPos
        if isSideA:
            childA.boundingBox.growToIncludeTriangle(allTriangles[i])
            childA.triangleCount += 1
            swap = childA.triangleIndex + childA.triangleCount - 1

            allTriangles[i], allTriangles[swap] = allTriangles[swap], allTriangles[i]
            childB.triangleIndex += 1
        else:
            childB.boundingBox.growToIncludeTriangle(allTriangles[i])
            childB.triangleCount += 1

    allNodes[nodeIndex] = childA
    nodeIndex += 1
    allNodes[nodeIndex] = childB
    nodeIndex += 1


    if progress_bar:
           progress_bar.update(parent.triangleCount)

    splitSAH(childA, depth+1, progress_bar)
    splitSAH(childB, depth+1, progress_bar)




def split(parent, depth, progress_bar=None):
    global allNodes, allTriangles, nodeIndex

    if depth >= MAX_DEPTH or parent.triangleCount < 4:
        return

    split_axis, split_pos = bestSplit(parent.boundingBox)

    parent.childIndex = nodeIndex
    childA = BVHNode()
    childB = BVHNode()
    childA.triangleIndex = parent.triangleIndex
    childB.triangleIndex = parent.triangleIndex

    for i in range(parent.triangleIndex, parent.triangleIndex + parent.triangleCount):
        isSideA = allTriangles[i].center[split_axis] < split_pos
        if isSideA:
            childA.boundingBox.growToIncludeTriangle(allTriangles[i])
            childA.triangleCount += 1
            swap = childA.triangleIndex + childA.triangleCount - 1

            allTriangles[i], allTriangles[swap] = allTriangles[swap], allTriangles[i]
            childB.triangleIndex += 1
        else:
            childB.boundingBox.growToIncludeTriangle(allTriangles[i])
            childB.triangleCount += 1

    allNodes[nodeIndex] = childA
    nodeIndex += 1
    allNodes[nodeIndex] = childB
    nodeIndex += 1


    if progress_bar:
           progress_bar.update(parent.triangleCount)

    split(childA, depth+1, progress_bar)
    split(childB, depth+1, progress_bar)


def createBVH():
    global allNodes, allTriangles, nodeIndex

    root = BVHNode()
    root.triangleCount = len(allTriangles)
    allNodes[0] = root
    nodeIndex += 1


    # Pre-compute root bounding box
    for triangle in allTriangles:
        root.boundingBox.growToIncludeTriangle(triangle)

    total_operations = len(allTriangles) * (1 + int(np.log2(len(allTriangles))))

    with tqdm(total=total_operations, desc="Building BVH") as progress_bar:
        splitSAH(root, 0, progress_bar)

    allNodes = allNodes[:nodeIndex]

    return root

BVHroot = createBVH()
print("BVH created")

ray = Ray(origin=np.array([-50, 0., 200.]), direction=np.array([0, -0.5, -1.]))
ray_end = ray.origin + 400*ray.direction

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

def BVH_optimized(ray):
    global allNodes
    global allTriangles
    t_min = float('inf')
    k = -1
    current_node = 0
    while current_node != -1:
        node = allNodes[current_node]
        if node.boundingBox.hit(ray) < t_min:
            if node.childIndex == 0:
                for i in range(node.triangleIndex, node.triangleCount+node.triangleIndex):
                    didHit, t = allTriangles[i].hit(ray)
                    if didHit and t < t_min:
                        k = i
                        t_min = t
                current_node = node.escapeIndex
            else:
                dstA = allNodes[node.childIndex].boundingBox.hit(ray)
                dstB = allNodes[node.childIndex+1].boundingBox.hit(ray)
                if dstA < dstB:
                    allNodes[node.childIndex].escapeIndex = node.childIndex + 1
                    allNodes[node.childIndex + 1].escapeIndex = node.escapeIndex
                    current_node = node.childIndex
                else:
                    allNodes[node.childIndex+1].escapeIndex = node.childIndex
                    allNodes[node.childIndex].escapeIndex = node.escapeIndex
                    current_node = node.childIndex+1
        else:
            current_node = node.escapeIndex  # If no hit, jump to escape index
    return k, t_min

k,t_min = BVH_optimized(ray)

# Pyvista Plot
# plotter = pv.Plotter()
# pv_util.draw_BVH_efficient(plotter, allNodes)
# pv_util.plot_triangles(plotter, allTriangles, k)
# #pv_util.draw_point(plotter, ray.origin)
# #pv_util.draw_line(plotter, ray.origin, ray.origin+500*ray.direction, 'green', 2)
# # if k >= 0:
# #     pv_util.draw_point(plotter, ray.origin+t_min*ray.direction, 'blue', 10)
# plotter.show()

# Save to numpy
from create_buffers import createTriangleBuffer, createNodeBuffer

createTriangleBuffer(allTriangles, f"{name}_triangle")
createNodeBuffer(allNodes, f"{name}_nodes")

print("Triangles and Nodes Saved")

# Pyvista Plot
plotter = pv.Plotter()
pv_util.draw_BVH_efficient(plotter, allNodes)
pv_util.plot_triangles(plotter, allTriangles, 0)
plotter.show()
