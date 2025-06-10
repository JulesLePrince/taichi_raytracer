import taichi as ti
from models.vector import MaterialCoord, Vec3, Point
from models.hit import HitInfo
from constants import *
from models.ray import Ray


@ti.dataclass
class BoundingBox:
    min: Vec3
    max: Vec3

    @ti.kernel
    def center(self):
        return (self.min + self.max) * 0.5

    @ti.kernel
    def size(self):
        return self.max - self.min

    # @ti.func
    # def growToIncludePoint(self, point):
    #     self.min = ti.math.min(self.min, point)
    #     self.max = ti.math.max(self.max, point)

    # @ti.func
    # def growToIncludeTriangle(self, triangle):
    #     self.growToIncludePoint(triangle.v0)
    #     self.growToIncludePoint(triangle.v1)
    #     self.growToIncludePoint(triangle.v2)

    @ti.func
    def hit(self, ray):
        """
        Check if the ray intersects with the bounding box.
        Returns True if there's an intersection, False otherwise.
        """
        t_min = float('-inf')
        t_max = float('inf')
        res = True

        for i in range(3):
            # Handle case where ray direction component is close to 0
            if ti.abs(ray.direction[i]) < EPS:
                if ray.origin[i] < self.min[i] or ray.origin[i] > self.max[i]:
                    res = False
            else:
                # Calculate intersection distances
                t1 = (self.min[i] - ray.origin[i]) / ray.direction[i]
                t2 = (self.max[i] - ray.origin[i]) / ray.direction[i]

                # Ensure t1 is the smaller value
                if t1 > t2:
                    t1, t2 = t2, t1

                t_min = ti.max(t_min, t1)
                t_max = ti.min(t_max, t2)

                if t_min > t_max:
                    res = False

        return ti.select(res, t_min, MAX_LEN)

@ti.dataclass
class BVHNode:
    boundingBox: BoundingBox
    childIndex: ti.i32
    triangleIndex: ti.i32
    triangleCount: ti.i32

@ti.dataclass
class Mesh:
    beginIndex: ti.i32
    meshLen: ti.i32
    material_coord: MaterialCoord
    worldToLocal: ti.types.matrix(4, 4, ti.f32)  # 4x4

    @ti.func
    def hit(self, ray, triangle_buffer, bvhNodes_buffer):
        localRay = self.transformRay(ray)
        closest_hit = HitInfo(didHit=False, dst=MAX_LEN)
        stack = ti.Vector([0] * 50)
        stack_ptr = 0
        current_node = 0

        triangle_test_count = 0
        box_test_count = 0

        while stack_ptr >= 0:
            current_node = stack[stack_ptr]
            stack_ptr -= 1
            box_test_count += 1

            node = bvhNodes_buffer[current_node]
            if node.childIndex == 0:  # Leaf node
                for i in range(node.triangleIndex, node.triangleIndex + node.triangleCount):
                    triangle = triangle_buffer[i]
                    intersect = triangle.hit(localRay)
                    triangle_test_count += 1
                    if intersect.didHit and intersect.dst < closest_hit.dst:
                        closest_hit = intersect
            else:  # Internal node
                leftChild = node.childIndex
                rightChild = node.childIndex + 1

                dstLeft = bvhNodes_buffer[leftChild].boundingBox.hit(localRay)
                dstRight = bvhNodes_buffer[rightChild].boundingBox.hit(localRay)

                # Push children to stack in far-to-near order
                if dstLeft < dstRight:
                    if dstRight < closest_hit.dst:
                        stack_ptr += 1
                        stack[stack_ptr] = rightChild
                    if dstLeft < closest_hit.dst:
                        stack_ptr += 1
                        stack[stack_ptr] = leftChild
                else:
                    if dstLeft < closest_hit.dst:
                        stack_ptr += 1
                        stack[stack_ptr] = leftChild
                    if dstRight < closest_hit.dst:
                        stack_ptr += 1
                        stack[stack_ptr] = rightChild

        if closest_hit.didHit:
            closest_hit = self.transformHitInfo(closest_hit, ray)
        return closest_hit, triangle_test_count, box_test_count


    @ti.func
    def transformRay(self, ray):
        # Transform ray origin and direction to local space
        origin = ti.Vector([ray.origin[0], ray.origin[1], ray.origin[2], 1.0])
        direction = ti.Vector([ray.direction[0], ray.direction[1], ray.direction[2], 0.0])

        local_origin = self.worldToLocal @ origin
        local_direction = self.worldToLocal @ direction

        return Ray(
            origin=ti.Vector([local_origin[0], local_origin[1], local_origin[2]]),
            direction=ti.Vector([local_direction[0], local_direction[1], local_direction[2]]).normalized()
        )

    @ti.func
    def transformHitInfo(self, hit, ray):
        # Transform hit point
        point = ti.Vector([hit.hitPoint.x, hit.hitPoint.y, hit.hitPoint.z, 1.0])
        world_point = (self.worldToLocal.inverse() @ point).xyz

        # Transform normal (using transpose of inverse matrix)
        normal = ti.Vector([hit.normal.x, hit.normal.y, hit.normal.z, 0.0])
        world_normal = (self.worldToLocal.transpose() @ normal).xyz.normalized()

        # Update hit point and normal in world space
        hit.hitPoint = Point(world_point[0], world_point[1], world_point[2])
        hit.normal = Vec3(world_normal[0], world_normal[1], world_normal[2])

        # Correctly update distance from ray origin to the world-space hit point
        ray_origin_world = Vec3([ray.origin.x, ray.origin.y, ray.origin.z])
        hit.dst = (world_point - ray_origin_world).norm()

        return hit

def createBoundingBox():
    return BoundingBox(min=Vec3(1.)*float('-inf'), max=Vec3(1.)*float('inf'))

def split(parent, depth):
    allNodes = []
    if depth < 10:
        size = parent.boundingBox.size()
        splitAxis = 0 if size.x > max(size.y, size.y) else (1 if size.y > size.y else 2)
        splitPos = parent.boundingBox.center()[splitAxis]
