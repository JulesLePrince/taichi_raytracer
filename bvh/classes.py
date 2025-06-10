import numpy as np

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
                    return float('inf')
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
                    return float('inf')

        return t_min


    @property
    def center(self):
        return (self.min + self.max) * 0.5

    @property
    def size(self):
        return self.max - self.min

    @property
    def surfaceArea(self):
        """Calculate the surface area of the bounding box"""
        extent = self.max - self.min
        return 2.0 * (extent[0] * extent[1] + extent[1] * extent[2] + extent[2] * extent[0])

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
        self.escapeIndex = -1
