import taichi as ti
from taichi.math import normalize, cross
import material
from models.hit import HitInfo
from models.ray import Ray
from models.vector import Color, Vec3, Point, MaterialCoord
from constants import *


dot = ti.math.dot
normalize = ti.math.normalize
sqrt = ti.math.sqrt

@ti.dataclass
class Triangle:
    v0: Point
    v1: Point
    v2: Point
    e1: Vec3
    e2: Vec3
    normal: Vec3
    material_coord: MaterialCoord

    @ti.func
    def hit(self, ray):
        hitInf = HitInfo(didHit=False)
        T = ray.origin - self.v0
        D = ray.direction

        # Precompute cross products
        P = ti.math.cross(D, self.e2)
        det = ti.math.dot(P, self.e1)

        # Early exit for determinant close to zero
        if ti.abs(det) > EPS:
            inv_det = 1.0 / det

            u = ti.math.dot(P, T) * inv_det

            # Early exit if u is out of range
            if u >= 0 and u <= 1:
                Q = ti.math.cross(T, self.e1)
                v = ti.math.dot(Q, D) * inv_det

                # Early exit if v is out of range
                if v >= 0 and (u + v) <= 1:
                    t = ti.math.dot(Q, self.e2) * inv_det

                    if t > EPS:
                        hitInf.didHit = True
                        hitInf.hitPoint = ray.origin + t * D
                        hitInf.dst = t
                        hitInf.normal = -ti.math.sign(ti.math.dot(self.normal, D)) * self.normal
                        hitInf.material_coord = self.material_coord

        return hitInf

@ti.kernel
def createTriangle(v0:Point, v1:Point, v2:Point, mat_coord:MaterialCoord) -> Triangle:
    e1 = v1 - v0
    e2 = v2 - v0
    normal = normalize(cross(e1, e2))
    return Triangle(v0=v0, v1=v1, v2=v2, e1=e1, e2=e2, normal=normal, material_coord=mat_coord)
