import taichi as ti
from models.hit import HitInfo
from models.ray import Ray
from models.vector import Color, Vec3, Point, MaterialCoord
from constants import *

dot = ti.math.dot
normalize = ti.math.normalize
sqrt = ti.math.sqrt
cross = ti.math.cross

@ti.kernel
def normalize_ker(u:Vec3) -> Vec3:
    return normalize(u)

@ti.kernel
def dot_ker(u:Vec3, v:Vec3) -> ti.f32:
    return dot(u, v)

@ti.kernel
def cross_ker(u:Vec3, v:Vec3) -> Vec3:
    return cross(u, v)

@ti.dataclass
class Quad:
    q: Point
    u: Vec3
    v: Vec3
    normal: Vec3
    w: Vec3
    d: ti.f32
    material_coord: MaterialCoord



    @ti.func
    def is_interior(self, a, b):
        return (a >= 0 and a <= 1) and (b >= 0 and b <= 1)

    @ti.func
    def hit(self, ray:Ray) -> HitInfo:
        hitInf = HitInfo(didHit=False)
        normal = self.normal
        d = self.d
        denom = dot(normal, ray.direction)

        if abs(denom) > EPS:
            dst = (d - dot(normal, ray.origin)) / denom

            if dst > EPS:
                intersection = ray.at(dst)
                planar_hitpt_vector = intersection - self.q

                # Cross product of u and v gives the area of the parallelogram
                area_uv = dot(self.w, cross(self.u, self.v))

                # Calculate alpha and beta
                alpha = dot(self.w, cross(planar_hitpt_vector, self.v)) / area_uv
                beta = dot(self.w, cross(self.u, planar_hitpt_vector)) / area_uv

                good_direction_normal = -ti.math.sign(dot(normal, ray.direction))*normal
                # Check if the hit point lies within the quad
                if self.is_interior(alpha, beta):
                    hitInf.didHit = True
                    hitInf.hitPoint = intersection
                    hitInf.dst = (ray.origin - intersection).norm()
                    hitInf.normal = good_direction_normal
                    hitInf.material_coord = self.material_coord
        return hitInf

def create_quad(q, u, v, material_coord):
    normal = normalize_ker(cross_ker(u, v))
    d = dot_ker(normal, q)
    w = normal / dot_ker(normal, normal)
    return Quad(q=q, u=u, v=v, normal=normal, d=d, w=w, material_coord=material_coord)
