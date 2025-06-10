import taichi as ti
from models.hit import HitInfo
from models.ray import Ray
from models.vector import Color, Vec3, Point, MaterialCoord
from constants import *
from tkinter.filedialog import Directory


dot = ti.math.dot
normalize = ti.math.normalize
sqrt = ti.math.sqrt


@ti.dataclass
class Sphere:
    center: Point
    radius: ti.f32
    material_coord: MaterialCoord

    @ti.func
    def hit(self, ray:Ray) -> HitInfo:
        hitInf = HitInfo(didHit=False)
        oc = self.center - ray.origin
        a = dot(ray.direction, ray.direction)
        b = -2.0*dot(ray.direction, oc)
        c = dot(oc, oc) - self.radius*self.radius
        discriminant = b*b - 4*a*c
        if discriminant >= 0:
            dst = (-b - sqrt(discriminant) ) / (2.0*a)
            if dst >= 0:
                hitInf.didHit = True
                hitInf.hitPoint = ray.at(dst)
                hitInf.normal = normalize(hitInf.hitPoint - self.center)
                hitInf.dst = (ray.origin-hitInf.hitPoint).norm()
                hitInf.material_coord = self.material_coord
        return hitInf
