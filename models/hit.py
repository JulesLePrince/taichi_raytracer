import taichi as ti
from models.vector import Point, Vec3, MaterialCoord
from material import Lambertian

@ti.dataclass
class HitInfo:
    didHit: bool
    dst: ti.types.f32
    hitPoint: Point
    normal: Vec3
    material_coord: MaterialCoord
    frontFace: bool
