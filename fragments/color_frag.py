import taichi as ti
from models.vector import Color, Point, Vec3
from models.sphere import Sphere
from models.hit import HitInfo
from models.scene import Scene

normalize = ti.math.normalize

@ti.func
def fragment(ray, u, v, res, scene):
    hitInf = scene.hit(ray)
    result = Color([0., 0., 0.])
    if hitInf.didHit:
        result = hitInf.material.albedo
    else:
        # unit_direction = ti.math.normalize(ray.direction);
        # a = 0.5*(unit_direction.y + 1.0);
        # result = (1.0-a)*Color(1.0, 1.0, 1.0) + a*Color(0.5, 0.7, 1.0);
        result = Color(0., 0., 0.)
    return result
