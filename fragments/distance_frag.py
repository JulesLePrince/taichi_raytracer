import taichi as ti
from models.vector import Color, Point, Vec3
from models.sphere import Sphere
from models.hit import HitInfo
from environments.dumb_environment import dumb_env


normalize = ti.math.normalize


@ti.func
def fragment(ray, u, v, res, scene):
    hitInf = scene.hit(ray)
    result = Color([0., 0., 0.])
    if hitInf.didHit:
        dst = hitInf.dst/700
        result = 0.5*Color(dst, dst, dst);
    else:
        result = Color(1., 1., 1.)
    return result
