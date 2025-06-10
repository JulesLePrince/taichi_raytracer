import taichi as ti
from models.vector import Color, Point, Vec3
from models.sphere import Sphere
from models.hit import HitInfo
from environments.dumb_environment import dumb_env


normalize = ti.math.normalize


@ti.func
def fragment(ray, u, v, res, scene):
    hitInf = scene.hit(ray, 0)[0]
    result = Color([0., 0., 0.])
    if hitInf.didHit:
        N = hitInf.normal
        result = 0.5*Color(N.x+1, N.y+1, N.z+1);
    else:
        Color(0., 0., 0.)
    return result
