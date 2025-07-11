import taichi as ti
from models.vector import Color, Point, Vec3
from models.sphere import Sphere
from models.hit import HitInfo
from models.scene import Scene
from models.integrator import trace



normalize = ti.math.normalize

@ti.func
def fragment(ray, u, v, res, scene):
    col, nb_triangle_tests = trace(ray, scene)
    return nb_triangle_tests
