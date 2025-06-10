import taichi as ti
from models.vector import Color, Point, Vec3
from models.sphere import Sphere
from models.hit import HitInfo
from models.scene import Scene
from models.integrator import trace
from constants import MAX_BOUNCE
from math import exp



normalize = ti.math.normalize


"""
Samothrace : 12 | 100
Dragon : 38 | 201
"""

@ti.func
def fragment(ray, u, v, res, scene):
    _, nb_triangle_tests, nb_boxes_tested = trace(ray, scene)
    y = nb_triangle_tests/35
    z = nb_boxes_tested/600

    neutral_color = {"r":203/255, "g":239/255 , "b":255/255}
    power_color = {"r":255/255, "g":0/255 , "b":0/255}

    x = (nb_triangle_tests+nb_boxes_tested)/600

    r = (power_color["r"]-neutral_color["r"])*x+neutral_color["r"]
    g = (power_color["g"]-neutral_color["g"])*x+neutral_color["g"]
    b = (power_color["b"]-neutral_color["b"])*x+neutral_color["b"]
    #r = r**3
    #b = b**3
    return Color(r, g, b)
