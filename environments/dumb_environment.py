import taichi as ti
from models.vector import Color

@ti.func
def dumb_env(ray):
    unit_direction = ti.math.normalize(ray.direction);
    a = 0.5*(unit_direction.y + 1.0);
    return (1.0-a)*Color(1.0, 1.0, 1.0) + a*Color(0.5, 0.7, 1.0)
