import taichi as ti
import numpy as np
from taichi.math.mathimpl import dot, sign
from models.vector import Vec3

normalize = ti.math.normalize
dot = ti.math.dot

@ti.func
def randomNormal():
    u1 = ti.random()
    u2 = ti.random()
    return ti.sqrt(-2.0 * ti.log(u1)) * ti.cos(2.0 *ti.math.pi * u2)

@ti.func
def randomDirection():
    x = randomNormal()
    y = randomNormal()
    z = randomNormal()
    return normalize(Vec3(x, y, z))

@ti.func
def randomHemisphereDirection(normal:Vec3):
    dir = randomDirection()
    return sign(dot(dir, normal))*dir
