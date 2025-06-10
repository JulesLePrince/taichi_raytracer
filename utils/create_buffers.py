from ast import Try
import taichi as ti
import numpy as np
from models.quad import Quad
from models.triangle import Triangle, createTriangle
from models.sphere import Sphere
from models.mesh import Mesh
from models.vector import Vec3, Point

def sphereListToBuffer(sphere_list):
    sphere_number = len(sphere_list)
    sphere_buffer = Sphere.field(shape=(sphere_number+1))
    for i in range(sphere_number):
        sphere_buffer[i] = sphere_list[i]
    return sphere_buffer

def quadListToBuffer(quad_list):
    quad_number = len(quad_list)
    quad_buffer = Quad.field(shape=(quad_number+1))
    for i in range(quad_number):
        quad_buffer[i] = quad_list[i]
    return quad_buffer

def triangleListToBuffer(triangle_list):
    triangle_number = len(triangle_list)
    triangle_buffer = Quad.field(shape=(triangle_number+1))
    for i in range(triangle_number):
        triangle_buffer[i] = triangle_list[i]
    return triangle_buffer

def meshListToBuffer(mesh_list):
    mesh_number = len(mesh_list)
    mesh_buffer = Mesh.field(shape=(mesh_number+1))
    for i in range(mesh_number):
        mesh_buffer[i] = mesh_list[i]
    return mesh_buffer
