import taichi as ti
from taichi.math import length
from taichi.math.mathimpl import normalize
from models.vector import Point, Vec3
from models.ray import Ray
from constants import *
from math import sqrt, tan

@ti.kernel
def len_ker(u:Vec3) -> ti.f32:
    return ti.math.length(u)

@ti.kernel
def normalize_ker(u:Vec3) -> Vec3:
    return ti.math.normalize(u)

@ti.kernel
def cross_ker(u:Vec3, v:Vec3) -> Vec3:
    return ti.math.cross(u, v)

@ti.data_oriented
class Camera:
    def __init__(self, lookfrom:Vec3, lookat:Vec3, vup:Vec3, aspect_ratio:ti.f32, image_width:ti.i32, vfov:ti.f32) -> None:
        self.lookfrom = lookfrom
        self.lookat = lookat
        self.vup = vup
        self.aspect_ratio = aspect_ratio
        self.image_width = image_width
        self.vfov = vfov

        #self.focal_length = sqrt((lookat.x - lookfrom.x)**2 + (lookat.y - lookfrom.y)**2 + (lookat.y - lookfrom.y)**2)

        self.image_height = int(image_width/aspect_ratio)
        self.resolution = (self.image_width, self.image_height)

        self.focal_length = len_ker(self.lookfrom-self.lookat)
        self.theta = self.vfov*(PI/180.)
        self.h = tan(self.theta/2)
        self.viewport_height = 2. * self.h * self.focal_length
        self.viewport_width = self.viewport_height * (self.image_width/self.image_height)

        self.w = normalize_ker(lookfrom - lookat)
        self.u = normalize_ker(cross_ker(vup, self.w))
        self.v = cross_ker(self.w, self.u)

        self.viewport_u = self.viewport_width*self.u
        self.viewport_v = self.viewport_height*self.v

        self.pixel_delta_u = self.viewport_u / self.image_width
        self.pixel_delta_v = self.viewport_v / self.image_height

        self.view_port_bottom_left = self.lookfrom - (self.focal_length*self.w) - self.viewport_u/2 - self.viewport_v/2
        self.pixel00_loc = self.view_port_bottom_left + 0.5*(self.pixel_delta_u + self.pixel_delta_v)

    @ti.func
    def get_ray(self, i:int, j:int) -> Ray:
        # Pour l'ant-aliasing
        offset_x = (ti.random() - 0.5)
        offset_y = (ti.random() - 0.5)

        pixel_center = self.pixel00_loc + ((i+offset_x) * self.pixel_delta_u) + ((j+offset_y) * self.pixel_delta_v)
        ray_direction = pixel_center - self.lookfrom
        return Ray(self.lookfrom, ray_direction)
