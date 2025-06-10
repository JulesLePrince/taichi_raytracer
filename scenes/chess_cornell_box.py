from numpy._core.fromnumeric import shape
from models.vector import Vec3, Color, Point, MaterialCoord
from material import Lambertian, Metal, DiffuseLight
from models.camera import Camera
from models.sphere import Sphere
from models.quad import create_quad
from models.triangle import createTriangle
from models.scene import Scene
from constants import *
import numpy as np
from models.triangle import Triangle
from utils.create_buffers import quadListToBuffer

aspect_ratio = 1.
lookfrom = Point(278., 278., -800.)
lookat = Vec3(278., 278., 0.)
vfov = 40.
vup = Vec3(0., 1., 0.)
im_width = 800

triangles = np.load('obj/king.npy')
triangle_number = len(triangles)
scale_factor = 100  # For example, make it twice as large
translation_vector = np.array([278.0, 0.0, 200.0])

#triangle_buffer = triangleVerticesListToBuffer(translated_triangles)
triangle_buffer = Triangle.field(shape=(631))


#array_dict = triangle_buffer.to_numpy()
#np.save('test_buffer.npy', array_dict, allow_pickle=True)

buff = np.load('test_buffer.npy', allow_pickle=True).item()
triangle_buffer.from_numpy(buff)

lambertian_materials = [
    Lambertian(albedo=Color(.65, .05, .05)), # Red
    Lambertian(albedo=Color(0.73, .73, .73)), # White
    Lambertian(albedo=Color(0.12, 0.45, .15)) # Green
]

metal_materials = [
    Metal(albedo=Color(0.8, 0.8, 0.8), fuzz=0.),
]

p = 20
diffuseLight_materials = [
    DiffuseLight(emmissionColor=Color(p+20, p+7, p))
]

cam = Camera(lookfrom=lookfrom, lookat=lookat, vup=vup, aspect_ratio=aspect_ratio, image_width=im_width, vfov=vfov)
res = cam.resolution



qu1 = create_quad(q=Point(555., 0., 0.), u=Vec3(0., 555., 0.), v=Vec3(0., 0., 555.), material_coord=(0, 2))
qu2 = create_quad(q=Point(0., 0., 0.), u=Vec3(0., 555., 0.), v=Vec3(0., 0., 555.), material_coord=(0, 0))
qu3 = create_quad(q=Point(343., 554., 332.), u=Vec3(-130., 0., 0.), v=Vec3(0., 0., -105.), material_coord=(2, 0))
qu4 = create_quad(q=Point(0., 0., 0.), u=Vec3(555., 0., 0.), v=Vec3(0., 0., 555.), material_coord=(0, 1))
qu5 = create_quad(q=Point(555., 555., 555.), u=Vec3(-555., 0., 0.), v=Vec3(0., 0., -555.), material_coord=(0, 1))
qu6 = create_quad(q=Point(0., 0., 555.), u=Vec3(555., 0., 0.), v=Vec3(0., 555., 0.), material_coord=(0, 1))
quad_list = [qu1, qu2, qu3, qu4, qu5, qu6]
quad_buffer = quadListToBuffer(quad_list)

t1 = createTriangle(v0=Point([0., 0., 250.]) ,v1=Point(200., 30., 200.), v2=Point(343., 200., 332.), mat_coord=(0, 1))


scene = Scene(quad_buffer=quad_buffer, triangle_buffer=triangle_buffer, lamb_materials=lambertian_materials, metal_materials=metal_materials, diffuseLight_materials=diffuseLight_materials)
