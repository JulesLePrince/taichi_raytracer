from models.vector import Vec3, Color, Point, MaterialCoord
from material import Lambertian, Metal, DiffuseLight
from models.camera import Camera
from models.sphere import Sphere
from models.quad import Quad, create_quad
from models.scene import Scene
from utils.create_buffers import quadListToBuffer, sphereListToBuffer

aspect_ratio = 1.
lookfrom = Point(0., 0., 9.)
lookat = Vec3(0., 0., 0.)
vfov = 80.
vup = Vec3(0., 1., 0.)

im_width = 720

#hdri_env = load_hdri_image_from_file('hdris/rogland_sunset_4k.exr')

lambertian_materials = [
    Lambertian(albedo=Color(1., 0.2, 0.2)),
    Lambertian(albedo=Color(0.2, 1., 0.2)),
    Lambertian(albedo=Color(0.2, 0.2, 1.)),
    Lambertian(albedo=Color(1.0, 0.5, 0.)),
    Lambertian(albedo=Color(0.2, 0.8, 0.8)),
]

metal_materials = [
    Metal(albedo=Color(0.8, 0.8, 0.8), fuzz=0.3),
    Metal(albedo=Color(0.8, 0.6, 0.2), fuzz=0.)
]

p = 3
diffuseLight_materials = [
    DiffuseLight(emmissionColor=Color(p, p, p)),
    DiffuseLight(emmissionColor=Color(1., 1., 1.)),
]

cam = Camera(lookfrom=lookfrom, lookat=lookat, vup=vup, aspect_ratio=aspect_ratio, image_width=im_width, vfov=vfov)
res = cam.resolution

qu1 = create_quad(q=Point(-3., -2., 5.), u=Vec3(0., 0., -4), v=Vec3(0., 4., 0.), material_coord=(0, 0))
qu2 = create_quad(q=Point(-2., -2., 0.), u=Vec3(4., 0., 0.), v=Vec3(0., 4., 0.), material_coord=(0, 1))
qu3 = create_quad(q=Point(3., -2., 1.), u=Vec3(0., 0., 4.), v=Vec3(0., 4., 0.), material_coord=(0, 2))
qu4 = create_quad(q=Point(-2., 3., 1.), u=Vec3(4., 0., 0.), v=Vec3(0., 0., 4.), material_coord=(0, 3))
qu5 = create_quad(q=Point(-2., -3., 5.), u=Vec3(4., 0., 0.), v=Vec3(0., 0., -4.), material_coord=(0, 4))
quad_list=[qu1, qu2, qu3, qu4, qu5]
quad_buffer = quadListToBuffer(quad_list)

sph1 = Sphere(center=Point(0., 0., 3.), radius=2, material_coord=(2, 0))
sphere_list = [sph1]
sphere_buffer = sphereListToBuffer(sphere_list)

scene = Scene(sphere_buffer=sphere_buffer, quad_buffer=quad_buffer, lamb_materials=lambertian_materials, metal_materials=metal_materials, diffuseLight_materials=diffuseLight_materials)
