from models.vector import Vec3, Color, Point, MaterialCoord
from material import Lambertian, Metal, DiffuseLight
from models.camera import Camera
from models.sphere import Sphere
from models.quad import Quad, create_quad
from models.scene import Scene
from utils.create_buffers import quadListToBuffer, sphereListToBuffer

aspect_ratio = 16./9.
lookfrom = Point(26., 3., 6.)
lookat = Vec3(0., 2., 0.)
vfov = 20.
vup = Vec3(0., 1., 0.)

im_width = 1080

#hdri_env = load_hdri_image_from_file('hdris/rogland_sunset_4k.exr')

lambertian_materials = [
    Lambertian(albedo=Color(1., 1., 1.)),
    Lambertian(albedo=Color(0.2, 1., 0.2)),
    Lambertian(albedo=Color(0.2, 0.2, 1.)),
    Lambertian(albedo=Color(1.0, 0.5, 0.)),
    Lambertian(albedo=Color(0.2, 0.8, 0.8)),
]

metal_materials = [
    Metal(albedo=Color(0.8, 0.8, 0.8), fuzz=0.3),
    Metal(albedo=Color(0.8, 0.6, 0.2), fuzz=0.)
]

p = 4
diffuseLight_materials = [
    DiffuseLight(emmissionColor=Color(p, p, p)),
]

cam = Camera(lookfrom=lookfrom, lookat=lookat, vup=vup, aspect_ratio=aspect_ratio, image_width=im_width, vfov=vfov)
res = cam.resolution

sph1 = Sphere(center=Point([0., -1000., 0.]), radius=1000, material_coord=MaterialCoord(0, 0))
sph2 = Sphere(center=Point([0., 2., 0.]), radius=2, material_coord=MaterialCoord(0, 4))
sph3 = Sphere(center=Point([0., 7., 0.]), radius=2, material_coord=MaterialCoord(2, 0))
sphere_list=[sph1, sph2, sph3]
sphere_buffer = sphereListToBuffer(sphere_list)

qu1 = create_quad(q=Point(3., 1., -2.), u=Vec3(2., 0., 0.), v=Vec3(0., 2., 0.), material_coord=(2, 0))
quad_list=[qu1]
quad_buffer = quadListToBuffer(quad_list)

scene = Scene(sphere_buffer=sphere_buffer, quad_buffer=quad_buffer, lamb_materials=lambertian_materials, metal_materials=metal_materials, diffuseLight_materials=diffuseLight_materials)
