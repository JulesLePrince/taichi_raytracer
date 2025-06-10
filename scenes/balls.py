from models.vector import Vec3, Color, Point, MaterialCoord
from material import Lambertian, Metal, DiffuseLight
from models.camera import Camera
from models.sphere import Sphere
from models.triangle import Triangle, createTriangle
from models.quad import Quad
from models.scene import Scene

center = Vec3([0., 0., 0.])
aspect_ratio = 16/9
im_width = 1080
focal_len = 1.0

#hdri_env = load_hdri_image_from_file('hdris/rogland_sunset_4k.exr')

lambertian_materials = [
    Lambertian(albedo=Color(0.8, 0.8, 0.)),
    Lambertian(albedo=Color(0.1, 0.2, 0.5)),
    Lambertian(albedo=Color(0.7, 0.7, 0.7))
]

metal_materials = [
    Metal(albedo=Color(0.8, 0.8, 0.8), fuzz=0.),
    Metal(albedo=Color(0.8, 0.6, 0.2), fuzz=0.)
]

p = 5
diffuseLight_materials = [
    DiffuseLight(emmissionColor=Color(p, p, p)),
    DiffuseLight(emmissionColor=Color(1., 1., 1.)),
]

cam = Camera(lookfrom=Point(-2., 2., 1.), lookat=Point(0., 0., -1.), vup=Vec3(0., 1., 0.), aspect_ratio=aspect_ratio, image_width=im_width, vfov=90.)
res = cam.resolution

sph1 = Sphere(center=Point([0., -100.5, -1]), radius=100, material_coord=MaterialCoord(0, 0))
sph2 = Sphere(center=Point([0., 10., -2]), radius=5, material_coord=MaterialCoord(2, 0))
sph3 = Sphere(center=Point([-1, 0., -1]), radius=0.5, material_coord=MaterialCoord(1, 0))
sph4 = Sphere(center=Point([1., 0., -1.]), radius=0.5, material_coord=MaterialCoord(0, 1))

t1 = createTriangle(v0=Point(0., 1., -2), v1=Point(-1, 0., -1), v2=Point(1., 0., -1), mat_coord=(0, 2))

scene = Scene(sphere_list=[sph1, sph2, sph3, sph4],triangle_list=[t1], lamb_materials=lambertian_materials, metal_materials=metal_materials, diffuseLight_materials=diffuseLight_materials)
