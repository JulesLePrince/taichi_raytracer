from models.vector import Vec3, Color, Point, MaterialCoord
from material import Lambertian, Metal, DiffuseLight
from models.camera import Camera
from models.sphere import Sphere
from models.quad import Quad, create_quad
from models.scene import Scene
from utils.create_buffers import quadListToBuffer, sphereListToBuffer, meshListToBuffer
from utils.make_matrix import make_transform_mat
import numpy as np
from models.mesh import Mesh, BVHNode
from models.triangle import Triangle

aspect_ratio = 16./9.
lookfrom = Point(26., 3., 6.)
lookat = Vec3(0., 2., 0.)
vfov = 20.
vup = Vec3(0., 1., 0.)

im_width = 1920

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

sph1 = Sphere(center=Point([0., 10., 0.]), radius=2, material_coord=MaterialCoord(2, 0))
sph2 = Sphere(center=Point([0., -1000., 0.]), radius=1000, material_coord=MaterialCoord(0, 0))

sphere_list=[sph1, sph2]
sphere_buffer = sphereListToBuffer(sphere_list)

qu1 = create_quad(q=Point(3., 1., -2.), u=Vec3(2., 0., 0.), v=Vec3(0., 2., 0.), material_coord=(2, 0))
quad_list=[qu1]
quad_buffer = quadListToBuffer(quad_list)

triangle_numpy = np.load('meshes/la_valse_triangle.npy', allow_pickle=True).item()
triangle_number = len(triangle_numpy['v0'])
triangle_buffer = Triangle.field(shape=(triangle_number+1))
triangle_buffer.from_numpy(triangle_numpy)

bvhNodes_numpy = np.load('meshes/la_valse_nodes.npy', allow_pickle=True).item()
node_number = len(bvhNodes_numpy['childIndex'])
bvhNode_buffer = BVHNode.field(shape=node_number)
bvhNode_buffer.from_numpy(bvhNodes_numpy)

transform = make_transform_mat(
    #ti.Vector([278.0, 240.0, 250.0]),  # translation
    Vec3([3.0, 2.7, 2.0]),  # translation
    Vec3([180.0, 90.0, 0.0]), # rotation (degrees)
    Vec3([0.015, 0.015, 0.015])   # scale
)

me1 = Mesh(beginIndex=0, meshLen=triangle_number, material_coord=(0, 2), worldToLocal=transform)
mesh_buffer = meshListToBuffer([me1])

scene = Scene(sphere_buffer=sphere_buffer, quad_buffer=quad_buffer, triangle_buffer=triangle_buffer, bvhNode_buffer=bvhNode_buffer, mesh_buffer=mesh_buffer, lamb_materials=lambertian_materials, metal_materials=metal_materials, diffuseLight_materials=diffuseLight_materials)
