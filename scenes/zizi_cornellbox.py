from models.vector import Vec3, Color, Point, MaterialCoord
from material import Lambertian, Metal, DiffuseLight
from models.camera import Camera
from models.sphere import Sphere
from models.quad import Quad, create_quad
from models.scene import Scene
from models.triangle import Triangle
from constants import *
from utils.create_buffers import quadListToBuffer, sphereListToBuffer, meshListToBuffer
from utils.make_matrix import make_transform_mat
import numpy as np
from models.mesh import BVHNode, Mesh

aspect_ratio = 1.
lookfrom = Point(278., 278., -800.)
lookat = Vec3(278., 278., 0.)
vfov = 38.
vup = Vec3(0., 1., 0.)
im_width = 756

#hdri_env = load_hdri_image_from_file('hdris/rogland_sunset_4k.exr')

lambertian_materials = [
    Lambertian(albedo=Color(.65, .05, .05)), # Red
    Lambertian(albedo=Color(0.73, .73, .73)), # White
    Lambertian(albedo=Color(0.12, 0.45, .15)) # Green
]

metal_materials = [
    Metal(albedo=Color(0.8, 0.8, 0.8), fuzz=0.),
]

p = 25
diffuseLight_materials = [
    DiffuseLight(emmissionColor=Color(p, p, p))
]

cam = Camera(lookfrom=lookfrom, lookat=lookat, vup=vup, aspect_ratio=aspect_ratio, image_width=im_width, vfov=vfov)
res = cam.resolution

sph1 = Sphere(center=Point([555/2., 75., 230.]), radius=75, material_coord=MaterialCoord(0, 1))
sphere_list = []
sphere_buffer = sphereListToBuffer(sphere_list)

qu1 = create_quad(q=Point(555., 0., 0.), u=Vec3(0., 555., 0.), v=Vec3(0., 0., 555.), material_coord=(0, 2))
qu2 = create_quad(q=Point(0., 0., 0.), u=Vec3(0., 555., 0.), v=Vec3(0., 0., 555.), material_coord=(0, 0))
qu3 = create_quad(q=Point(343., 554., 332.), u=Vec3(-130., 0., 0.), v=Vec3(0., 0., -105.), material_coord=(2, 0))
qu4 = create_quad(q=Point(0., 0., 0.), u=Vec3(555., 0., 0.), v=Vec3(0., 0., 555.), material_coord=(0, 1))
qu5 = create_quad(q=Point(555., 555., 555.), u=Vec3(-555., 0., 0.), v=Vec3(0., 0., -555.), material_coord=(0, 1))
qu6 = create_quad(q=Point(0., 0., 555.), u=Vec3(555., 0., 0.), v=Vec3(0., 555., 0.), material_coord=(0, 1))
qu7 = create_quad(q=Point(0., 0., 0.), u=Vec3(555., 0., 0.), v=Vec3(0., 555., 0.), material_coord=(0, 1))

quad_list = [qu1, qu2, qu3, qu4, qu5, qu6, qu7]
#quad_list = []
quad_buffer = quadListToBuffer(quad_list=quad_list)


triangle_numpy = np.load('meshes/zizi_triangle.npy', allow_pickle=True).item()
triangle_number = len(triangle_numpy['v0'])
print(f"Chargement de {triangle_number} triangles")
triangle_buffer = Triangle.field(shape=(triangle_number+1))
triangle_buffer.from_numpy(triangle_numpy)

bvhNodes_numpy = np.load('meshes/zizi_nodes.npy', allow_pickle=True).item()
node_number = len(bvhNodes_numpy['childIndex'])
bvhNode_buffer = BVHNode.field(shape=node_number)
bvhNode_buffer.from_numpy(bvhNodes_numpy)

transform = make_transform_mat(
    #ti.Vector([278.0, 240.0, 250.0]),  # translation
    ti.Vector([278.0, 200.0, 250.0]),  # translation
    ti.Vector([0.0, 300.0, 0.0]), # rotation (degrees)
    ti.Vector([0.8, 0.8, 0.8])   # scale
)

me1 = Mesh(beginIndex= 0, meshLen= triangle_number, material_coord=(0, 1), worldToLocal=transform)
mesh_buffer = meshListToBuffer([me1])


scene = Scene(sphere_buffer=sphere_buffer, quad_buffer=quad_buffer, triangle_buffer=triangle_buffer, bvhNode_buffer=bvhNode_buffer, mesh_buffer=mesh_buffer, lamb_materials=lambertian_materials, metal_materials=metal_materials, diffuseLight_materials=diffuseLight_materials)
