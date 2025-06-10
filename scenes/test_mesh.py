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
from models.mesh import Mesh, BVHNode
from utils.create_buffers import meshListToBuffer, quadListToBuffer
from utils.make_matrix import make_transform_mat

aspect_ratio = 1.
lookfrom = Point(1000., 278., -1500.)
lookat = Vec3(30., 0., 0.)
vfov = 20.
vup = Vec3(0., 1., 0.)
im_width = 800

triangle_numpy = np.load('meshes/la_valse_triangle.npy', allow_pickle=True).item()
triangle_number = len(triangle_numpy['v0'])
triangle_buffer = Triangle.field(shape=(triangle_number+1))
triangle_buffer.from_numpy(triangle_numpy)

bvhNodes_numpy = np.load('meshes/la_valse_nodes.npy', allow_pickle=True).item()
node_number = len(bvhNodes_numpy['childIndex'])
bvhNode_buffer = BVHNode.field(shape=node_number)
bvhNode_buffer.from_numpy(bvhNodes_numpy)

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
    DiffuseLight(emmissionColor=Color(p, p, p)),
    DiffuseLight(emmissionColor=Color(4*0.36, 4*0.3, 4*0.8))
]


cam = Camera(lookfrom=lookfrom, lookat=lookat, vup=vup, aspect_ratio=aspect_ratio, image_width=im_width, vfov=vfov)
res = cam.resolution


qu1 = create_quad(q=Point(-50.89234 , -500.5676  ,  -41.291376), u=Vec3(300., 200., 200.), v=Vec3(200., -200., -200.), material_coord=(2, 0))
qu2 = create_quad(q=Point(-122.89234 , 200  ,  -41.291376), u=Vec3(1500., 0., 1000.), v=Vec3(-1000., -0., -1000.), material_coord=(2, 1))
quad_buffer = quadListToBuffer([])

transform = make_transform_mat(
    ti.Vector([180.0, 0.0, 0.0]),  # translation
    ti.Vector([0.0, 0.0, 0.0]), # rotation (degrees)
    ti.Vector([1.0, 1.0, 1.0])   # scale
)

me1 = Mesh(beginIndex= 0, meshLen= triangle_number, material_coord=(0, 1), worldToLocal=transform)
mesh_buffer = meshListToBuffer([me1])


scene = Scene(quad_buffer=quad_buffer, triangle_buffer=triangle_buffer,bvhNode_buffer=bvhNode_buffer, mesh_buffer=mesh_buffer, lamb_materials=lambertian_materials, metal_materials=metal_materials, diffuseLight_materials=diffuseLight_materials)
