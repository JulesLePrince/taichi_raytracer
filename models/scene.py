from numpy._core.fromnumeric import shape
import taichi as ti
from taichi.lang.struct import dataclass
from models.hit import HitInfo
from models.sphere import Sphere
from models.quad import Quad
from models.triangle import Triangle
from models.mesh import BVHNode, Mesh
from models.vector import Vec3, Point, Color
from material import DiffuseLight, Lambertian, Metal, Dielectric
from constants import *



hdr_image = Color.field(shape=3)
emptyTriangleBuffer = Triangle.field(shape=1)
emptyQuadBuffer = Quad.field(shape=1)
emptySphereBuffer = Sphere.field(shape=1)
emptyMeshBuffer = Mesh.field(shape=1)
emptyBVHNodeBuffer = BVHNode.field(shape=1)

@ti.data_oriented
class Scene:
    def __init__(self, sphere_buffer=emptySphereBuffer, quad_buffer=emptyQuadBuffer, triangle_buffer=emptyTriangleBuffer, bvhNode_buffer=emptyBVHNodeBuffer, mesh_buffer=emptyMeshBuffer, lamb_materials=[], metal_materials=[], diffuseLight_materials=[], dielectric_materials=[], hdr_image=hdr_image) -> None:
        self.lamb_materials_number = len(lamb_materials)
        self.lambertian_materials = Lambertian.field(shape=self.lamb_materials_number+1)
        for i in range(self.lamb_materials_number):
            self.lambertian_materials[i] = lamb_materials[i]

        self.metal_materials_number = len(metal_materials)
        self.metal_materials = Metal.field(shape=self.metal_materials_number+1)
        for i in range(self.metal_materials_number):
            self.metal_materials[i] = metal_materials[i]

        self.diffuseLight_materials_number = len(diffuseLight_materials)
        self.diffuseLight_materials = DiffuseLight.field(shape=self.diffuseLight_materials_number+1)
        for i in range(self.diffuseLight_materials_number):
            self.diffuseLight_materials[i] = diffuseLight_materials[i]

        self.dielectric_materials_number = len(dielectric_materials)
        self.dielectric_materials = Dielectric.field(shape=self.dielectric_materials_number+1)
        for i in range(self.dielectric_materials_number):
            self.dielectric_materials[i] = dielectric_materials[i]

        self.sphere_number = sphere_buffer.shape[0]
        self.sphere_buffer = sphere_buffer

        self.quad_number = quad_buffer.shape[0]
        self.quad_buffer = quad_buffer

        self.triangle_number = triangle_buffer.shape[0]
        self.triangle_buffer = triangle_buffer

        self.bvhNode_number = bvhNode_buffer.shape[0]
        self.bvhNode_buffer = bvhNode_buffer

        self.mesh_number = mesh_buffer.shape[0]
        self.mesh_buffer = mesh_buffer

        self.hdr_image = hdr_image

    @ti.func
    def hit(self, ray, k):
        closest_hit = HitInfo(didHit=False, dst=MAX_LEN)
        triangle_count = 0
        box_count = 0
        # Spheres
        for index in range(self.sphere_number):
            intersect = self.sphere_buffer[index].hit(ray)
            if intersect.dst < closest_hit.dst and intersect.didHit:
                closest_hit = intersect

        tmp = 0
        if k == 0:
            tmp = 1

        for index in range(self.quad_number-1-tmp):
            intersect = self.quad_buffer[index].hit(ray)
            if intersect.dst < closest_hit.dst and intersect.didHit:
                closest_hit = intersect


        for index in range(self.mesh_number-1):
            intersect, tc, bc = self.mesh_buffer[index].hit(ray, self.triangle_buffer, self.bvhNode_buffer)
            triangle_count += tc
            box_count += bc
            if intersect.dst < closest_hit.dst and intersect.didHit:
                closest_hit = intersect
                closest_hit.material_coord = self.mesh_buffer[0].material_coord

        return closest_hit, triangle_count, box_count
