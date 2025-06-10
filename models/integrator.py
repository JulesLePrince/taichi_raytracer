import taichi as ti
from constants import *
from material import Lambertian
from models.hit import HitInfo
from .vector import Vec3, Color
from utils.alea import randomHemisphereDirection
from environments.simple_sky import simpleSkyEnv
from environments.hdri_env import hdr_background

@ti.func
def trace(ray, scene):
    current_attenuation = Color(1., 1., 1.)
    accumulated_emission = Color(0., 0., 0.)
    nb_triangles_tested = 0
    nb_boxes_tested = 0

    for k in range(MAX_BOUNCE):
        hitInfos, tc, bc = scene.hit(ray, k)
        nb_triangles_tested += tc
        nb_boxes_tested += bc
        if hitInfos.didHit and hitInfos.dst >= 0.0 :
            ray.origin = hitInfos.hitPoint
            mat_id, mat_num = hitInfos.material_coord
            attenuation = Color(1., 1., 1.)

            if mat_id == 0: # Lambertian
                mat = scene.lambertian_materials[mat_num]
                accumulated_emission += current_attenuation*mat.emmited()
                attenuation, ray, bounce = mat.scatter(ray, hitInfos)
                if not bounce:
                    break
            if mat_id == 1: # Metal
                mat = scene.metal_materials[mat_num]
                accumulated_emission += current_attenuation*mat.emmited()
                attenuation, ray, bounce = mat.scatter(ray, hitInfos)
                if not bounce:
                    break
            if mat_id == 2: # DiffuseLight
                mat = scene.diffuseLight_materials[mat_num]
                accumulated_emission += current_attenuation*mat.emmited()
                attenuation, ray, bounce = mat.scatter(ray, hitInfos)
                if not bounce:
                    break

            if mat_id == 3: # DiffuseLight
                mat = scene.dielectric_materials[mat_num]
                accumulated_emission += current_attenuation*mat.emmited()
                attenuation, ray, bounce = mat.scatter(ray, hitInfos)
                if not bounce:
                    break

            current_attenuation *= attenuation
        else:
            temp = Color(0., 0., 0.)
            #temp += simpleSkyEnv(ray)
            accumulated_emission += current_attenuation * temp
            break
    return accumulated_emission, nb_triangles_tested, nb_boxes_tested
