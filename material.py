import taichi as ti
from taichi.math import dot, normalize, sqrt, length
from models.vector import Point, Vec3, Color
from models.ray import Ray
from utils.alea import randomHemisphereDirection
from utils.alea import randomDirection


class _material:
    @ti.func
    def emmited(self):
        return Color(0., 0., 0.)

    @ti.func
    def scatter(self, ray, hitRec):
        return Color(0., 0., 0.), Ray(), False

# ID : 0
@ti.dataclass
class Lambertian(_material):
    albedo: Vec3

    @ti.func
    def scatter(self, ray, hitRec):
        attenuation = self.albedo
        scatter_dir = hitRec.normal/4 + randomHemisphereDirection(hitRec.normal)
        return attenuation, Ray(hitRec.hitPoint, scatter_dir), True


@ti.func
def reflect(v, normal):
    return v - 2*dot(v, normal)*normal

# ID : 1
@ti.dataclass
class Metal(_material):
    albedo: Vec3
    fuzz: ti.f32

    @ti.func
    def scatter(self, ray, hitRec):
        bounce = True
        reflected = normalize(reflect(ray.direction, hitRec.normal)) + (self.fuzz*randomDirection())
        attenuation = self.albedo
        scattered = Ray(hitRec.hitPoint, reflected)
        if dot(scattered.direction, hitRec.normal) < 0:
            attenuation = Color(0., 0., 0.)
            bounce = False
        return attenuation, scattered, bounce

# ID : 2
@ti.dataclass
class DiffuseLight(_material):
    emmissionColor: Vec3

    @ti.func
    def emmited(self):
        return self.emmissionColor




# ID : 3

@ti.func
def refract(uv, n, etaiOverEtat):
    cosTheta = min(dot(-uv, n), 1.0)
    rOutPerp = etaiOverEtat * (uv + cosTheta * n)
    rOutParallel = -ti.sqrt(abs(1.0 - rOutPerp.dot(rOutPerp))) * n
    return rOutPerp + rOutParallel


# Use Schlick's approximation for reflectance
@ti.func
def reflectance(cosine, ref_idx):
    r0 = (1.0 - ref_idx) / (1.0 + ref_idx)
    r0 = r0 * r0
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5)

@ti.dataclass
class Dielectric(_material):
    ir: ti.f32  # Index of Refraction

    @ti.func
    def scatter(self, ray, hitRec):
        bounce = True
        attenuation = Color(1.0, 1.0, 1.0)  # Glass surface absorbs nothing

        refractionRatio = 1.0/self.ir if hitRec.frontFace else self.ir

        unitDirection = normalize(ray.direction)
        cosTheta = min(dot(-unitDirection, hitRec.normal), 1.0)
        sinTheta = ti.sqrt(1.0 - cosTheta * cosTheta)

        cannotRefract = refractionRatio * sinTheta > 1.0
        direction = Vec3(0.0, 0.0, 0.0)

        if cannotRefract or reflectance(cosTheta, refractionRatio) > ti.random():
            direction = reflect(unitDirection, hitRec.normal)
        else:
            direction = refract(unitDirection, hitRec.normal, refractionRatio)

        scattered = Ray(hitRec.hitPoint, direction)
        return attenuation, scattered, bounce
