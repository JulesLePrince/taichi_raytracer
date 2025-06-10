import taichi as ti
from models.vector import Color, Vec3
from taichi.math import dot, smoothstep, mix

skyColorHorizon = Color(1, .5, .5)
skyColorZenith = Color(0.64, 0.7, 1.)

sunLightDirection = Vec3(-10., -100, -10.)
sunFocus = 5
sunIntensity = 10

groundColor = Color(0.5, 0.5, 0.5)

@ti.func
def simpleSkyEnv(ray):
    skyGradientT = smoothstep(0, 0.4, ray.direction.y) ** 0.35
    skyGradient = mix(skyColorHorizon, skyColorZenith, skyGradientT);
    sun = (max(0, dot(ray.direction, -sunLightDirection)))**sunFocus * sunIntensity

    groundToSkyT = smoothstep(-0.01, 0, ray.direction.y)
    sunMask = groundToSkyT >= 1;
    return mix(groundColor, skyGradient, groundToSkyT)
