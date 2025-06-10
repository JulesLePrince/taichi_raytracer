import taichi as ti

from constants import PI

@ti.func
def make_translation_matrix(tx: ti.f32, ty: ti.f32, tz: ti.f32):
    return ti.Matrix([
        [1.0, 0.0, 0.0, tx],
        [0.0, 1.0, 0.0, ty],
        [0.0, 0.0, 1.0, tz],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def make_scale_matrix(sx: ti.f32, sy: ti.f32, sz: ti.f32):
    return ti.Matrix([
        [sx,  0.0, 0.0, 0.0],
        [0.0, sy,  0.0, 0.0],
        [0.0, 0.0, sz,  0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def make_rotation_x(angle_degrees: ti.f32):
    angle = angle_degrees * PI / 180.0
    cos_theta = ti.cos(angle)
    sin_theta = ti.sin(angle)
    return ti.Matrix([
        [1.0, 0.0,      0.0,       0.0],
        [0.0, cos_theta, -sin_theta, 0.0],
        [0.0, sin_theta, cos_theta,  0.0],
        [0.0, 0.0,      0.0,       1.0]
    ])

@ti.func
def make_rotation_y(angle_degrees: ti.f32):
    angle = angle_degrees * PI / 180.0
    cos_theta = ti.cos(angle)
    sin_theta = ti.sin(angle)
    return ti.Matrix([
        [cos_theta,  0.0, sin_theta, 0.0],
        [0.0,       1.0, 0.0,      0.0],
        [-sin_theta, 0.0, cos_theta, 0.0],
        [0.0,       0.0, 0.0,      1.0]
    ])

@ti.func
def make_rotation_z(angle_degrees: ti.f32):
    angle = angle_degrees * PI / 180.0
    cos_theta = ti.cos(angle)
    sin_theta = ti.sin(angle)
    return ti.Matrix([
        [cos_theta, -sin_theta, 0.0, 0.0],
        [sin_theta, cos_theta,  0.0, 0.0],
        [0.0,      0.0,       1.0, 0.0],
        [0.0,      0.0,       0.0, 1.0]
    ])

@ti.kernel
def make_transform_mat(translation: ti.types.vector(3, ti.f32),
                      rotation_angles: ti.types.vector(3, ti.f32),  # in degrees
                      scale: ti.types.vector(3, ti.f32)) -> ti.types.matrix(4, 4, ti.f32):
    # Create individual transformation matrices
    T = make_translation_matrix(translation[0], translation[1], translation[2])
    Rx = make_rotation_x(rotation_angles[0])
    Ry = make_rotation_y(rotation_angles[1])
    Rz = make_rotation_z(rotation_angles[2])
    S = make_scale_matrix(scale[0], scale[1], scale[2])

    # Combine them (T * R * S)
    world_matrix = T @ Rz @ Ry @ Rx @ S
    return world_matrix.inverse()
