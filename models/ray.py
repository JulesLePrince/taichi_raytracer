import taichi as ti

@ti.dataclass
class Ray:
    origin: ti.types.vector(3, ti.f32)
    direction: ti.types.vector(3, ti.f32)

    @ti.func
    def at(self, t):
        return self.origin + t * self.direction
