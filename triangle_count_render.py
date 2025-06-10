import taichi as ti
import numpy as np
ti.init(arch=ti.gpu)

from fragments.triangle_tested_frag import fragment
from utils.gamma_correction import gamma_correction
from models.vector import Vec3, Color

from scenes.nike_of_samothrace_cornellbox import scene, cam
res = cam.resolution
color_buffer = ti.Vector.field(1, dtype=ti.f32, shape=res)

@ti.kernel
def paint():
    x_len, y_len = res[0], res[1]
    for u, v in color_buffer:
        ray = cam.get_ray(u, v)
        new_color = fragment(ray, u, v, res, scene)
        color_buffer[u, v] += new_color

number_of_cast = 1
position = [0.01, 0.99]
gui = ti.GUI("RayTracer", res)

while True <= 20:
    for e in gui.get_events(gui.PRESS):
                if e.key == gui.ESCAPE:
                    gui.running = False
    paint()
    img = color_buffer.to_numpy()
    ma = np.amax(img)
    img = img/number_of_cast
    img = np.where(img > 30, img, 0)
    gui.set_image(img)
    #gui.text(content=f"{number_of_cast}", pos=position, font_size=20, color=0xFFFFFF)
    print(f"number of sample calculated : {number_of_cast}", end='\r')
    gui.show()
    number_of_cast += 1
