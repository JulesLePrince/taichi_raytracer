import taichi as ti
import numpy as np
ti.init(arch=ti.gpu)
from PIL import Image
import os

from fragments.bounce_count_frag import fragment
from utils.gamma_correction import gamma_correction
from models.vector import Vec3, Color

from scenes.la_valse_cornellbox import scene, cam
res = cam.resolution
color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=res)

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


while True :
    paint()
    img = color_buffer.to_numpy() * (1 / number_of_cast)
    max_r = np.rint(np.max(img[:, :, 0])*600)
    mean_r = np.rint(np.mean(img[:, :, 0])*600)
    #img = gamma_correction(img)
    gui.set_image(img)
    #gui.text(content=f"{number_of_cast}", pos=position, font_size=20, color=0xFFFFFF)
    print(f"number of cast : {number_of_cast} | maximum intersection tests : {max_r} | mean intersection tests {mean_r}", end='\r')
    for e in gui.get_events(gui.PRESS):
                if e.key == gui.ESCAPE:
                    gui.running = False
                elif e.key == "s":
                    img = np.rot90(img, k=1)
                    output_image = Image.fromarray((img * 255).astype(np.uint8))
                    output_image.save(f"dragon_{number_of_cast}.png")

    gui.show()
    number_of_cast += 1
