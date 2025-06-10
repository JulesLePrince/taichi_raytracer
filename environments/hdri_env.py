import taichi as ti
from taichi.math import normalize
from models.vector import Color
import numpy as np
import imageio



# Load HDRI image from file and initialize Taichi field
def load_hdri_image_from_file(file_path):
    # Load HDRI image using imageio
    hdr_data = imageio.imread(file_path)/255
    height, width, _ = hdr_data.shape
    hdr_data = hdr_data.astype(np.float32)  # Load as float32 for HDR values

    # Define the Taichi field size based on image dimensions
    hdr_image_shape = (height, width)
    hdr_image = Color.field(shape=hdr_image_shape)

    hdr_image.from_numpy(hdr_data)
    return hdr_image



@ti.func
def hdr_background(ray, hdr_image):
    # Normalize the input direction
    ray_dir = normalize(ray.direction)
    x, y, z = ray_dir[0], ray_dir[1], ray_dir[2]

    # Convert to spherical coordinates
    phi = ti.atan2(z, x)  # azimuth (-π to π)
    theta = ti.acos(y)     # altitude (0 to π)

    # Normalize spherical coordinates to [0, 1]
    u = (phi + np.pi) / (2 * np.pi)   # Map phi from [-π, π] to [0, 1]
    v = theta / np.pi                 # Map theta from [0, π] to [0, 1]

    # Get the image dimensions dynamically
    width, height = hdr_image.shape[1], hdr_image.shape[0]

    # Map [0, 1] to image coordinates
    px = int(u * width)
    py = int(v * height)

    # Handle wrapping around horizontally
    px = px % width  # Wrap horizontally
    py = max(0, min(height - 1, py))  # Clamp vertically

    # Return the HDRI color at that pixel
    return hdr_image[py, px]
