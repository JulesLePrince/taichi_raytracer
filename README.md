# Taichi Raytracer

A GPU-accelerated raytracer implementation using the Taichi programming language, featuring real-time rendering with Bounding Volume Hierarchy (BVH) acceleration and multiple material types.

## Features

- **GPU Acceleration**: Leverages Taichi's GPU computing capabilities for fast parallel ray tracing
- **BVH Acceleration**: Efficient ray-triangle intersection using Bounding Volume Hierarchy
- **Multiple Material Types**: 
  - Lambertian (diffuse) materials
  - Metal materials with configurable roughness
  - Emissive/light materials
- **Complex Geometry Support**: 
  - Sphere primitives
  - Quad primitives  
  - Triangle mesh loading from .obj files
- **Real-time Rendering**: Interactive viewport with progressive accumulation
- **Cornell Box Scenes**: Classic computer graphics test scenes
- **HDRI Environment Maps**: Support for HDR environment lighting

## Requirements

- Python 3.7+
- Taichi (`pip install taichi`)
- NumPy
- PIL/Pillow
- PyVista (for mesh processing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JulesLePrince/taichi_raytracer.git
cd taichi_raytracer
```

2. Install dependencies:
```bash
pip install taichi numpy pillow pyvista
```

## Usage

### Basic Rendering

Run the main raytracer with the default scene:
```bash
python render.py
```

### Controls

- **ESC**: Exit the application
- **S**: Save current frame as PNG image
- The renderer will progressively accumulate samples for better quality over time

### Scene Configuration

The current scene is configured in `scenes/la_valse_cornellbox.py`, featuring:
- A Cornell box setup with colored walls
- The "La Valse" 3D model as the main subject
- Configurable camera position and field of view
- Multiple material types showcasing different shading models

## Project Structure

```
├── render.py                 # Main rendering loop
├── scenes/
│   └── la_valse_cornellbox.py # Scene definition
├── models/                   # Core rendering primitives
│   ├── camera.py            # Camera implementation
│   ├── scene.py             # Scene container
│   ├── sphere.py            # Sphere primitive
│   ├── quad.py              # Quad primitive
│   ├── triangle.py          # Triangle primitive
│   ├── mesh.py              # BVH mesh loader
│   ├── ray.py               # Ray definition
│   ├── hit.py               # Ray-surface intersection
│   └── vector.py            # Vector math utilities
├── fragments/               # Shader-like rendering fragments
│   ├── raytrace_frag.py     # Main ray tracing logic
│   ├── color_frag.py        # Color accumulation
│   ├── normal_frag.py       # Normal visualization
│   └── distance_frag.py     # Distance visualization
├── environments/            # Environment lighting
│   ├── hdri_env.py          # HDRI environment maps
│   └── simple_sky.py        # Procedural sky
├── bvh/                     # BVH acceleration structure
│   ├── classes.py           # BVH node definitions
│   └── obj_to_triangles.py  # Mesh processing
├── utils/                   # Utility functions
│   ├── gamma_correction.py  # Post-processing
│   └── make_matrix.py       # Matrix operations
├── meshes/                  # 3D model files
│   ├── la_valse_nodes.npy   # BVH nodes for La Valse model
│   └── la_valse_triangle.npy # Triangle data for La Valse model
└── material.py              # Material definitions
```

## Technical Details

### Ray Tracing Pipeline

1. **Ray Generation**: Camera generates primary rays for each pixel
2. **BVH Traversal**: Efficient ray-scene intersection using bounding volume hierarchy
3. **Material Shading**: Surface interaction based on material properties
4. **Light Transport**: Recursive ray bouncing for global illumination
5. **Accumulation**: Progressive sampling for noise reduction

### Performance Features

- **GPU Parallelization**: All computations run on GPU using Taichi
- **BVH Acceleration**: O(log n) ray-triangle intersection complexity
- **Memory Efficient**: Optimized data structures for GPU memory layout
- **Real-time Feedback**: Interactive preview with progressive refinement

## Customization

### Adding New Scenes

1. Create a new file in `scenes/` directory
2. Define camera, materials, and geometry
3. Update the import in `render.py`

### Material Types

- **Lambertian**: Perfect diffuse reflection
- **Metal**: Specular reflection with optional roughness
- **DiffuseLight**: Emissive materials for lighting

### Mesh Loading

To use custom 3D models:
1. Convert .obj files using the BVH utilities
2. Generate .npy files for nodes and triangles
3. Load in scene definition

## Performance Notes

- Rendering speed depends on GPU capability
- Scene complexity affects performance (triangle count, material complexity)
- Progressive accumulation provides better quality over time
- BVH acceleration significantly improves performance for complex meshes

## License

This project is open source. Please check the repository for specific license terms.

## Contributing

Contributions are welcome! Areas for improvement:
- Additional material types (glass, subsurface scattering)
- Advanced lighting features (area lights, volumetrics)
- Post-processing effects
- Performance optimizations
- Additional primitive types

## Acknowledgments

- Built with [Taichi](https://github.com/taichi-dev/taichi) for GPU acceleration
- Inspired by Peter Shirley's "Ray Tracing in One Weekend" series
- Cornell Box scene for lighting validation

