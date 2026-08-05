"""
3D Model Framework for the Automation repository.

Provides core 3D classes, scene management, file I/O, and rendering pipeline basics.
"""

from .math3d import Vector3, Matrix4, Quaternion, BoundingBox, Ray
from .transform import Transform
from .mesh import Mesh
from .material import Material
from .camera import Camera
from .model3d import Model3D
from .light import Light
from .scene import Scene
from .scene_graph import SceneGraph
from .file_io import OBJLoader, OBJSaver, STLLoader, STLSaver
from .renderer import RenderQueue, Shader, CullingSystem

__all__ = [
    "Vector3",
    "Matrix4",
    "Quaternion",
    "BoundingBox",
    "Ray",
    "Transform",
    "Mesh",
    "Material",
    "Camera",
    "Model3D",
    "Light",
    "Scene",
    "SceneGraph",
    "OBJLoader",
    "OBJSaver",
    "STLLoader",
    "STLSaver",
    "RenderQueue",
    "Shader",
    "CullingSystem",
]
