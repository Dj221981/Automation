"""
Model3D: base class combining a Mesh, Material, and Transform.
"""

from __future__ import annotations

from typing import List, Optional
from .transform import Transform
from .mesh import Mesh
from .material import Material
from .math3d import BoundingBox, Matrix4


class Model3D:
    """A 3D model composed of a mesh, a material, and a transform.

    Attributes:
        name: Human-readable model name.
        mesh: Geometric data.
        material: Surface appearance.
        transform: World-space position/rotation/scale.
        visible: Whether the model should be rendered.
        tags: Arbitrary string tags for grouping/filtering.
    """

    def __init__(
        self,
        name: str = "model",
        mesh: Optional[Mesh] = None,
        material: Optional[Material] = None,
        transform: Optional[Transform] = None,
    ) -> None:
        self.name: str = name
        self.mesh: Mesh = mesh if mesh is not None else Mesh()
        self.material: Material = material if material is not None else Material()
        self.transform: Transform = transform if transform is not None else Transform()
        self.visible: bool = True
        self.tags: List[str] = []

    # ------------------------------------------------------------------
    # Spatial queries
    # ------------------------------------------------------------------

    def world_bounding_box(self) -> BoundingBox:
        """Return the axis-aligned bounding box in world space."""
        local_bb = self.mesh.bounding_box()
        if not local_bb.is_valid() or not self.mesh.vertices:
            return BoundingBox()
        mat = self.transform.to_matrix()
        transformed_pts = [mat.transform_point(v) for v in self.mesh.vertices]
        return BoundingBox.from_points(transformed_pts)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a plain Python dictionary."""
        return {
            "name": self.name,
            "visible": self.visible,
            "tags": list(self.tags),
            "transform": {
                "position": self.transform.position.to_tuple(),
                "rotation": (
                    self.transform.rotation.w,
                    self.transform.rotation.x,
                    self.transform.rotation.y,
                    self.transform.rotation.z,
                ),
                "scale": self.transform.scale.to_tuple(),
            },
            "mesh": {
                "name": self.mesh.name,
                "vertices": [v.to_tuple() for v in self.mesh.vertices],
                "faces": list(self.mesh.faces),
                "normals": [n.to_tuple() for n in self.mesh.normals],
                "uvs": list(self.mesh.uvs),
            },
            "material": {
                "name": self.material.name,
                "ambient": list(self.material.ambient),
                "diffuse": list(self.material.diffuse),
                "specular": list(self.material.specular),
                "shininess": self.material.shininess,
                "opacity": self.material.opacity,
                "texture_path": self.material.texture_path,
                "emission": list(self.material.emission),
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Model3D":
        """Deserialise from a plain Python dictionary."""
        from .math3d import Vector3, Quaternion

        td = data.get("transform", {})
        pos = Vector3(*td.get("position", (0, 0, 0)))
        rot_data = td.get("rotation", (1, 0, 0, 0))
        rot = Quaternion(*rot_data)
        scl = Vector3(*td.get("scale", (1, 1, 1)))

        md = data.get("mesh", {})
        vertices = [Vector3(*v) for v in md.get("vertices", [])]
        faces = [tuple(f) for f in md.get("faces", [])]
        normals = [Vector3(*n) for n in md.get("normals", [])]
        uvs = [tuple(u) for u in md.get("uvs", [])]
        mesh = Mesh(
            vertices=vertices,
            faces=faces,
            normals=normals,
            uvs=uvs,
            name=md.get("name", "mesh"),
        )

        mat_d = data.get("material", {})
        material = Material(
            name=mat_d.get("name", "default"),
            ambient=tuple(mat_d.get("ambient", [0.2, 0.2, 0.2])),
            diffuse=tuple(mat_d.get("diffuse", [0.8, 0.8, 0.8])),
            specular=tuple(mat_d.get("specular", [1.0, 1.0, 1.0])),
            shininess=mat_d.get("shininess", 32.0),
            opacity=mat_d.get("opacity", 1.0),
            texture_path=mat_d.get("texture_path"),
            emission=tuple(mat_d.get("emission", [0.0, 0.0, 0.0])),
        )

        model = cls(
            name=data.get("name", "model"),
            mesh=mesh,
            material=material,
            transform=Transform(position=pos, rotation=rot, scale=scl),
        )
        model.visible = data.get("visible", True)
        model.tags = data.get("tags", [])
        return model

    def __repr__(self) -> str:
        return (
            f"Model3D(name={self.name!r}, mesh={self.mesh}, visible={self.visible})"
        )
