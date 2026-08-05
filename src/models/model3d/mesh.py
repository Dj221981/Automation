"""
Mesh: geometric data container (vertices, faces, normals, UVs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .math3d import Vector3, BoundingBox


@dataclass
class Mesh:
    """Represents a polygon mesh.

    Attributes:
        vertices: List of vertex positions.
        faces: List of triangular faces, each a tuple of three vertex indices.
        normals: Per-vertex normals (optional).
        uvs: Per-vertex UV coordinates as (u, v) tuples (optional).
        name: Human-readable mesh name.
    """

    vertices: List[Vector3] = field(default_factory=list)
    faces: List[Tuple[int, int, int]] = field(default_factory=list)
    normals: List[Vector3] = field(default_factory=list)
    uvs: List[Tuple[float, float]] = field(default_factory=list)
    name: str = "mesh"

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    def bounding_box(self) -> BoundingBox:
        """Compute the axis-aligned bounding box of this mesh."""
        if not self.vertices:
            return BoundingBox()
        return BoundingBox.from_points(self.vertices)

    # ------------------------------------------------------------------
    # Normal computation
    # ------------------------------------------------------------------

    def compute_face_normals(self) -> List[Vector3]:
        """Return a flat normal per face."""
        face_normals: List[Vector3] = []
        for i0, i1, i2 in self.faces:
            v0, v1, v2 = self.vertices[i0], self.vertices[i1], self.vertices[i2]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = edge1.cross(edge2).normalized()
            face_normals.append(normal)
        return face_normals

    def compute_smooth_normals(self) -> None:
        """Compute and store averaged per-vertex normals."""
        accum = [Vector3(0, 0, 0) for _ in self.vertices]
        face_normals = self.compute_face_normals()
        for (i0, i1, i2), fn in zip(self.faces, face_normals):
            for idx in (i0, i1, i2):
                accum[idx] = accum[idx] + fn
        self.normals = [v.normalized() for v in accum]

    # ------------------------------------------------------------------
    # Primitive factories
    # ------------------------------------------------------------------

    @staticmethod
    def create_cube(size: float = 1.0) -> "Mesh":
        """Create a unit cube mesh."""
        h = size / 2
        vertices = [
            Vector3(-h, -h, -h), Vector3( h, -h, -h),
            Vector3( h,  h, -h), Vector3(-h,  h, -h),
            Vector3(-h, -h,  h), Vector3( h, -h,  h),
            Vector3( h,  h,  h), Vector3(-h,  h,  h),
        ]
        faces = [
            (0, 2, 1), (0, 3, 2),  # back
            (4, 5, 6), (4, 6, 7),  # front
            (0, 1, 5), (0, 5, 4),  # bottom
            (2, 3, 7), (2, 7, 6),  # top
            (0, 4, 7), (0, 7, 3),  # left
            (1, 2, 6), (1, 6, 5),  # right
        ]
        mesh = Mesh(vertices=vertices, faces=faces, name="cube")
        mesh.compute_smooth_normals()
        return mesh

    @staticmethod
    def create_plane(width: float = 1.0, depth: float = 1.0) -> "Mesh":
        """Create a flat quad mesh in the XZ plane."""
        hw, hd = width / 2, depth / 2
        vertices = [
            Vector3(-hw, 0, -hd),
            Vector3( hw, 0, -hd),
            Vector3( hw, 0,  hd),
            Vector3(-hw, 0,  hd),
        ]
        faces = [(0, 2, 1), (0, 3, 2)]
        uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        normals = [Vector3(0, 1, 0)] * 4
        return Mesh(vertices=vertices, faces=faces, normals=normals, uvs=uvs, name="plane")

    @staticmethod
    def create_sphere(radius: float = 1.0, segments: int = 16, rings: int = 8) -> "Mesh":
        """Create a UV sphere mesh."""
        import math
        vertices: List[Vector3] = []
        uvs: List[Tuple[float, float]] = []

        for ring in range(rings + 1):
            phi = math.pi * ring / rings
            for seg in range(segments + 1):
                theta = 2 * math.pi * seg / segments
                x = radius * math.sin(phi) * math.cos(theta)
                y = radius * math.cos(phi)
                z = radius * math.sin(phi) * math.sin(theta)
                vertices.append(Vector3(x, y, z))
                uvs.append((seg / segments, ring / rings))

        faces: List[Tuple[int, int, int]] = []
        row = segments + 1
        for ring in range(rings):
            for seg in range(segments):
                a = ring * row + seg
                b = a + 1
                c = a + row
                d = c + 1
                faces.append((a, c, b))
                faces.append((b, c, d))

        mesh = Mesh(vertices=vertices, faces=faces, uvs=uvs, name="sphere")
        mesh.compute_smooth_normals()
        return mesh

    def __repr__(self) -> str:
        return f"Mesh(name={self.name!r}, vertices={self.vertex_count}, faces={self.face_count})"
