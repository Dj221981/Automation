"""
File I/O: load and save 3D models in OBJ and STL formats.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import List, Tuple

from .mesh import Mesh
from .material import Material
from .model3d import Model3D
from .math3d import Vector3


# ---------------------------------------------------------------------------
# OBJ
# ---------------------------------------------------------------------------

class OBJLoader:
    """Load Wavefront OBJ files into Mesh objects."""

    def load(self, path: str | Path) -> Mesh:
        """Parse *path* and return a :class:`Mesh`."""
        path = Path(path)
        vertices: List[Vector3] = []
        normals: List[Vector3] = []
        uvs: List[Tuple[float, float]] = []
        faces: List[Tuple[int, int, int]] = []

        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                tag = parts[0]

                if tag == "v":
                    vertices.append(Vector3(float(parts[1]), float(parts[2]), float(parts[3])))
                elif tag == "vn":
                    normals.append(Vector3(float(parts[1]), float(parts[2]), float(parts[3])))
                elif tag == "vt":
                    uvs.append((float(parts[1]), float(parts[2])))
                elif tag == "f":
                    # Triangulate fan (handles quads and polygons)
                    indices = [self._parse_face_vertex(p) for p in parts[1:]]
                    for i in range(1, len(indices) - 1):
                        faces.append((indices[0][0], indices[i][0], indices[i + 1][0]))

        mesh = Mesh(
            vertices=vertices,
            faces=faces,
            normals=normals,
            uvs=uvs,
            name=path.stem,
        )
        if not mesh.normals:
            mesh.compute_smooth_normals()
        return mesh

    @staticmethod
    def _parse_face_vertex(token: str) -> Tuple[int, int, int]:
        """Parse a face vertex token like '1', '1/2', '1/2/3', or '1//3'."""
        parts = token.split("/")
        v_idx = int(parts[0]) - 1
        vt_idx = int(parts[1]) - 1 if len(parts) > 1 and parts[1] else -1
        vn_idx = int(parts[2]) - 1 if len(parts) > 2 and parts[2] else -1
        return (v_idx, vt_idx, vn_idx)


class OBJSaver:
    """Save a :class:`Mesh` as a Wavefront OBJ file."""

    def save(self, mesh: Mesh, path: str | Path) -> None:
        """Write *mesh* to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            fh.write(f"# Exported by model3d framework\n")
            fh.write(f"# Mesh: {mesh.name}\n\n")

            for v in mesh.vertices:
                fh.write(f"v {v.x} {v.y} {v.z}\n")

            if mesh.uvs:
                for u, v in mesh.uvs:
                    fh.write(f"vt {u} {v}\n")

            if mesh.normals:
                for n in mesh.normals:
                    fh.write(f"vn {n.x} {n.y} {n.z}\n")

            fh.write(f"\ng {mesh.name}\n")
            for i0, i1, i2 in mesh.faces:
                fh.write(f"f {i0 + 1} {i1 + 1} {i2 + 1}\n")


# ---------------------------------------------------------------------------
# STL (binary)
# ---------------------------------------------------------------------------

_STL_HEADER_SIZE = 80
_STL_TRIANGLE_SIZE = 50  # 3*float normal + 9*float vertices + 2 attr bytes


class STLLoader:
    """Load binary STL files into Mesh objects."""

    def load(self, path: str | Path) -> Mesh:
        """Parse a binary STL file and return a :class:`Mesh`."""
        path = Path(path)
        with path.open("rb") as fh:
            fh.read(_STL_HEADER_SIZE)  # skip header
            num_triangles = struct.unpack("<I", fh.read(4))[0]

            vertices: List[Vector3] = []
            normals: List[Vector3] = []
            faces: List[Tuple[int, int, int]] = []

            for i in range(num_triangles):
                nx, ny, nz = struct.unpack("<3f", fh.read(12))
                v1 = struct.unpack("<3f", fh.read(12))
                v2 = struct.unpack("<3f", fh.read(12))
                v3 = struct.unpack("<3f", fh.read(12))
                fh.read(2)  # attribute byte count

                base = len(vertices)
                vertices.extend([Vector3(*v1), Vector3(*v2), Vector3(*v3)])
                normals.extend([Vector3(nx, ny, nz)] * 3)
                faces.append((base, base + 1, base + 2))

        return Mesh(
            vertices=vertices,
            faces=faces,
            normals=normals,
            name=path.stem,
        )


class STLSaver:
    """Save a :class:`Mesh` as a binary STL file."""

    def save(self, mesh: Mesh, path: str | Path) -> None:
        """Write *mesh* to *path* in binary STL format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        face_normals = mesh.compute_face_normals()

        with path.open("wb") as fh:
            header = f"model3d {mesh.name}".encode("utf-8")[:_STL_HEADER_SIZE]
            fh.write(header.ljust(_STL_HEADER_SIZE, b"\x00"))
            fh.write(struct.pack("<I", len(mesh.faces)))

            for (i0, i1, i2), fn in zip(mesh.faces, face_normals):
                fh.write(struct.pack("<3f", fn.x, fn.y, fn.z))
                for idx in (i0, i1, i2):
                    v = mesh.vertices[idx]
                    fh.write(struct.pack("<3f", v.x, v.y, v.z))
                fh.write(b"\x00\x00")  # attribute byte count
