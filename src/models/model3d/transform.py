"""
Transform: encapsulates position, rotation (quaternion), and scale.
"""

from __future__ import annotations

from .math3d import Vector3, Matrix4, Quaternion


class Transform:
    """Represents a 3D transformation (TRS: translate, rotate, scale)."""

    def __init__(
        self,
        position: Vector3 | None = None,
        rotation: Quaternion | None = None,
        scale: Vector3 | None = None,
    ) -> None:
        self.position: Vector3 = position if position is not None else Vector3.zero()
        self.rotation: Quaternion = rotation if rotation is not None else Quaternion.identity()
        self.scale: Vector3 = scale if scale is not None else Vector3.one()

    # ------------------------------------------------------------------
    # Matrix composition
    # ------------------------------------------------------------------

    def to_matrix(self) -> Matrix4:
        """Compose TRS into a single 4x4 matrix (T * R * S)."""
        t = Matrix4.translation(self.position)
        r = self.rotation.to_matrix4()
        s = Matrix4.scale(self.scale)
        return t * r * s

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def translate(self, delta: Vector3) -> None:
        self.position = self.position + delta

    def rotate(self, rotation: Quaternion) -> None:
        self.rotation = (rotation * self.rotation).normalized()

    def scale_by(self, factor: Vector3) -> None:
        self.scale = Vector3(
            self.scale.x * factor.x,
            self.scale.y * factor.y,
            self.scale.z * factor.z,
        )

    def forward(self) -> Vector3:
        """Return the local forward direction (-Z axis) in world space."""
        return self.rotation.rotate_vector(Vector3.forward())

    def right(self) -> Vector3:
        """Return the local right direction (+X axis) in world space."""
        return self.rotation.rotate_vector(Vector3.right())

    def up(self) -> Vector3:
        """Return the local up direction (+Y axis) in world space."""
        return self.rotation.rotate_vector(Vector3.up())

    def __repr__(self) -> str:
        return (
            f"Transform(position={self.position}, rotation={self.rotation}, scale={self.scale})"
        )
