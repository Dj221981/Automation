"""
Camera: view/projection management for a 3D scene.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .math3d import Vector3, Matrix4, Ray


@dataclass
class Camera:
    """Perspective or orthographic camera.

    Attributes:
        position: World-space camera position.
        target: Look-at target position.
        up: World up vector.
        fov_y: Vertical field of view in degrees (perspective only).
        aspect: Viewport width / height ratio.
        near: Near clipping plane distance.
        far: Far clipping plane distance.
        orthographic: If True use orthographic projection.
        ortho_size: Half-height of the orthographic frustum.
    """

    position: Vector3 = field(default_factory=lambda: Vector3(0, 0, 5))
    target: Vector3 = field(default_factory=Vector3.zero)
    up: Vector3 = field(default_factory=Vector3.up)
    fov_y: float = 60.0
    aspect: float = 16 / 9
    near: float = 0.1
    far: float = 1000.0
    orthographic: bool = False
    ortho_size: float = 5.0

    # ------------------------------------------------------------------
    # Matrix builders
    # ------------------------------------------------------------------

    def view_matrix(self) -> Matrix4:
        """Build the view (world-to-camera) matrix."""
        return Matrix4.look_at(self.position, self.target, self.up)

    def projection_matrix(self) -> Matrix4:
        """Build the projection matrix."""
        if self.orthographic:
            return self._ortho_matrix()
        return Matrix4.perspective(
            math.radians(self.fov_y), self.aspect, self.near, self.far
        )

    def _ortho_matrix(self) -> Matrix4:
        """Build an orthographic projection matrix."""
        h = self.ortho_size
        w = h * self.aspect
        n, f = self.near, self.far
        return Matrix4([
            1 / w, 0,     0,                0,
            0,     1 / h, 0,                0,
            0,     0,     -2 / (f - n),    -(f + n) / (f - n),
            0,     0,     0,                1,
        ])

    # ------------------------------------------------------------------
    # Ray casting
    # ------------------------------------------------------------------

    def ray_from_screen(self, ndc_x: float, ndc_y: float) -> Ray:
        """Return a world-space ray from normalised device coordinates [-1, 1]."""
        fov_rad = math.radians(self.fov_y)
        tan_half = math.tan(fov_rad / 2)
        forward = (self.target - self.position).normalized()
        right = forward.cross(self.up).normalized()
        real_up = right.cross(forward)

        ray_dir = (
            forward
            + right * (ndc_x * tan_half * self.aspect)
            + real_up * (ndc_y * tan_half)
        ).normalized()
        return Ray(origin=self.position, direction=ray_dir)

    # ------------------------------------------------------------------
    # Movement helpers
    # ------------------------------------------------------------------

    def look_at(self, target: Vector3) -> None:
        self.target = target

    def move(self, delta: Vector3) -> None:
        self.position = self.position + delta
        self.target = self.target + delta

    def zoom(self, factor: float) -> None:
        """Move the camera closer to / farther from the target."""
        direction = (self.target - self.position).normalized()
        self.position = self.position + direction * factor

    def __repr__(self) -> str:
        return (
            f"Camera(position={self.position}, target={self.target}, "
            f"fov_y={self.fov_y}, aspect={self.aspect:.2f})"
        )
