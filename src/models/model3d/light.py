"""
Light: ambient, directional, and point light sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

from .math3d import Vector3

Color = Tuple[float, float, float]


class LightType(Enum):
    AMBIENT = "ambient"
    DIRECTIONAL = "directional"
    POINT = "point"
    SPOT = "spot"


@dataclass
class Light:
    """A light source in the scene.

    Attributes:
        name: Light identifier.
        light_type: Variant of this light.
        color: Light colour (R, G, B) in [0, 1].
        intensity: Brightness multiplier.
        position: World position (used by point and spot lights).
        direction: Normalised direction (used by directional and spot lights).
        range: Maximum distance for point/spot lights (0 = unlimited).
        spot_angle: Half-angle in degrees for spot lights.
        cast_shadows: Whether the light should cast shadows.
    """

    name: str = "light"
    light_type: LightType = LightType.POINT
    color: Color = field(default_factory=lambda: (1.0, 1.0, 1.0))
    intensity: float = 1.0
    position: Vector3 = field(default_factory=Vector3.zero)
    direction: Vector3 = field(default_factory=lambda: Vector3(0, -1, 0))
    range: float = 0.0
    spot_angle: float = 30.0
    cast_shadows: bool = False

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @staticmethod
    def ambient(color: Color = (1.0, 1.0, 1.0), intensity: float = 0.2) -> "Light":
        return Light(name="ambient", light_type=LightType.AMBIENT, color=color, intensity=intensity)

    @staticmethod
    def directional(
        direction: Vector3,
        color: Color = (1.0, 1.0, 1.0),
        intensity: float = 1.0,
    ) -> "Light":
        return Light(
            name="directional",
            light_type=LightType.DIRECTIONAL,
            color=color,
            intensity=intensity,
            direction=direction.normalized(),
        )

    @staticmethod
    def point(
        position: Vector3,
        color: Color = (1.0, 1.0, 1.0),
        intensity: float = 1.0,
        range: float = 10.0,
    ) -> "Light":
        return Light(
            name="point",
            light_type=LightType.POINT,
            color=color,
            intensity=intensity,
            position=position,
            range=range,
        )

    def __repr__(self) -> str:
        return (
            f"Light(name={self.name!r}, type={self.light_type.value}, "
            f"intensity={self.intensity})"
        )
