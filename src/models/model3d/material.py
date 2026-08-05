"""
Material: surface appearance properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


# Colour as (R, G, B) floats in [0, 1]
Color = Tuple[float, float, float]


@dataclass
class Material:
    """Describes the visual appearance of a surface.

    Attributes:
        name: Material identifier.
        ambient: Ambient colour (R, G, B).
        diffuse: Diffuse colour (R, G, B).
        specular: Specular colour (R, G, B).
        shininess: Specular shininess exponent (Phong model).
        opacity: Opacity in [0, 1]; 1 = fully opaque.
        texture_path: Optional path to a diffuse texture image.
        normal_map_path: Optional path to a normal-map image.
        emission: Emissive colour (R, G, B).
    """

    name: str = "default"
    ambient: Color = field(default_factory=lambda: (0.2, 0.2, 0.2))
    diffuse: Color = field(default_factory=lambda: (0.8, 0.8, 0.8))
    specular: Color = field(default_factory=lambda: (1.0, 1.0, 1.0))
    shininess: float = 32.0
    opacity: float = 1.0
    texture_path: Optional[str] = None
    normal_map_path: Optional[str] = None
    emission: Color = field(default_factory=lambda: (0.0, 0.0, 0.0))

    # ------------------------------------------------------------------
    # Convenience factories
    # ------------------------------------------------------------------

    @staticmethod
    def default() -> "Material":
        return Material()

    @staticmethod
    def from_color(r: float, g: float, b: float, name: str = "color") -> "Material":
        """Create a simple flat-shaded material from a single colour."""
        return Material(
            name=name,
            ambient=(r * 0.2, g * 0.2, b * 0.2),
            diffuse=(r, g, b),
            specular=(1.0, 1.0, 1.0),
        )

    @staticmethod
    def matte(r: float, g: float, b: float, name: str = "matte") -> "Material":
        """Create a matte (no specular) material."""
        return Material(
            name=name,
            ambient=(r * 0.1, g * 0.1, b * 0.1),
            diffuse=(r, g, b),
            specular=(0.0, 0.0, 0.0),
            shininess=1.0,
        )

    @staticmethod
    def metallic(r: float, g: float, b: float, name: str = "metal") -> "Material":
        """Create a high-specular metallic material."""
        return Material(
            name=name,
            ambient=(r * 0.05, g * 0.05, b * 0.05),
            diffuse=(r, g, b),
            specular=(1.0, 1.0, 1.0),
            shininess=128.0,
        )

    def __repr__(self) -> str:
        return (
            f"Material(name={self.name!r}, diffuse={self.diffuse}, shininess={self.shininess})"
        )
