"""
Rendering pipeline basics: RenderQueue, Shader, CullingSystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .model3d import Model3D
from .camera import Camera
from .math3d import Vector3


class RenderLayer(Enum):
    OPAQUE = 0
    TRANSPARENT = 1
    OVERLAY = 2


@dataclass(order=True)
class RenderItem:
    """A queued draw call.

    Sorting is by (layer, depth) so the queue can be iterated in
    the correct render order.
    """

    layer: int
    depth: float
    model: Model3D = field(compare=False)


class RenderQueue:
    """Manages the order in which models are submitted for rendering.

    Models are sorted front-to-back for opaque layers (reduces overdraw)
    and back-to-front for transparent layers (correct blending).
    """

    def __init__(self) -> None:
        self._items: List[RenderItem] = []

    def submit(self, model: Model3D, camera: Camera) -> None:
        """Submit *model* to the queue relative to *camera*."""
        layer = (
            RenderLayer.TRANSPARENT.value
            if model.material.opacity < 1.0
            else RenderLayer.OPAQUE.value
        )
        bb = model.world_bounding_box()
        center = bb.center() if bb.is_valid() else model.transform.position
        depth = center.distance_to(camera.position)
        self._items.append(RenderItem(layer=layer, depth=depth, model=model))

    def sorted_items(self) -> List[RenderItem]:
        """Return items in draw order (opaque front-to-back, transparent back-to-front)."""
        opaque = sorted(
            [i for i in self._items if i.layer == RenderLayer.OPAQUE.value],
            key=lambda i: i.depth,
        )
        transparent = sorted(
            [i for i in self._items if i.layer == RenderLayer.TRANSPARENT.value],
            key=lambda i: -i.depth,
        )
        overlay = [i for i in self._items if i.layer == RenderLayer.OVERLAY.value]
        return opaque + transparent + overlay

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class Shader:
    """An abstraction over a vertex/fragment shader pair.

    Attributes:
        name: Shader programme name.
        vertex_source: Vertex shader source code (GLSL or pseudo-code).
        fragment_source: Fragment shader source code.
        uniforms: Key-value uniform bindings.
    """

    name: str = "default"
    vertex_source: str = ""
    fragment_source: str = ""
    uniforms: Dict[str, Any] = field(default_factory=dict)

    def set_uniform(self, name: str, value: Any) -> None:
        self.uniforms[name] = value

    def get_uniform(self, name: str, default: Any = None) -> Any:
        return self.uniforms.get(name, default)

    def __repr__(self) -> str:
        return f"Shader(name={self.name!r}, uniforms={list(self.uniforms.keys())})"

    # ------------------------------------------------------------------
    # Built-in shader templates
    # ------------------------------------------------------------------

    @staticmethod
    def phong() -> "Shader":
        """Return a Phong lighting shader stub."""
        return Shader(
            name="phong",
            vertex_source="// Phong vertex shader",
            fragment_source="// Phong fragment shader",
        )

    @staticmethod
    def unlit() -> "Shader":
        """Return an unlit (flat colour) shader stub."""
        return Shader(
            name="unlit",
            vertex_source="// Unlit vertex shader",
            fragment_source="// Unlit fragment shader",
        )


class CullingSystem:
    """View-frustum culling to exclude off-screen objects.

    This implementation uses a simple sphere-against-frustum test based
    on the bounding box of each model.
    """

    def __init__(self, camera: Camera) -> None:
        self.camera: Camera = camera

    def is_visible(self, model: Model3D) -> bool:
        """Return True if *model* should be rendered given the current camera."""
        if not model.visible:
            return False
        bb = model.world_bounding_box()
        if not bb.is_valid():
            return True  # No geometry – assume visible
        # Simple distance-based culling using the far plane
        center = bb.center()
        radius = bb.size().length() / 2
        dist = center.distance_to(self.camera.position)
        return dist - radius <= self.camera.far

    def cull(self, models: List[Model3D]) -> List[Model3D]:
        """Return only the models that pass the culling test."""
        return [m for m in models if self.is_visible(m)]
