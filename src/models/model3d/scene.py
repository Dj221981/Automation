"""
Scene: container for 3D objects, lights, and the active camera.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .model3d import Model3D
from .camera import Camera
from .light import Light


class Scene:
    """A 3D scene containing models, lights, and a camera.

    Attributes:
        name: Scene name.
        models: Registered 3D models, keyed by name.
        lights: List of lights.
        camera: Active camera.
        background_color: Clear colour as (R, G, B) in [0, 1].
    """

    def __init__(self, name: str = "scene") -> None:
        self.name: str = name
        self.models: Dict[str, Model3D] = {}
        self.lights: List[Light] = []
        self.camera: Camera = Camera()
        self.background_color: tuple = (0.1, 0.1, 0.1)

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def add_model(self, model: Model3D) -> None:
        """Add *model* to the scene (overwrites if name clashes)."""
        self.models[model.name] = model

    def remove_model(self, name: str) -> Optional[Model3D]:
        """Remove and return the model with *name*, or None."""
        return self.models.pop(name, None)

    def get_model(self, name: str) -> Optional[Model3D]:
        return self.models.get(name)

    def visible_models(self) -> List[Model3D]:
        return [m for m in self.models.values() if m.visible]

    # ------------------------------------------------------------------
    # Light management
    # ------------------------------------------------------------------

    def add_light(self, light: Light) -> None:
        self.lights.append(light)

    def remove_light(self, light: Light) -> None:
        self.lights.remove(light)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "background_color": list(self.background_color),
            "models": {n: m.to_dict() for n, m in self.models.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Scene":
        scene = cls(name=data.get("name", "scene"))
        scene.background_color = tuple(data.get("background_color", [0.1, 0.1, 0.1]))
        for model_data in data.get("models", {}).values():
            scene.add_model(Model3D.from_dict(model_data))
        return scene

    def __repr__(self) -> str:
        return (
            f"Scene(name={self.name!r}, models={len(self.models)}, lights={len(self.lights)})"
        )
