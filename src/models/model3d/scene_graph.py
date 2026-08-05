"""
SceneGraph: hierarchical parent–child organisation of 3D objects.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .model3d import Model3D
from .math3d import Matrix4


class SceneNode:
    """A node in the scene graph, wrapping a Model3D.

    Each node may have a parent and multiple children.  The world
    transform of a node is the product of all ancestor local transforms.
    """

    def __init__(self, model: Model3D) -> None:
        self.model: Model3D = model
        self.parent: Optional["SceneNode"] = None
        self.children: List["SceneNode"] = []

    def add_child(self, child: "SceneNode") -> None:
        if child.parent is not None:
            child.parent.remove_child(child)
        child.parent = self
        self.children.append(child)

    def remove_child(self, child: "SceneNode") -> bool:
        if child in self.children:
            self.children.remove(child)
            child.parent = None
            return True
        return False

    def world_matrix(self) -> Matrix4:
        """Return the accumulated world-space matrix."""
        local = self.model.transform.to_matrix()
        if self.parent is None:
            return local
        return self.parent.world_matrix() * local

    def descendants(self) -> List["SceneNode"]:
        """Return all descendant nodes (depth-first)."""
        result: List["SceneNode"] = []
        for child in self.children:
            result.append(child)
            result.extend(child.descendants())
        return result

    def __repr__(self) -> str:
        return f"SceneNode(model={self.model.name!r}, children={len(self.children)})"


class SceneGraph:
    """Hierarchical container of SceneNodes.

    Attributes:
        roots: Top-level nodes without a parent.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, SceneNode] = {}
        self.roots: List[SceneNode] = []

    def add(self, model: Model3D, parent_name: Optional[str] = None) -> SceneNode:
        """Add *model* as a new node, optionally under *parent_name*."""
        node = SceneNode(model)
        self._nodes[model.name] = node
        if parent_name is not None:
            parent_node = self._nodes.get(parent_name)
            if parent_node is None:
                raise KeyError(f"Parent node {parent_name!r} not found in scene graph")
            parent_node.add_child(node)
        else:
            self.roots.append(node)
        return node

    def remove(self, name: str) -> bool:
        """Remove the node identified by *name* and detach its children."""
        node = self._nodes.pop(name, None)
        if node is None:
            return False
        if node.parent is not None:
            node.parent.remove_child(node)
        else:
            self.roots.remove(node)
        # Reparent children to roots
        for child in list(node.children):
            node.remove_child(child)
            self.roots.append(child)
        return True

    def get(self, name: str) -> Optional[SceneNode]:
        return self._nodes.get(name)

    def all_nodes(self) -> List[SceneNode]:
        """Return every node in the graph."""
        return list(self._nodes.values())

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        return f"SceneGraph(nodes={len(self._nodes)}, roots={len(self.roots)})"
