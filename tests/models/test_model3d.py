"""
Unit tests for the 3D model framework (src/models/model3d).
"""

from __future__ import annotations

import json
import math
import struct
import tempfile
from pathlib import Path

import pytest

from src.models.model3d import (
    BoundingBox,
    Camera,
    CullingSystem,
    Light,
    Material,
    Matrix4,
    Mesh,
    Model3D,
    OBJLoader,
    OBJSaver,
    Quaternion,
    Ray,
    RenderQueue,
    Scene,
    SceneGraph,
    Shader,
    STLLoader,
    STLSaver,
    Transform,
    Vector3,
)
from src.models.model3d.light import LightType


# ===========================================================================
# Vector3
# ===========================================================================


class TestVector3:
    def test_init_default(self):
        v = Vector3()
        assert v.x == 0.0 and v.y == 0.0 and v.z == 0.0

    def test_init_values(self):
        v = Vector3(1, 2, 3)
        assert v.x == 1 and v.y == 2 and v.z == 3

    def test_add(self):
        assert Vector3(1, 2, 3) + Vector3(4, 5, 6) == Vector3(5, 7, 9)

    def test_sub(self):
        assert Vector3(5, 7, 9) - Vector3(4, 5, 6) == Vector3(1, 2, 3)

    def test_mul_scalar(self):
        assert Vector3(1, 2, 3) * 2 == Vector3(2, 4, 6)

    def test_rmul_scalar(self):
        assert 2 * Vector3(1, 2, 3) == Vector3(2, 4, 6)

    def test_div_scalar(self):
        result = Vector3(2, 4, 6) / 2
        assert result == Vector3(1, 2, 3)

    def test_div_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            Vector3(1, 2, 3) / 0

    def test_neg(self):
        assert -Vector3(1, -2, 3) == Vector3(-1, 2, -3)

    def test_length(self):
        assert math.isclose(Vector3(3, 4, 0).length(), 5.0)

    def test_length_squared(self):
        assert Vector3(3, 4, 0).length_squared() == 25.0

    def test_normalized(self):
        v = Vector3(0, 5, 0).normalized()
        assert math.isclose(v.y, 1.0)
        assert math.isclose(v.x, 0.0)

    def test_normalized_zero(self):
        v = Vector3(0, 0, 0).normalized()
        assert v == Vector3(0, 0, 0)

    def test_dot(self):
        assert Vector3(1, 0, 0).dot(Vector3(0, 1, 0)) == 0.0
        assert Vector3(1, 0, 0).dot(Vector3(1, 0, 0)) == 1.0

    def test_cross(self):
        c = Vector3(1, 0, 0).cross(Vector3(0, 1, 0))
        assert math.isclose(c.z, 1.0)

    def test_distance_to(self):
        assert math.isclose(Vector3(0, 0, 0).distance_to(Vector3(3, 4, 0)), 5.0)

    def test_lerp(self):
        result = Vector3(0, 0, 0).lerp(Vector3(10, 10, 10), 0.5)
        assert result == Vector3(5, 5, 5)

    def test_reflect(self):
        v = Vector3(1, -1, 0).normalized()
        n = Vector3(0, 1, 0)
        r = v.reflect(n)
        assert r.y > 0  # reflected upward

    def test_iter(self):
        assert list(Vector3(1, 2, 3)) == [1, 2, 3]

    def test_to_tuple(self):
        assert Vector3(1, 2, 3).to_tuple() == (1, 2, 3)

    def test_static_constructors(self):
        assert Vector3.zero() == Vector3(0, 0, 0)
        assert Vector3.one() == Vector3(1, 1, 1)
        assert Vector3.up().y == 1.0
        assert Vector3.forward().z == -1.0
        assert Vector3.right().x == 1.0


# ===========================================================================
# Matrix4
# ===========================================================================


class TestMatrix4:
    def test_identity(self):
        m = Matrix4.identity()
        for i in range(4):
            for j in range(4):
                expected = 1.0 if i == j else 0.0
                assert math.isclose(m[i * 4 + j], expected)

    def test_mul_identity(self):
        m = Matrix4.translation(Vector3(1, 2, 3))
        result = m * Matrix4.identity()
        for i in range(16):
            assert math.isclose(result[i], m[i])

    def test_translation(self):
        m = Matrix4.translation(Vector3(3, 4, 5))
        v = m.transform_point(Vector3(0, 0, 0))
        assert math.isclose(v.x, 3) and math.isclose(v.y, 4) and math.isclose(v.z, 5)

    def test_scale(self):
        m = Matrix4.scale(Vector3(2, 3, 4))
        v = m.transform_point(Vector3(1, 1, 1))
        assert math.isclose(v.x, 2) and math.isclose(v.y, 3) and math.isclose(v.z, 4)

    def test_rotation_x(self):
        m = Matrix4.rotation_x(math.pi / 2)
        v = m.transform_direction(Vector3(0, 1, 0))
        assert math.isclose(v.y, 0, abs_tol=1e-9)
        assert math.isclose(v.z, 1, abs_tol=1e-6)

    def test_rotation_y(self):
        m = Matrix4.rotation_y(math.pi / 2)
        v = m.transform_direction(Vector3(1, 0, 0))
        assert math.isclose(v.x, 0, abs_tol=1e-6)
        assert math.isclose(v.z, -1, abs_tol=1e-6)

    def test_rotation_z(self):
        m = Matrix4.rotation_z(math.pi / 2)
        v = m.transform_direction(Vector3(1, 0, 0))
        assert math.isclose(v.x, 0, abs_tol=1e-6)
        assert math.isclose(v.y, 1, abs_tol=1e-6)

    def test_transposed(self):
        m = Matrix4([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        t = m.transposed()
        assert t[1] == m[4]
        assert t[4] == m[1]

    def test_invalid_data(self):
        with pytest.raises(ValueError):
            Matrix4([1, 2, 3])

    def test_perspective(self):
        m = Matrix4.perspective(math.radians(60), 16 / 9, 0.1, 100.0)
        assert m[15] == 0.0  # perspective projection

    def test_look_at(self):
        eye = Vector3(0, 0, 5)
        target = Vector3(0, 0, 0)
        up = Vector3(0, 1, 0)
        m = Matrix4.look_at(eye, target, up)
        # The origin in view space should be (0, 0, 5)
        assert m[15] == 1.0


# ===========================================================================
# Quaternion
# ===========================================================================


class TestQuaternion:
    def test_identity(self):
        q = Quaternion.identity()
        v = q.rotate_vector(Vector3(1, 0, 0))
        assert math.isclose(v.x, 1, abs_tol=1e-9)

    def test_from_axis_angle(self):
        q = Quaternion.from_axis_angle(Vector3(0, 1, 0), math.pi / 2)
        v = q.rotate_vector(Vector3(1, 0, 0))
        assert math.isclose(v.x, 0, abs_tol=1e-6)
        assert math.isclose(v.z, -1, abs_tol=1e-6)

    def test_normalized(self):
        q = Quaternion(2, 0, 0, 0).normalized()
        assert math.isclose(q.w, 1.0)

    def test_conjugate(self):
        q = Quaternion(1, 2, 3, 4)
        c = q.conjugate()
        assert c.x == -2 and c.y == -3 and c.z == -4 and c.w == 1

    def test_to_matrix4(self):
        q = Quaternion.identity()
        m = q.to_matrix4()
        for i in range(4):
            for j in range(4):
                expected = 1.0 if i == j else 0.0
                assert math.isclose(m[i * 4 + j], expected, abs_tol=1e-9)

    def test_from_euler(self):
        q = Quaternion.from_euler(0, 0, 0)
        assert math.isclose(q.w, 1.0, abs_tol=1e-9)

    def test_slerp(self):
        q1 = Quaternion.identity()
        q2 = Quaternion.from_axis_angle(Vector3(0, 1, 0), math.pi / 2)
        q_mid = q1.slerp(q2, 0.5)
        v = q_mid.rotate_vector(Vector3(1, 0, 0))
        # 45° rotation: x and z should be equal magnitude
        assert math.isclose(abs(v.x), abs(v.z), abs_tol=1e-5)

    def test_mul(self):
        q1 = Quaternion.from_axis_angle(Vector3(0, 1, 0), math.pi / 4)
        q2 = Quaternion.from_axis_angle(Vector3(0, 1, 0), math.pi / 4)
        q = (q1 * q2).normalized()
        v = q.rotate_vector(Vector3(1, 0, 0))
        # Combined 90° rotation around Y
        assert math.isclose(v.x, 0, abs_tol=1e-6)
        assert math.isclose(v.z, -1, abs_tol=1e-6)


# ===========================================================================
# BoundingBox
# ===========================================================================


class TestBoundingBox:
    def test_empty_invalid(self):
        bb = BoundingBox()
        assert not bb.is_valid()

    def test_expand(self):
        bb = BoundingBox()
        bb.expand(Vector3(1, 2, 3))
        bb.expand(Vector3(-1, -2, -3))
        assert bb.is_valid()
        assert bb.min_point == Vector3(-1, -2, -3)
        assert bb.max_point == Vector3(1, 2, 3)

    def test_contains(self):
        bb = BoundingBox.from_points([Vector3(-1, -1, -1), Vector3(1, 1, 1)])
        assert bb.contains(Vector3(0, 0, 0))
        assert not bb.contains(Vector3(2, 0, 0))

    def test_center(self):
        bb = BoundingBox.from_points([Vector3(-2, -2, -2), Vector3(2, 2, 2)])
        assert bb.center() == Vector3(0, 0, 0)

    def test_intersects(self):
        a = BoundingBox.from_points([Vector3(0, 0, 0), Vector3(2, 2, 2)])
        b = BoundingBox.from_points([Vector3(1, 1, 1), Vector3(3, 3, 3)])
        c = BoundingBox.from_points([Vector3(5, 5, 5), Vector3(6, 6, 6)])
        assert a.intersects(b)
        assert not a.intersects(c)

    def test_intersects_ray(self):
        bb = BoundingBox.from_points([Vector3(-1, -1, -1), Vector3(1, 1, 1)])
        ray = Ray(origin=Vector3(0, 0, -5), direction=Vector3(0, 0, 1))
        hit, t = bb.intersects_ray(ray)
        assert hit
        assert t > 0

    def test_no_ray_intersection(self):
        bb = BoundingBox.from_points([Vector3(5, 5, 5), Vector3(6, 6, 6)])
        ray = Ray(origin=Vector3(0, 0, 0), direction=Vector3(0, 1, 0))
        hit, _ = bb.intersects_ray(ray)
        assert not hit


# ===========================================================================
# Ray
# ===========================================================================


class TestRay:
    def test_point_at(self):
        ray = Ray(origin=Vector3(0, 0, 0), direction=Vector3(1, 0, 0))
        assert ray.point_at(5.0) == Vector3(5, 0, 0)


# ===========================================================================
# Transform
# ===========================================================================


class TestTransform:
    def test_default(self):
        t = Transform()
        assert t.position == Vector3(0, 0, 0)
        assert t.scale == Vector3(1, 1, 1)

    def test_translate(self):
        t = Transform()
        t.translate(Vector3(1, 2, 3))
        assert t.position == Vector3(1, 2, 3)

    def test_to_matrix_identity(self):
        t = Transform()
        m = t.to_matrix()
        v = m.transform_point(Vector3(1, 2, 3))
        assert math.isclose(v.x, 1) and math.isclose(v.y, 2) and math.isclose(v.z, 3)

    def test_to_matrix_with_translation(self):
        t = Transform(position=Vector3(5, 0, 0))
        m = t.to_matrix()
        v = m.transform_point(Vector3(0, 0, 0))
        assert math.isclose(v.x, 5)

    def test_directions(self):
        t = Transform()
        assert math.isclose(t.up().y, 1.0, abs_tol=1e-9)
        assert math.isclose(t.right().x, 1.0, abs_tol=1e-9)


# ===========================================================================
# Mesh
# ===========================================================================


class TestMesh:
    def test_create_cube(self):
        mesh = Mesh.create_cube()
        assert mesh.vertex_count == 8
        assert mesh.face_count == 12
        assert len(mesh.normals) == 8

    def test_create_plane(self):
        mesh = Mesh.create_plane()
        assert mesh.vertex_count == 4
        assert mesh.face_count == 2

    def test_create_sphere(self):
        mesh = Mesh.create_sphere(segments=8, rings=4)
        assert mesh.vertex_count > 0
        assert mesh.face_count > 0

    def test_bounding_box(self):
        mesh = Mesh.create_cube(size=2.0)
        bb = mesh.bounding_box()
        assert bb.is_valid()
        assert math.isclose(bb.size().x, 2.0, abs_tol=1e-9)

    def test_face_normals(self):
        mesh = Mesh.create_cube()
        fns = mesh.compute_face_normals()
        assert len(fns) == mesh.face_count
        for fn in fns:
            assert math.isclose(fn.length(), 1.0, abs_tol=1e-6)


# ===========================================================================
# Material
# ===========================================================================


class TestMaterial:
    def test_default(self):
        m = Material.default()
        assert m.name == "default"

    def test_from_color(self):
        m = Material.from_color(1.0, 0.0, 0.0)
        assert m.diffuse[0] == 1.0

    def test_matte(self):
        m = Material.matte(0.5, 0.5, 0.5)
        assert m.shininess == 1.0

    def test_metallic(self):
        m = Material.metallic(0.5, 0.5, 0.5)
        assert m.shininess == 128.0

    def test_opacity_default(self):
        assert Material().opacity == 1.0


# ===========================================================================
# Camera
# ===========================================================================


class TestCamera:
    def test_default(self):
        c = Camera()
        assert c.fov_y == 60.0

    def test_view_matrix(self):
        c = Camera(position=Vector3(0, 0, 5), target=Vector3(0, 0, 0))
        m = c.view_matrix()
        assert m[15] == 1.0

    def test_projection_matrix(self):
        c = Camera()
        m = c.projection_matrix()
        assert m[15] == 0.0  # perspective projection

    def test_orthographic_projection(self):
        c = Camera(orthographic=True)
        m = c.projection_matrix()
        assert m[15] == 1.0

    def test_ray_from_screen(self):
        c = Camera(position=Vector3(0, 0, 5), target=Vector3(0, 0, 0))
        ray = c.ray_from_screen(0.0, 0.0)
        assert ray.origin == c.position
        assert math.isclose(ray.direction.length(), 1.0, abs_tol=1e-6)

    def test_move(self):
        c = Camera(position=Vector3(0, 0, 5), target=Vector3(0, 0, 0))
        c.move(Vector3(1, 0, 0))
        assert math.isclose(c.position.x, 1.0)
        assert math.isclose(c.target.x, 1.0)

    def test_zoom(self):
        c = Camera(position=Vector3(0, 0, 5), target=Vector3(0, 0, 0))
        original_dist = c.position.distance_to(c.target)
        c.zoom(1.0)
        new_dist = c.position.distance_to(c.target)
        assert new_dist < original_dist


# ===========================================================================
# Model3D
# ===========================================================================


class TestModel3D:
    def test_init(self):
        model = Model3D(name="test")
        assert model.name == "test"
        assert model.visible is True

    def test_world_bounding_box(self):
        mesh = Mesh.create_cube()
        model = Model3D(mesh=mesh)
        bb = model.world_bounding_box()
        assert bb.is_valid()

    def test_serialisation_round_trip(self):
        mesh = Mesh.create_cube()
        material = Material.from_color(0.5, 0.5, 1.0, name="blue")
        model = Model3D(name="cube_model", mesh=mesh, material=material)
        model.tags = ["static", "opaque"]

        data = model.to_dict()
        restored = Model3D.from_dict(data)

        assert restored.name == "cube_model"
        assert restored.mesh.vertex_count == mesh.vertex_count
        assert restored.material.name == "blue"
        assert "static" in restored.tags


# ===========================================================================
# Light
# ===========================================================================


class TestLight:
    def test_ambient(self):
        l = Light.ambient()
        assert l.light_type == LightType.AMBIENT

    def test_directional(self):
        l = Light.directional(Vector3(0, -1, 0))
        assert l.light_type == LightType.DIRECTIONAL
        assert math.isclose(l.direction.length(), 1.0, abs_tol=1e-9)

    def test_point(self):
        l = Light.point(Vector3(0, 5, 0), range=20.0)
        assert l.light_type == LightType.POINT
        assert l.range == 20.0


# ===========================================================================
# Scene
# ===========================================================================


class TestScene:
    def test_add_remove(self):
        scene = Scene("test_scene")
        model = Model3D(name="cube", mesh=Mesh.create_cube())
        scene.add_model(model)
        assert scene.get_model("cube") is model
        removed = scene.remove_model("cube")
        assert removed is model
        assert scene.get_model("cube") is None

    def test_visible_models(self):
        scene = Scene()
        scene.add_model(Model3D(name="visible"))
        hidden = Model3D(name="hidden")
        hidden.visible = False
        scene.add_model(hidden)
        assert len(scene.visible_models()) == 1

    def test_serialisation(self):
        scene = Scene("s1")
        scene.add_model(Model3D(name="m1", mesh=Mesh.create_cube()))
        data = scene.to_dict()
        restored = Scene.from_dict(data)
        assert restored.name == "s1"
        assert "m1" in restored.models


# ===========================================================================
# SceneGraph
# ===========================================================================


class TestSceneGraph:
    def test_add_and_get(self):
        sg = SceneGraph()
        model = Model3D(name="root")
        node = sg.add(model)
        assert sg.get("root") is node

    def test_parent_child(self):
        sg = SceneGraph()
        parent = Model3D(name="parent")
        child = Model3D(name="child")
        sg.add(parent)
        sg.add(child, parent_name="parent")
        assert sg.get("child").parent is sg.get("parent")

    def test_world_matrix_propagation(self):
        sg = SceneGraph()
        from src.models.model3d.math3d import Vector3 as V3
        parent_model = Model3D(name="p")
        parent_model.transform.position = V3(5, 0, 0)
        child_model = Model3D(name="c")
        sg.add(parent_model)
        sg.add(child_model, parent_name="p")
        wm = sg.get("c").world_matrix()
        origin = wm.transform_point(V3(0, 0, 0))
        assert math.isclose(origin.x, 5.0)

    def test_remove(self):
        sg = SceneGraph()
        sg.add(Model3D(name="n1"))
        assert sg.remove("n1")
        assert sg.get("n1") is None

    def test_unknown_parent_raises(self):
        sg = SceneGraph()
        sg.add(Model3D(name="a"))
        with pytest.raises(KeyError):
            sg.add(Model3D(name="b"), parent_name="nonexistent")


# ===========================================================================
# File I/O – OBJ
# ===========================================================================


class TestOBJIO:
    def test_save_and_load(self):
        mesh = Mesh.create_cube()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cube.obj"
            OBJSaver().save(mesh, path)
            assert path.exists()
            loaded = OBJLoader().load(path)
            assert loaded.vertex_count == mesh.vertex_count
            assert loaded.face_count == mesh.face_count

    def test_load_with_normals(self):
        obj_text = (
            "# test\n"
            "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
            "vn 0 0 1\n"
            "f 1 2 3\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.obj"
            path.write_text(obj_text)
            mesh = OBJLoader().load(path)
            assert mesh.vertex_count == 3
            assert mesh.face_count == 1

    def test_load_quad(self):
        obj_text = (
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
            "f 1 2 3 4\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "quad.obj"
            path.write_text(obj_text)
            mesh = OBJLoader().load(path)
            # quad triangulated to 2 faces
            assert mesh.face_count == 2


# ===========================================================================
# File I/O – STL
# ===========================================================================


class TestSTLIO:
    def test_save_and_load(self):
        mesh = Mesh.create_cube()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cube.stl"
            STLSaver().save(mesh, path)
            assert path.exists()
            loaded = STLLoader().load(path)
            # STL stores per-triangle vertices (unshared)
            assert loaded.face_count == mesh.face_count

    def test_stl_binary_format(self):
        mesh = Mesh.create_cube()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cube.stl"
            STLSaver().save(mesh, path)
            with path.open("rb") as fh:
                fh.read(80)  # header
                num_tris = struct.unpack("<I", fh.read(4))[0]
            assert num_tris == mesh.face_count


# ===========================================================================
# Renderer
# ===========================================================================


class TestRenderQueue:
    def test_submit_and_sort(self):
        camera = Camera(position=Vector3(0, 0, 10), target=Vector3(0, 0, 0))
        queue = RenderQueue()

        far_model = Model3D(name="far", mesh=Mesh.create_cube())
        far_model.transform.position = Vector3(0, 0, -5)

        near_model = Model3D(name="near", mesh=Mesh.create_cube())
        near_model.transform.position = Vector3(0, 0, 5)

        queue.submit(far_model, camera)
        queue.submit(near_model, camera)

        items = queue.sorted_items()
        assert items[0].model.name == "near"  # front-to-back for opaque

    def test_transparent_back_to_front(self):
        camera = Camera(position=Vector3(0, 0, 10), target=Vector3(0, 0, 0))
        queue = RenderQueue()

        far_model = Model3D(name="far_t", mesh=Mesh.create_cube(), material=Material(opacity=0.5))
        far_model.transform.position = Vector3(0, 0, -5)

        near_model = Model3D(name="near_t", mesh=Mesh.create_cube(), material=Material(opacity=0.5))
        near_model.transform.position = Vector3(0, 0, 5)

        queue.submit(far_model, camera)
        queue.submit(near_model, camera)

        items = queue.sorted_items()
        assert items[0].model.name == "far_t"  # back-to-front for transparent

    def test_clear(self):
        queue = RenderQueue()
        queue.submit(Model3D(name="a"), Camera())
        queue.clear()
        assert len(queue) == 0


class TestShader:
    def test_default_shader(self):
        s = Shader()
        assert s.name == "default"

    def test_set_get_uniform(self):
        s = Shader()
        s.set_uniform("light_pos", (1.0, 2.0, 3.0))
        assert s.get_uniform("light_pos") == (1.0, 2.0, 3.0)
        assert s.get_uniform("missing", "default_val") == "default_val"

    def test_phong_factory(self):
        s = Shader.phong()
        assert s.name == "phong"

    def test_unlit_factory(self):
        s = Shader.unlit()
        assert s.name == "unlit"


class TestCullingSystem:
    def test_visible(self):
        camera = Camera(position=Vector3(0, 0, 10), far=100.0)
        cs = CullingSystem(camera)
        model = Model3D(name="visible", mesh=Mesh.create_cube())
        assert cs.is_visible(model)

    def test_hidden_flag(self):
        camera = Camera(position=Vector3(0, 0, 10), far=100.0)
        cs = CullingSystem(camera)
        model = Model3D(name="hidden", mesh=Mesh.create_cube())
        model.visible = False
        assert not cs.is_visible(model)

    def test_cull_list(self):
        camera = Camera(position=Vector3(0, 0, 0), far=5.0)
        cs = CullingSystem(camera)

        near = Model3D(name="near", mesh=Mesh.create_cube())
        near.transform.position = Vector3(0, 0, 0)

        far = Model3D(name="far", mesh=Mesh.create_cube())
        far.transform.position = Vector3(0, 0, 200)  # beyond far plane

        visible = cs.cull([near, far])
        names = [m.name for m in visible]
        assert "near" in names
        assert "far" not in names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
