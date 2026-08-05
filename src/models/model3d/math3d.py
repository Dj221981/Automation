"""
3D math primitives: Vector3, Matrix4, Quaternion, BoundingBox, Ray.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, Tuple


@dataclass
class Vector3:
    """3D vector with common arithmetic and geometric operations."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vector3":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vector3":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> "Vector3":
        return Vector3(-self.x, -self.y, -self.z)

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y
        yield self.z

    def __repr__(self) -> str:
        return f"Vector3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"

    # ------------------------------------------------------------------
    # Geometric
    # ------------------------------------------------------------------

    def length(self) -> float:
        """Return the Euclidean length of the vector."""
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def length_squared(self) -> float:
        """Return the squared length (cheaper than length())."""
        return self.x ** 2 + self.y ** 2 + self.z ** 2

    def normalized(self) -> "Vector3":
        """Return a unit vector in the same direction."""
        mag = self.length()
        if mag == 0:
            return Vector3(0.0, 0.0, 0.0)
        return self / mag

    def dot(self, other: "Vector3") -> float:
        """Return the dot product with *other*."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3") -> "Vector3":
        """Return the cross product with *other*."""
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def distance_to(self, other: "Vector3") -> float:
        """Return the Euclidean distance to *other*."""
        return (self - other).length()

    def lerp(self, other: "Vector3", t: float) -> "Vector3":
        """Linear interpolation between self and *other* at factor *t*."""
        return self + (other - self) * t

    def reflect(self, normal: "Vector3") -> "Vector3":
        """Reflect this vector about *normal* (normal should be unit length)."""
        return self - normal * (2 * self.dot(normal))

    def to_tuple(self) -> Tuple[float, float, float]:
        """Return as a plain tuple."""
        return (self.x, self.y, self.z)

    @staticmethod
    def zero() -> "Vector3":
        return Vector3(0.0, 0.0, 0.0)

    @staticmethod
    def one() -> "Vector3":
        return Vector3(1.0, 1.0, 1.0)

    @staticmethod
    def up() -> "Vector3":
        return Vector3(0.0, 1.0, 0.0)

    @staticmethod
    def forward() -> "Vector3":
        return Vector3(0.0, 0.0, -1.0)

    @staticmethod
    def right() -> "Vector3":
        return Vector3(1.0, 0.0, 0.0)


class Matrix4:
    """Column-major 4x4 matrix for 3D transformations."""

    def __init__(self, data: list[float] | None = None) -> None:
        """Initialise with *data* (16 floats, row-major) or identity."""
        if data is not None:
            if len(data) != 16:
                raise ValueError("Matrix4 requires exactly 16 values")
            self._m: list[float] = list(data)
        else:
            self._m = [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1,
            ]

    def __getitem__(self, index: int) -> float:
        return self._m[index]

    def __setitem__(self, index: int, value: float) -> None:
        self._m[index] = value

    def __repr__(self) -> str:
        rows = [self._m[i * 4: i * 4 + 4] for i in range(4)]
        return "Matrix4(\n" + "\n".join(f"  {r}" for r in rows) + "\n)"

    def __mul__(self, other: "Matrix4") -> "Matrix4":
        a, b = self._m, other._m
        result = [0.0] * 16
        for row in range(4):
            for col in range(4):
                result[row * 4 + col] = sum(
                    a[row * 4 + k] * b[k * 4 + col] for k in range(4)
                )
        return Matrix4(result)

    def transform_point(self, v: Vector3) -> Vector3:
        """Apply the matrix to a point (w=1)."""
        m = self._m
        x = m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3]
        y = m[4] * v.x + m[5] * v.y + m[6] * v.z + m[7]
        z = m[8] * v.x + m[9] * v.y + m[10] * v.z + m[11]
        w = m[12] * v.x + m[13] * v.y + m[14] * v.z + m[15]
        if w != 0 and w != 1:
            x, y, z = x / w, y / w, z / w
        return Vector3(x, y, z)

    def transform_direction(self, v: Vector3) -> Vector3:
        """Apply the matrix to a direction (w=0)."""
        m = self._m
        return Vector3(
            m[0] * v.x + m[1] * v.y + m[2] * v.z,
            m[4] * v.x + m[5] * v.y + m[6] * v.z,
            m[8] * v.x + m[9] * v.y + m[10] * v.z,
        )

    def transposed(self) -> "Matrix4":
        m = self._m
        return Matrix4([
            m[0], m[4], m[8],  m[12],
            m[1], m[5], m[9],  m[13],
            m[2], m[6], m[10], m[14],
            m[3], m[7], m[11], m[15],
        ])

    def to_list(self) -> list[float]:
        return list(self._m)

    @staticmethod
    def identity() -> "Matrix4":
        return Matrix4()

    @staticmethod
    def translation(v: Vector3) -> "Matrix4":
        return Matrix4([
            1, 0, 0, v.x,
            0, 1, 0, v.y,
            0, 0, 1, v.z,
            0, 0, 0, 1,
        ])

    @staticmethod
    def scale(v: Vector3) -> "Matrix4":
        return Matrix4([
            v.x, 0,   0,   0,
            0,   v.y, 0,   0,
            0,   0,   v.z, 0,
            0,   0,   0,   1,
        ])

    @staticmethod
    def rotation_x(angle_rad: float) -> "Matrix4":
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        return Matrix4([
            1, 0,  0, 0,
            0, c, -s, 0,
            0, s,  c, 0,
            0, 0,  0, 1,
        ])

    @staticmethod
    def rotation_y(angle_rad: float) -> "Matrix4":
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        return Matrix4([
             c, 0, s, 0,
             0, 1, 0, 0,
            -s, 0, c, 0,
             0, 0, 0, 1,
        ])

    @staticmethod
    def rotation_z(angle_rad: float) -> "Matrix4":
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        return Matrix4([
            c, -s, 0, 0,
            s,  c, 0, 0,
            0,  0, 1, 0,
            0,  0, 0, 1,
        ])

    @staticmethod
    def perspective(fov_y_rad: float, aspect: float, near: float, far: float) -> "Matrix4":
        """Build a perspective projection matrix."""
        tan_half = math.tan(fov_y_rad / 2)
        return Matrix4([
            1 / (aspect * tan_half), 0, 0, 0,
            0, 1 / tan_half, 0, 0,
            0, 0, -(far + near) / (far - near), -(2 * far * near) / (far - near),
            0, 0, -1, 0,
        ])

    @staticmethod
    def look_at(eye: Vector3, target: Vector3, up: Vector3) -> "Matrix4":
        """Build a view matrix looking from *eye* toward *target*."""
        f = (target - eye).normalized()
        r = f.cross(up).normalized()
        u = r.cross(f)
        return Matrix4([
            r.x,  r.y,  r.z, -r.dot(eye),
            u.x,  u.y,  u.z, -u.dot(eye),
            -f.x, -f.y, -f.z, f.dot(eye),
            0,    0,    0,   1,
        ])


@dataclass
class Quaternion:
    """Unit quaternion for representing 3D rotations."""

    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(
            w=self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x=self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y=self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z=self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def conjugate(self) -> "Quaternion":
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def length(self) -> float:
        return math.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalized(self) -> "Quaternion":
        mag = self.length()
        if mag == 0:
            return Quaternion()
        return Quaternion(self.w / mag, self.x / mag, self.y / mag, self.z / mag)

    def rotate_vector(self, v: Vector3) -> Vector3:
        """Rotate *v* by this quaternion."""
        qv = Quaternion(0, v.x, v.y, v.z)
        rotated = self * qv * self.conjugate()
        return Vector3(rotated.x, rotated.y, rotated.z)

    def to_matrix4(self) -> Matrix4:
        """Convert to a 4x4 rotation matrix."""
        w, x, y, z = self.w, self.x, self.y, self.z
        return Matrix4([
            1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w),   0,
            2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w),   0,
            2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y), 0,
            0,                   0,                  0,               1,
        ])

    def __repr__(self) -> str:
        return f"Quaternion(w={self.w:.4f}, x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"

    @staticmethod
    def identity() -> "Quaternion":
        return Quaternion(1.0, 0.0, 0.0, 0.0)

    @staticmethod
    def from_axis_angle(axis: Vector3, angle_rad: float) -> "Quaternion":
        """Construct from an axis-angle representation."""
        half = angle_rad / 2
        s = math.sin(half)
        n = axis.normalized()
        return Quaternion(math.cos(half), n.x * s, n.y * s, n.z * s)

    @staticmethod
    def from_euler(pitch: float, yaw: float, roll: float) -> "Quaternion":
        """Construct from Euler angles (radians, XYZ order)."""
        qx = Quaternion.from_axis_angle(Vector3(1, 0, 0), pitch)
        qy = Quaternion.from_axis_angle(Vector3(0, 1, 0), yaw)
        qz = Quaternion.from_axis_angle(Vector3(0, 0, 1), roll)
        return qz * qy * qx

    def slerp(self, other: "Quaternion", t: float) -> "Quaternion":
        """Spherical linear interpolation between self and *other*."""
        dot = self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
        # Clamp to valid range
        dot = max(-1.0, min(1.0, dot))
        # If quaternions are close, use lerp for numerical stability
        if dot > 0.9995:
            result = Quaternion(
                self.w + t * (other.w - self.w),
                self.x + t * (other.x - self.x),
                self.y + t * (other.y - self.y),
                self.z + t * (other.z - self.z),
            )
            return result.normalized()
        theta_0 = math.acos(dot)
        theta = theta_0 * t
        sin_theta = math.sin(theta)
        sin_theta_0 = math.sin(theta_0)
        s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return Quaternion(
            s0 * self.w + s1 * other.w,
            s0 * self.x + s1 * other.x,
            s0 * self.y + s1 * other.y,
            s0 * self.z + s1 * other.z,
        )


@dataclass
class BoundingBox:
    """Axis-aligned bounding box (AABB)."""

    min_point: Vector3 = field(default_factory=lambda: Vector3(float("inf"), float("inf"), float("inf")))
    max_point: Vector3 = field(default_factory=lambda: Vector3(float("-inf"), float("-inf"), float("-inf")))

    def is_valid(self) -> bool:
        """Return True if the bounding box is non-empty."""
        return (
            self.min_point.x <= self.max_point.x
            and self.min_point.y <= self.max_point.y
            and self.min_point.z <= self.max_point.z
        )

    def center(self) -> Vector3:
        return (self.min_point + self.max_point) * 0.5

    def size(self) -> Vector3:
        return self.max_point - self.min_point

    def expand(self, point: Vector3) -> None:
        """Expand the box to contain *point*."""
        self.min_point = Vector3(
            min(self.min_point.x, point.x),
            min(self.min_point.y, point.y),
            min(self.min_point.z, point.z),
        )
        self.max_point = Vector3(
            max(self.max_point.x, point.x),
            max(self.max_point.y, point.y),
            max(self.max_point.z, point.z),
        )

    def contains(self, point: Vector3) -> bool:
        return (
            self.min_point.x <= point.x <= self.max_point.x
            and self.min_point.y <= point.y <= self.max_point.y
            and self.min_point.z <= point.z <= self.max_point.z
        )

    def intersects(self, other: "BoundingBox") -> bool:
        return (
            self.min_point.x <= other.max_point.x
            and self.max_point.x >= other.min_point.x
            and self.min_point.y <= other.max_point.y
            and self.max_point.y >= other.min_point.y
            and self.min_point.z <= other.max_point.z
            and self.max_point.z >= other.min_point.z
        )

    def intersects_ray(self, ray: "Ray") -> tuple[bool, float]:
        """Slab method ray-AABB intersection. Returns (hit, t)."""
        t_min = float("-inf")
        t_max = float("inf")
        for axis in ("x", "y", "z"):
            origin = getattr(ray.origin, axis)
            direction = getattr(ray.direction, axis)
            box_min = getattr(self.min_point, axis)
            box_max = getattr(self.max_point, axis)
            if abs(direction) < 1e-9:
                if origin < box_min or origin > box_max:
                    return False, 0.0
            else:
                t1 = (box_min - origin) / direction
                t2 = (box_max - origin) / direction
                if t1 > t2:
                    t1, t2 = t2, t1
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                if t_min > t_max:
                    return False, 0.0
        return True, t_min

    @staticmethod
    def from_points(points: list[Vector3]) -> "BoundingBox":
        box = BoundingBox()
        for p in points:
            box.expand(p)
        return box


@dataclass
class Ray:
    """A ray defined by an origin and a (normalised) direction."""

    origin: Vector3 = field(default_factory=Vector3.zero)
    direction: Vector3 = field(default_factory=Vector3.forward)

    def point_at(self, t: float) -> Vector3:
        """Return the point along the ray at parameter *t*."""
        return self.origin + self.direction * t

    def __repr__(self) -> str:
        return f"Ray(origin={self.origin}, direction={self.direction})"
