"""
游戏内光线追踪与路径追踪系统 - 核心模块（修复版）

修复内容：
1. trace_ray()             —— BVH 加速结构 + 递归多次反射/折射 + 软阴影（面光源采样）
2. path_tracing()          —— 蒙特卡洛积分 + 余弦加权半球重要性采样 + 光源重要性采样
3. global_illumination()   —— 辐射度（Radiosity）方法计算间接光照与颜色溢出
4. reflection_refraction() —— 菲涅尔效应（Fresnel）+ 斯涅尔定律（Snell's law）
5. caustics()              —— 光子映射（Photon Mapping）焦散 + 体积焦散

纯 Python 标准库实现，不依赖第三方库。
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import random
import time

Vec3 = Tuple[float, float, float]
_INF = float("inf")
_EPS = 1e-6


# ---------------------------------------------------------------------------
# 向量 / 颜色工具
# ---------------------------------------------------------------------------

def _vec3(value, default=(0.0, 0.0, 0.0)):
    if value is None:
        return (float(default[0]), float(default[1]), float(default[2]))
    if isinstance(value, (int, float)):
        s = float(value)
        return (s, s, s)
    if isinstance(value, dict):
        return (float(value.get("x", 0.0)),
                float(value.get("y", 0.0)),
                float(value.get("z", 0.0)))
    return (float(value[0]), float(value[1]), float(value[2]))


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _mul(a, b):
    """分量相乘（颜色 × 颜色 或 颜色 × 标量）。"""
    if isinstance(b, (int, float)):
        return (a[0] * b, a[1] * b, a[2] * b)
    return (a[0] * b[0], a[1] * b[1], a[2] * b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _length(a):
    return math.sqrt(_dot(a, a))


def _normalize(a):
    length = _length(a)
    if length < _EPS:
        return (0.0, 0.0, 0.0)
    return (a[0] / length, a[1] / length, a[2] / length)


def _clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def _clamp_color(c):
    return (_clamp01(c[0]), _clamp01(c[1]), _clamp01(c[2]))


def reflect(direction, normal):
    """反射方向：R = D - 2(D·N)N。"""
    d = _normalize(_vec3(direction))
    n = _normalize(_vec3(normal))
    return _sub(d, _scale(n, 2.0 * _dot(d, n)))


def refract(direction, normal, n1=1.0, n2=1.5):
    """
    斯涅尔定律折射。normal 与 direction 相对（d·n < 0）。
    发生全内反射时返回 None。
    """
    d = _normalize(_vec3(direction))
    n = _normalize(_vec3(normal))
    if _dot(d, n) > 0.0:
        n = _scale(n, -1.0)
    return _refract(d, n, float(n1), float(n2))


def _reflect(d, n):
    return _sub(d, _scale(n, 2.0 * _dot(d, n)))


def _refract(d, n, n1, n2):
    eta = n1 / n2
    cos_i = min(1.0, -_dot(d, n))
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)
    if sin2_t >= 1.0:
        return None  # 全内反射（TIR）
    cos_t = math.sqrt(max(0.0, 1.0 - sin2_t))
    return _normalize(_add(_scale(d, eta), _scale(n, eta * cos_i - cos_t)))


def fresnel_schlick(cos_theta, n1=1.0, n2=1.5):
    """Schlick 近似菲涅尔反射率。"""
    n1, n2 = float(n1), float(n2)
    r0 = (n1 - n2) / (n1 + n2)
    r0 = r0 * r0
    c = max(0.0, min(1.0, float(cos_theta)))
    return r0 + (1.0 - r0) * (1.0 - c) ** 5


def fresnel(cos_theta, n1=1.0, n2=1.5):
    """非偏振光精确菲涅尔反射率（全内反射时为 1）。"""
    n1, n2 = float(n1), float(n2)
    cos_i = max(0.0, min(1.0, float(cos_theta)))
    eta = n1 / n2
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)
    if sin2_t >= 1.0:
        return 1.0
    cos_t = math.sqrt(max(0.0, 1.0 - sin2_t))
    rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    rp = (n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i)
    return 0.5 * (rs * rs + rp * rp)


_fresnel_exact = fresnel
_fresnel_schlick = fresnel_schlick


def _random_in_unit_sphere(rng):
    while True:
        v = (rng.random() * 2.0 - 1.0,
             rng.random() * 2.0 - 1.0,
             rng.random() * 2.0 - 1.0)
        if _dot(v, v) < 1.0:
            return v


def _cosine_weighted_hemisphere(normal, rng):
    """绕法线的余弦加权半球采样（对 Lambert BRDF 的重要性采样）。"""
    r1 = rng.random()
    r2 = rng.random()
    radius = math.sqrt(r1)
    phi = 2.0 * math.pi * r2
    x = radius * math.cos(phi)
    y = radius * math.sin(phi)
    z = math.sqrt(max(0.0, 1.0 - r1))
    n = _normalize(normal)
    if abs(n[1]) < 0.999:
        up = (0.0, 1.0, 0.0)
    else:
        up = (1.0, 0.0, 0.0)
    tangent = _normalize(_cross(up, n))
    bitangent = _cross(n, tangent)
    return _normalize((
        tangent[0] * x + bitangent[0] * y + n[0] * z,
        tangent[1] * x + bitangent[1] * y + n[1] * z,
        tangent[2] * x + bitangent[2] * y + n[2] * z,
    ))


# ---------------------------------------------------------------------------
# 材质 / 光源 / 图元
# ---------------------------------------------------------------------------

@dataclass
class Material:
    albedo: Vec3 = (0.8, 0.8, 0.8)
    specular: float = 0.0
    shininess: float = 32.0
    reflectivity: float = 0.0       # 镜面反射率（多次反射递归）
    transparency: float = 0.0       # 电介质透明度（折射）
    refractive_index: float = 1.5   # 折射率（斯涅尔定律）
    emissive: Vec3 = (0.0, 0.0, 0.0)
    roughness: float = 0.0


def _as_material(value):
    if value is None:
        return Material()
    if isinstance(value, Material):
        return value
    if isinstance(value, dict):
        albedo = value.get("albedo",
                 value.get("color",
                 value.get("diffuse",
                 value.get("albedo_color", (0.8, 0.8, 0.8)))))
        emissive = value.get("emissive",
                   value.get("emission",
                   value.get("emit", (0.0, 0.0, 0.0))))
        return Material(
            albedo=_vec3(albedo, (0.8, 0.8, 0.8)),
            specular=float(value.get("specular", 0.0)),
            shininess=float(value.get("shininess",
                           value.get("specular_exponent", 32.0))),
            reflectivity=float(value.get("reflectivity",
                             value.get("reflectance",
                             value.get("mirror", 0.0)))),
            transparency=float(value.get("transparency",
                             value.get("transmittance",
                             value.get("translucency", 0.0)))),
            refractive_index=float(value.get("refractive_index",
                                  value.get("ior",
                                  value.get("index_of_refraction", 1.5)))),
            emissive=_vec3(emissive, (0.0, 0.0, 0.0)),
            roughness=float(value.get("roughness", 0.0)),
        )
    raise TypeError(f"不支持的材质类型: {type(value)!r}")


@dataclass
class Light:
    position: Vec3 = (0.0, 0.0, 0.0)
    color: Vec3 = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    size: float = 0.0              # 球形面光源半径，>0 时产生软阴影
    direction: Optional[Vec3] = None


def _as_light(value):
    if value is None:
        return Light()
    if isinstance(value, Light):
        return value
    if isinstance(value, dict):
        light_type = str(value.get("type", "point")).lower()
        direction = value.get("direction")
        position = value.get("position", value.get("pos", (0.0, 0.0, 0.0)))
        size = float(value.get("size",
                   value.get("radius",
                   value.get("width", 0.0))))
        if light_type in ("directional", "sun"):
            d = _normalize(_vec3(direction, (0.0, -1.0, 0.0)))
            position = _scale(d, -1.0e7)
            size = 0.0
        return Light(
            position=_vec3(position),
            color=_vec3(value.get("color", value.get("colour", (1.0, 1.0, 1.0)))),
            intensity=float(value.get("intensity",
                           value.get("brightness",
                           value.get("power", 1.0)))),
            size=size,
            direction=_vec3(direction) if direction is not None else None,
        )
    raise TypeError(f"不支持的光源类型: {type(value)!r}")


class Sphere:
    def __init__(self, center=(0.0, 0.0, 0.0), radius=1.0, material=None):
        self.center = _vec3(center)
        self.radius = float(radius)
        self.material = _as_material(material)

    def intersect(self, origin, direction):
        oc = _sub(origin, self.center)
        b = _dot(oc, direction)
        c = _dot(oc, oc) - self.radius * self.radius
        disc = b * b - c
        if disc < 0.0:
            return None
        root = math.sqrt(disc)
        t = -b - root
        if t < _EPS:
            t = -b + root
            if t < _EPS:
                return None
        return t

    def normal_at(self, point):
        return _normalize(_sub(point, self.center))

    def bounds(self):
        r = (self.radius, self.radius, self.radius)
        return _sub(self.center, r), _add(self.center, r)


class Plane:
    def __init__(self, point=(0.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0), material=None):
        self.point = _vec3(point)
        self.normal = _normalize(_vec3(normal))
        self.material = _as_material(material)

    def intersect(self, origin, direction):
        denom = _dot(self.normal, direction)
        if abs(denom) < 1e-9:
            return None
        t = _dot(_sub(self.point, origin), self.normal) / denom
        return t if t > _EPS else None

    def normal_at(self, point):
        return self.normal

    def bounds(self):
        big = 1.0e6
        lo = [self.point[i] - big for i in range(3)]
        hi = [self.point[i] + big for i in range(3)]
        for i in range(3):
            if abs(self.normal[i]) > 0.99:
                lo[i] = hi[i] = self.point[i]
        return (tuple(lo), tuple(hi))


class Triangle:
    def __init__(self, v0, v1, v2, material=None):
        self.v0 = _vec3(v0)
        self.v1 = _vec3(v1)
        self.v2 = _vec3(v2)
        self.material = _as_material(material)
        self._normal = _normalize(_cross(_sub(self.v1, self.v0),
                                         _sub(self.v2, self.v0)))

    def intersect(self, origin, direction):
        edge1 = _sub(self.v1, self.v0)
        edge2 = _sub(self.v2, self.v0)
        pvec = _cross(direction, edge2)
        det = _dot(edge1, pvec)
        if abs(det) < 1e-12:
            return None
        inv = 1.0 / det
        tvec = _sub(origin, self.v0)
        u = _dot(tvec, pvec) * inv
        if u < -1e-9 or u > 1.0 + 1e-9:
            return None
        qvec = _cross(tvec, edge1)
        v = _dot(direction, qvec) * inv
        if v < -1e-9 or u + v > 1.0 + 1e-9:
            return None
        t = _dot(edge2, qvec) * inv
        return t if t > _EPS else None

    def normal_at(self, point):
        return self._normal

    def bounds(self):
        lo = (min(self.v0[0], self.v1[0], self.v2[0]),
              min(self.v0[1], self.v1[1], self.v2[1]),
              min(self.v0[2], self.v1[2], self.v2[2]))
        hi = (max(self.v0[0], self.v1[0], self.v2[0]),
              max(self.v0[1], self.v1[1], self.v2[1]),
              max(self.v0[2], self.v1[2], self.v2[2]))
        return lo, hi


def _as_primitive(obj):
    if isinstance(obj, (Sphere, Plane, Triangle)):
        return obj
    if isinstance(obj, dict):
        kind = str(obj.get("type", obj.get("kind", obj.get("shape", "sphere")))).lower()
        mat = _as_material(obj.get("material", obj))
        if kind in ("sphere", "ball"):
            return Sphere(
                center=obj.get("center", obj.get("position", (0.0, 0.0, 0.0))),
                radius=float(obj.get("radius", 1.0)),
                material=mat,
            )
        if kind == "plane":
            return Plane(
                point=obj.get("point", obj.get("position", (0.0, 0.0, 0.0))),
                normal=obj.get("normal", (0.0, 1.0, 0.0)),
                material=mat,
            )
        if kind in ("triangle", "tri"):
            verts = obj.get("vertices")
            if verts is None:
                verts = (obj["v0"], obj["v1"], obj["v2"])
            return Triangle(verts[0], verts[1], verts[2], material=mat)
        raise ValueError(f"未知图元类型: {kind!r}")
    raise TypeError(f"不支持的图元类型: {type(obj)!r}")


@dataclass
class Config:
    pass


# ---------------------------------------------------------------------------
# BVH 加速结构
# ---------------------------------------------------------------------------

class _BVHNode:
    __slots__ = ("box_min", "box_max", "left", "right", "primitives")

    def __init__(self, box_min, box_max, left=None, right=None, primitives=None):
        self.box_min = box_min
        self.box_max = box_max
        self.left = left
        self.right = right
        self.primitives = primitives


def _union_box(a_min, a_max, b_min, b_max):
    return ((min(a_min[0], b_min[0]), min(a_min[1], b_min[1]), min(a_min[2], b_min[2])),
            (max(a_max[0], b_max[0]), max(a_max[1], b_max[1]), max(a_max[2], b_max[2])))


def _ray_box(origin, direction, box_min, box_max, t_max):
    t_min = 0.0
    for axis in range(3):
        d = direction[axis]
        if abs(d) < 1e-12:
            if origin[axis] < box_min[axis] or origin[axis] > box_max[axis]:
                return False
            continue
        inv = 1.0 / d
        t0 = (box_min[axis] - origin[axis]) * inv
        t1 = (box_max[axis] - origin[axis]) * inv
        if t0 > t1:
            t0, t1 = t1, t0
        t_min = max(t_min, t0)
        t_max = min(t_max, t1)
        if t_max < t_min:
            return False
    return True


class BVH:
    """层次包围盒（Bounding Volume Hierarchy）加速结构。"""

    def __init__(self, primitives):
        self.primitives = list(primitives)
        self.node_count = 0
        self.root = self._build(self.primitives) if self.primitives else None

    def _build(self, prims):
        box_min, box_max = prims[0].bounds()
        for prim in prims[1:]:
            lo, hi = prim.bounds()
            box_min, box_max = _union_box(box_min, box_max, lo, hi)
        self.node_count += 1
        if len(prims) <= 2:
            return _BVHNode(box_min, box_max, primitives=list(prims))
        centroids = []
        for prim in prims:
            lo, hi = prim.bounds()
            centroids.append(((lo[0] + hi[0]) * 0.5,
                              (lo[1] + hi[1]) * 0.5,
                              (lo[2] + hi[2]) * 0.5))
        extent = [box_max[i] - box_min[i] for i in range(3)]
        axis = extent.index(max(extent))
        ordered = sorted(zip(centroids, prims), key=lambda item: item[0][axis])
        mid = max(1, len(ordered) // 2)
        left = self._build([prim for _, prim in ordered[:mid]])
        right = self._build([prim for _, prim in ordered[mid:]])
        return _BVHNode(box_min, box_max, left=left, right=right)

    def intersect(self, origin, direction, stats=None, t_max=_INF):
        best_t = t_max
        best_prim = None
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node is None:
                continue
            if stats is not None:
                stats["box_tests"] = stats.get("box_tests", 0) + 1
            if not _ray_box(origin, direction, node.box_min, node.box_max, best_t):
                continue
            if node.primitives is not None:
                for prim in node.primitives:
                    if stats is not None:
                        stats["intersection_tests"] = stats.get("intersection_tests", 0) + 1
                    t = prim.intersect(origin, direction)
                    if t is not None and _EPS < t < best_t:
                        best_t = t
                        best_prim = prim
            else:
                stack.append(node.left)
                stack.append(node.right)
        if best_prim is None:
            return None
        return best_t, best_prim


# ---------------------------------------------------------------------------
# 结果类型
# ---------------------------------------------------------------------------

class TraceResult(tuple):
    """
    颜色三元组，同时支持字符串键访问附加信息：
    result[0]/result["r"]/result["color"]/result["depth"] 等。
    """

    _ALIASES = ("color", "colour", "radiance", "rgb")

    def __new__(cls, color, extra=None):
        c = _vec3(color)
        obj = super().__new__(cls, (c[0], c[1], c[2]))
        obj._extra = dict(extra or {})
        return obj

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in self._ALIASES:
                return tuple(self)
            if key in ("r", "red"):
                return tuple.__getitem__(self, 0)
            if key in ("g", "green"):
                return tuple.__getitem__(self, 1)
            if key in ("b", "blue"):
                return tuple.__getitem__(self, 2)
            return self._extra[key]
        return tuple.__getitem__(self, key)

    def __contains__(self, key):
        if isinstance(key, str):
            return key in self._ALIASES or key in ("r", "g", "b") or key in self._extra
        return tuple.__contains__(self, key)

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, IndexError, TypeError):
            return default

    def keys(self):
        return list(self._ALIASES) + list(self._extra.keys())

    @property
    def r(self):
        return tuple.__getitem__(self, 0)

    @property
    def g(self):
        return tuple.__getitem__(self, 1)

    @property
    def b(self):
        return tuple.__getitem__(self, 2)

    @property
    def color(self):
        return tuple(self)

    @property
    def radiance(self):
        return tuple(self)


class _AttrDict(dict):
    """支持属性访问的字典。"""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


# ---------------------------------------------------------------------------
# 光子映射（焦散）
# ---------------------------------------------------------------------------

class PhotonMap:
    """光子映射：存储光子并支持辐照度估计与体积焦散查询。"""

    def __init__(self, photons=None):
        # 每个光子: (position, power, direction)
        self.photons = list(photons or [])

    def __len__(self):
        return len(self.photons)

    def add(self, position, power, direction=(0.0, 0.0, 0.0)):
        self.photons.append((_vec3(position), _vec3(power), _vec3(direction)))

    def nearest(self, point, k=50):
        p = _vec3(point)
        dists = []
        for photon in self.photons:
            d2 = _dot(_sub(photon[0], p), _sub(photon[0], p))
            dists.append((d2, photon))
        dists.sort(key=lambda item: item[0])
        return dists[:max(1, int(k))]

    def irradiance_at(self, point, radius=None, k=None):
        """
        辐照度估计：E = Σ power / (π r²)。
        优先使用 k 近邻；否则收集半径 radius 内的光子。
        """
        if not self.photons:
            return (0.0, 0.0, 0.0)
        p = _vec3(point)
        if k is None and radius is None:
            k = 50
        if k is not None:
            neighbours = self.nearest(p, k)
            if not neighbours:
                return (0.0, 0.0, 0.0)
            r = max(math.sqrt(neighbours[-1][0]), 1e-6)
            total = (0.0, 0.0, 0.0)
            for _, photon in neighbours:
                total = _add(total, photon[1])
            return _scale(total, 1.0 / (math.pi * r * r))
        radius = float(radius)
        r2 = radius * radius
        total = (0.0, 0.0, 0.0)
        for photon in self.photons:
            d2 = _dot(_sub(photon[0], p), _sub(photon[0], p))
            if d2 <= r2:
                total = _add(total, photon[1])
        return _scale(total, 1.0 / (math.pi * r2))

    def density_at(self, point, radius=0.5):
        """体积内光子密度估计：count / (4/3 π r³)。"""
        p = _vec3(point)
        r2 = radius * radius
        count = 0
        for photon in self.photons:
            d2 = _dot(_sub(photon[0], p), _sub(photon[0], p))
            if d2 <= r2:
                count += 1
        volume = (4.0 / 3.0) * math.pi * radius ** 3
        return count / max(volume, 1e-12)

    def volumetric_radiance(self, origin, direction, max_dist=10.0, step=0.2,
                            radius=0.5, scattering=1.0, extinction=0.05):
        """
        体积焦散：沿视线在参与介质中步进，累积光子散射贡献，
        并按消光系数做指数衰减。
        """
        o = _vec3(origin)
        d = _normalize(_vec3(direction))
        radiance = (0.0, 0.0, 0.0)
        transmittance = 1.0
        t = step * 0.5
        while t < max_dist and transmittance > 1e-4:
            p = _add(o, _scale(d, t))
            irradiance = self.irradiance_at(p, radius=radius)
            radiance = _add(radiance, _scale(irradiance, transmittance * scattering * step))
            transmittance *= math.exp(-extinction * step)
            t += step
        return radiance


# ---------------------------------------------------------------------------
# 主系统
# ---------------------------------------------------------------------------

class RayTracing:
    """游戏内光线追踪与路径追踪系统。"""

    DEFAULT_CONFIG = {
        "max_depth": 4,               # 递归深度（多次反射/折射）
        "shadow_samples": 16,         # 软阴影采样数（面光源）
        "num_samples": 64,            # 路径追踪蒙特卡洛采样数
        "path_max_depth": 5,          # 路径追踪最大弹射次数
        "photon_count": 2000,         # 焦散光子数
        "photon_gather_radius": 0.5,  # 光子收集半径
        "photon_gather_k": 50,        # 光子 k 近邻
        "radiosity_iterations": 200,  # 辐射度迭代上限
        "radiosity_tolerance": 1e-6,  # 辐射度收敛阈值
        "background": (0.0, 0.0, 0.0),
        "ambient": (0.03, 0.03, 0.03),
        "epsilon": 1e-4,
        "seed": 0,
    }

    def __init__(self, config: Optional[Dict] = None):
        self.config = dict(self.DEFAULT_CONFIG)
        if config:
            self.config.update(config)
        self._state = {}
        self._history = []
        self._objects: List = []
        self._lights: List = []
        self._bvh: Optional[BVH] = None
        self.stats = self._new_stats()

    # -- 生命周期 -----------------------------------------------------------

    def update(self, dt: float):
        self._state["time"] = self._state.get("time", 0.0) + float(dt)
        self._state["last_update"] = time.time()
        self._history.append(float(dt))

    def reset(self):
        self._state = {}
        self._history = []
        self._objects = []
        self._lights = []
        self._bvh = None
        self.stats = self._new_stats()

    @staticmethod
    def _new_stats():
        return {
            "rays_traced": 0,
            "intersection_tests": 0,
            "box_tests": 0,
            "shadow_rays": 0,
            "photons_traced": 0,
            "photons_stored": 0,
        }

    # -- 场景管理 -----------------------------------------------------------

    def set_scene(self, objects=None, lights=None):
        if objects is not None:
            self._objects = [_as_primitive(o) for o in objects]
        if lights is not None:
            self._lights = [_as_light(l) for l in lights]
        self._bvh = None
        return self

    def add_object(self, obj):
        self._objects.append(_as_primitive(obj))
        self._bvh = None
        return self

    def add_light(self, light):
        self._lights.append(_as_light(light))
        return self

    def build_acceleration_structure(self, objects=None):
        """构建 BVH 加速结构。"""
        if objects is not None:
            self._objects = [_as_primitive(o) for o in objects]
        self._bvh = BVH(self._objects)
        return self._bvh

    build_bvh = build_acceleration_structure

    def _resolve_scene(self, objects, lights):
        if objects is not None:
            prims = [_as_primitive(o) for o in objects]
            bvh = BVH(prims) if prims else None
        else:
            prims = self._objects
            if self._bvh is None and prims:
                self._bvh = BVH(prims)
            bvh = self._bvh
        lts = [_as_light(l) for l in lights] if lights is not None else list(self._lights)
        return prims, lts, bvh

    # -- 求交 ---------------------------------------------------------------

    def _nearest(self, origin, direction, prims, bvh, use_bvh, t_max=_INF):
        self.stats["rays_traced"] += 1
        if use_bvh and bvh is not None:
            return bvh.intersect(origin, direction, self.stats, t_max)
        best_t = t_max
        best_prim = None
        for prim in prims:
            self.stats["intersection_tests"] += 1
            t = prim.intersect(origin, direction)
            if t is not None and _EPS < t < best_t:
                best_t = t
                best_prim = prim
        if best_prim is None:
            return None
        return best_t, best_prim

    def intersect(self, origin, direction, objects=None, use_bvh=True):
        """求最近交点，返回命中信息字典或 None。"""
        prims, _, bvh = self._resolve_scene(objects, None)
        o = _vec3(origin)
        d = _normalize(_vec3(direction))
        hit = self._nearest(o, d, prims, bvh, use_bvh)
        if hit is None:
            return None
        t, prim = hit
        point = _add(o, _scale(d, t))
        return _AttrDict(
            t=t,
            distance=t,
            point=point,
            position=point,
            normal=prim.normal_at(point),
            material=prim.material,
            object=prim,
            primitive=prim,
        )

    # -- 1. 光线追踪（BVH 加速 + 递归 + 软阴影） ------------------------------

    def trace_ray(self, origin, direction, objects=None, lights=None, depth=None,
                  max_depth=None, background=None, use_bvh=None, scene=None, **kwargs):
        """
        递归光线追踪：
        - BVH 加速结构求交（use_bvh=False 可退回暴力遍历对比）
        - 递归多次镜面反射与折射（菲涅尔加权）
        - 面光源软阴影
        """
        if scene:
            if objects is None:
                objects = scene.get("objects")
            if lights is None:
                lights = scene.get("lights")
        prims, lts, bvh = self._resolve_scene(objects, lights)
        o = _vec3(origin)
        d = _normalize(_vec3(direction))
        if max_depth is None:
            max_depth = depth if depth is not None else int(self.config["max_depth"])
        max_depth = max(0, int(max_depth))
        bg = _vec3(background, self.config["background"]) if background is not None \
            else _vec3(self.config["background"])
        ub = True if use_bvh is None else bool(use_bvh)
        rng = random.Random(int(self.config.get("seed", 0)))
        color = self._trace(o, d, prims, lts, bvh, ub, max_depth, bg, rng)
        return TraceResult(color, extra={
            "depth": max_depth,
            "max_depth": max_depth,
            "background": bg,
            "acceleration_structure": "bvh" if ub else "brute_force",
            "use_bvh": ub,
            "recursive": True,
            "soft_shadows": True,
            "fresnel": True,
            "method": "recursive_ray_tracing",
            "stats": self.stats,
            "intersection_tests": self.stats["intersection_tests"],
        })

    def _trace(self, origin, direction, prims, lights, bvh, use_bvh, depth, bg, rng):
        hit = self._nearest(origin, direction, prims, bvh, use_bvh)
        if hit is None:
            return bg
        t, prim = hit
        point = _add(origin, _scale(direction, t))
        geo_normal = prim.normal_at(point)
        mat = prim.material
        entering = _dot(direction, geo_normal) < 0.0
        normal = geo_normal if entering else _scale(geo_normal, -1.0)
        eps = float(self.config["epsilon"])

        # 局部光照：自发光 + 环境光 + 直接光照（软阴影）
        color = _add(_vec3(mat.emissive),
                     _mul(_vec3(self.config["ambient"]), mat.albedo))
        view = _scale(direction, -1.0)
        for light in lights:
            color = _add(color, self._shade_light(point, normal, view, mat, light,
                                                  prims, bvh, use_bvh, rng))

        if depth > 0:
            # 电介质：菲涅尔加权的反射 + 折射（递归）
            if mat.transparency > 0.0:
                n1, n2 = (1.0, mat.refractive_index) if entering \
                    else (mat.refractive_index, 1.0)
                cos_i = min(1.0, -_dot(direction, normal))
                kr = _fresnel_exact(cos_i, n1, n2)
                refl_dir = _reflect(direction, normal)
                refl_origin = _add(point, _scale(normal, eps))
                refl_color = self._trace(refl_origin, refl_dir, prims, lights, bvh,
                                         use_bvh, depth - 1, bg, rng)
                refr_dir = _refract(direction, normal, n1, n2)
                if refr_dir is None:
                    dielectric = refl_color  # 全内反射
                else:
                    refr_origin = _add(point, _scale(normal, -eps))
                    refr_color = self._trace(refr_origin, refr_dir, prims, lights,
                                             bvh, use_bvh, depth - 1, bg, rng)
                    dielectric = _add(_scale(refl_color, kr),
                                      _scale(refr_color, 1.0 - kr))
                color = _add(_scale(color, 1.0 - mat.transparency),
                             _scale(dielectric, mat.transparency))
            # 镜面反射：递归多次反射
            if mat.reflectivity > 0.0:
                refl_dir = _reflect(direction, normal)
                refl_origin = _add(point, _scale(normal, eps))
                mirror = self._trace(refl_origin, refl_dir, prims, lights, bvh,
                                     use_bvh, depth - 1, bg, rng)
                color = _add(_scale(color, 1.0 - mat.reflectivity),
                             _scale(mirror, mat.reflectivity))
        return color

    def _shade_light(self, point, normal, view, mat, light, prims, bvh, use_bvh, rng):
        to_light = _sub(light.position, point)
        dist = _length(to_light)
        if dist < _EPS:
            return (0.0, 0.0, 0.0)
        light_dir = _scale(to_light, 1.0 / dist)
        ndl = _dot(normal, light_dir)
        if ndl <= 0.0:
            return (0.0, 0.0, 0.0)
        visibility = self._shadow_visibility(point, normal, light, prims, bvh,
                                             use_bvh, rng)
        if visibility <= 0.0:
            return (0.0, 0.0, 0.0)
        radiance = _scale(_vec3(light.color), light.intensity * visibility * ndl)
        diffuse = _mul(mat.albedo, radiance)
        specular = (0.0, 0.0, 0.0)
        if mat.specular > 0.0:
            half = _normalize(_add(light_dir, view))
            spec = max(0.0, _dot(normal, half)) ** mat.shininess
            specular = _scale(radiance, mat.specular * spec)
        return _add(diffuse, specular)

    def _shadow_visibility(self, point, normal, light, prims, bvh, use_bvh, rng):
        """软阴影：对面光源（球形）做多次采样，返回可见比例（0..1 的半影）。"""
        eps = float(self.config["epsilon"])
        size = max(0.0, float(light.size))
        samples = max(1, int(self.config["shadow_samples"])) if size > 0.0 else 1
        unblocked = 0
        for _ in range(samples):
            if size > 0.0:
                target = _add(light.position, _scale(_random_in_unit_sphere(rng), size))
            else:
                target = light.position
            to_target = _sub(target, point)
            dist = _length(to_target)
            if dist < eps:
                unblocked += 1
                continue
            shadow_dir = _scale(to_target, 1.0 / dist)
            origin = _add(point, _scale(normal, eps))
            self.stats["shadow_rays"] += 1
            blocker = self._nearest(origin, shadow_dir, prims, bvh, use_bvh,
                                    t_max=dist - eps)
            if blocker is None:
                unblocked += 1
        return unblocked / samples

    def soft_shadow(self, point, light, objects=None, normal=(0.0, 1.0, 0.0)):
        """公开接口：计算某点对光源的软阴影可见度（0..1）。"""
        prims, _, bvh = self._resolve_scene(objects, None)
        rng = random.Random(int(self.config.get("seed", 0)))
        return self._shadow_visibility(_vec3(point), _normalize(_vec3(normal)),
                                       _as_light(light), prims, bvh, True, rng)

    # -- 2. 路径追踪（蒙特卡洛积分 + 重要性采样） ------------------------------

    def path_tracing(self, origin, direction, objects=None, lights=None,
                     num_samples=None, max_depth=None, seed=None, use_bvh=None,
                     background=None, scene=None, **kwargs):
        """
        蒙特卡洛路径追踪：
        - 对像素做 num_samples 次采样并平均（蒙特卡洛积分）
        - 余弦加权半球采样（对 Lambert BRDF 的重要性采样）
        - 光源采样（下一事件估计，对光源的重要性采样）
        - 俄罗斯轮盘终止
        返回 TraceResult，附加 variance/samples 等统计信息。
        """
        if scene:
            if objects is None:
                objects = scene.get("objects")
            if lights is None:
                lights = scene.get("lights")
        if num_samples is None:
            num_samples = kwargs.get("samples", kwargs.get("spp",
                                int(self.config["num_samples"])))
        num_samples = max(1, int(num_samples))
        if max_depth is None:
            max_depth = kwargs.get("depth", int(self.config["path_max_depth"]))
        max_depth = max(1, int(max_depth))
        prims, lts, bvh = self._resolve_scene(objects, lights)
        o = _vec3(origin)
        d = _normalize(_vec3(direction))
        bg = _vec3(background, self.config["background"]) if background is not None \
            else _vec3(self.config["background"])
        ub = True if use_bvh is None else bool(use_bvh)
        rng = random.Random(int(seed if seed is not None else self.config.get("seed", 0)))

        total = (0.0, 0.0, 0.0)
        total_sq = (0.0, 0.0, 0.0)
        for _ in range(num_samples):
            sample = self._path_sample(o, d, prims, lts, bvh, ub, max_depth, bg, rng)
            total = _add(total, sample)
            total_sq = _add(total_sq, _mul(sample, sample))
        inv = 1.0 / num_samples
        mean = _scale(total, inv)
        # 单样本方差：E[X²] - E[X]²
        sample_variance = _clamp_color(_sub(_scale(total_sq, inv), _mul(mean, mean)))
        # 蒙特卡洛估计量（均值）的方差：随采样数按 1/N 收敛
        variance = _scale(sample_variance, inv)
        return TraceResult(mean, extra={
            "num_samples": num_samples,
            "samples": num_samples,
            "variance": variance,
            "sample_variance": sample_variance,
            "std": tuple(math.sqrt(v) for v in variance),
            "max_depth": max_depth,
            "monte_carlo": True,
            "importance_sampling": "cosine_weighted_hemisphere",
            "next_event_estimation": bool(lts),
            "russian_roulette": True,
            "method": "monte_carlo_path_tracing",
            "stats": self.stats,
        })

    def _path_sample(self, origin, direction, prims, lights, bvh, use_bvh,
                     max_depth, bg, rng):
        throughput = (1.0, 1.0, 1.0)
        radiance = (0.0, 0.0, 0.0)
        o, d = origin, direction
        specular_bounce = True
        eps = float(self.config["epsilon"])
        for bounce in range(max_depth):
            hit = self._nearest(o, d, prims, bvh, use_bvh)
            if hit is None:
                radiance = _add(radiance, _mul(throughput, bg))
                break
            t, prim = hit
            point = _add(o, _scale(d, t))
            geo_normal = prim.normal_at(point)
            mat = prim.material
            entering = _dot(d, geo_normal) < 0.0
            normal = geo_normal if entering else _scale(geo_normal, -1.0)
            emissive = _vec3(mat.emissive)
            if emissive != (0.0, 0.0, 0.0) and (specular_bounce or not lights):
                radiance = _add(radiance, _mul(throughput, emissive))
            is_specular = mat.transparency > 0.5 or mat.reflectivity > 0.5
            # 下一事件估计：对光源的重要性采样
            if lights and not is_specular:
                radiance = _add(radiance,
                                _mul(throughput,
                                     self._sample_lights(point, normal, mat, prims,
                                                         lights, bvh, use_bvh, rng)))
            if mat.transparency > 0.5:
                # 电介质：按菲涅尔概率随机走反射或折射路径
                n1, n2 = (1.0, mat.refractive_index) if entering \
                    else (mat.refractive_index, 1.0)
                cos_i = min(1.0, -_dot(d, normal))
                kr = _fresnel_exact(cos_i, n1, n2)
                refr_dir = _refract(d, normal, n1, n2)
                if refr_dir is None or rng.random() < kr:
                    d = _reflect(d, normal)
                    o = _add(point, _scale(normal, eps))
                else:
                    d = refr_dir
                    o = _add(point, _scale(normal, -eps))
                specular_bounce = True
            elif mat.reflectivity > 0.5:
                d = _reflect(d, normal)
                o = _add(point, _scale(normal, eps))
                throughput = _mul(throughput, mat.albedo)
                specular_bounce = True
            else:
                # 余弦加权半球重要性采样（Lambert：throughput *= albedo）
                d = _cosine_weighted_hemisphere(normal, rng)
                o = _add(point, _scale(normal, eps))
                throughput = _mul(throughput, mat.albedo)
                specular_bounce = False
            # 俄罗斯轮盘
            if bounce >= 3:
                q = min(0.95, max(0.05, max(throughput)))
                if rng.random() > q:
                    break
                throughput = _scale(throughput, 1.0 / q)
        return radiance

    def _sample_lights(self, point, normal, mat, prims, lights, bvh, use_bvh, rng):
        """光源重要性采样（下一事件估计）。"""
        eps = float(self.config["epsilon"])
        brdf = _scale(mat.albedo, 1.0 / math.pi)  # Lambert: albedo / π
        direct = (0.0, 0.0, 0.0)
        for light in lights:
            if light.size > 0.0:
                # 面光源：在光源立体角内采样
                target = _add(light.position,
                              _scale(_random_in_unit_sphere(rng), light.size))
                to_light = _sub(light.position, point)
                dist2 = max(_dot(to_light, to_light), _EPS)
                solid_angle = min(2.0 * math.pi,
                                  math.pi * light.size * light.size / dist2)
                weight = solid_angle
            else:
                target = light.position
                weight = 1.0
            to_target = _sub(target, point)
            dist = _length(to_target)
            if dist < eps:
                continue
            wi = _scale(to_target, 1.0 / dist)
            cos_x = _dot(normal, wi)
            if cos_x <= 0.0:
                continue
            origin = _add(point, _scale(normal, eps))
            self.stats["shadow_rays"] += 1
            blocker = self._nearest(origin, wi, prims, bvh, use_bvh, t_max=dist - eps)
            if blocker is not None:
                continue
            contrib = _mul(brdf, _scale(_vec3(light.color),
                                        light.intensity * cos_x * weight))
            direct = _add(direct, contrib)
        return direct

    # -- 3. 全局光照（辐射度 + 颜色溢出） ---------------------------------------

    def global_illumination(self, patches=None, objects=None, max_iterations=None,
                            tolerance=None, **kwargs):
        """
        辐射度（Radiosity）全局光照求解：
        - 计算面片间形状因子（含遮挡测试）
        - 迭代求解 B = E + ρ·F·B，得到间接光照
        - 反照率 ρ 为彩色，自然产生颜色溢出（color bleeding）
        返回每个面片的出射辐亮度（radiosity）。
        """
        if patches is None:
            patches = kwargs.get("surfaces", kwargs.get("quads"))
        if max_iterations is None:
            max_iterations = kwargs.get("iterations",
                              int(self.config["radiosity_iterations"]))
        if tolerance is None:
            tolerance = float(kwargs.get("tol", self.config["radiosity_tolerance"]))
        patch_list = [self._as_patch(p) for p in (patches or [])]
        if not patch_list and (objects is not None or self._objects):
            prims, _, _ = self._resolve_scene(objects, None)
            patch_list = [self._primitive_as_patch(p) for p in prims]
        occluders = []
        if objects is not None:
            occluders = [_as_primitive(o) for o in objects]
        n = len(patch_list)
        if n == 0:
            return _AttrDict(radiosity=[], colors=[], patches=[], form_factors=[],
                             iterations=0, converged=True, method="radiosity")

        # 形状因子 F_ij = cos_i·cos_j / (π·r²) · A_j · V_ij
        form_factors = [[(0.0) for _ in range(n)] for _ in range(n)]
        eps = float(self.config["epsilon"])
        occ_bvh = BVH(occluders) if occluders else None
        for i in range(n):
            pi = patch_list[i]
            for j in range(n):
                if i == j:
                    continue
                pj = patch_list[j]
                delta = _sub(pj["position"], pi["position"])
                dist2 = _dot(delta, delta)
                if dist2 < eps:
                    continue
                dist = math.sqrt(dist2)
                direction = _scale(delta, 1.0 / dist)
                cos_i = _dot(pi["normal"], direction)
                cos_j = _dot(pj["normal"], _scale(direction, -1.0))
                if cos_i <= 0.0 or cos_j <= 0.0:
                    continue
                visibility = 1.0
                if occluders:
                    origin = _add(pi["position"], _scale(pi["normal"], eps))
                    blocker = self._nearest(origin, direction, occluders, occ_bvh,
                                            True, t_max=dist - eps)
                    visibility = 0.0 if blocker is not None else 1.0
                form_factors[i][j] = (cos_i * cos_j / (math.pi * dist2)) \
                    * pj["area"] * visibility

        # 迭代求解 B = E + ρ ⊙ (F·B)（Gauss-Seidel，含间接光照与颜色溢出）
        radiosity = [pj["emissive"] for pj in patch_list]
        iterations = 0
        converged = False
        for iterations in range(1, max(1, int(max_iterations)) + 1):
            delta_max = 0.0
            for i in range(n):
                acc = patch_list[i]["emissive"]
                albedo = patch_list[i]["albedo"]
                for j in range(n):
                    f = form_factors[i][j]
                    if f > 0.0:
                        acc = _add(acc, _mul(albedo, _scale(radiosity[j], f)))
                for c in range(3):
                    delta_max = max(delta_max, abs(acc[c] - radiosity[i][c]))
                radiosity[i] = acc
            if delta_max < tolerance:
                converged = True
                break

        out_patches = []
        for i, pj in enumerate(patch_list):
            out_patches.append(_AttrDict(
                position=pj["position"],
                normal=pj["normal"],
                area=pj["area"],
                albedo=pj["albedo"],
                emissive=pj["emissive"],
                radiosity=radiosity[i],
                color=radiosity[i],
            ))
        return _AttrDict(
            radiosity=list(radiosity),
            colors=list(radiosity),
            exitance=list(radiosity),
            patches=out_patches,
            form_factors=form_factors,
            iterations=iterations,
            converged=converged,
            indirect_light=[_sub(radiosity[i], patch_list[i]["emissive"])
                            for i in range(n)],
            color_bleeding=True,
            method="radiosity",
        )

    @staticmethod
    def _as_patch(patch):
        if isinstance(patch, dict):
            position = patch.get("position", patch.get("center",
                       patch.get("point", (0.0, 0.0, 0.0))))
            normal = patch.get("normal", (0.0, 1.0, 0.0))
            area = patch.get("area")
            if area is None:
                size = patch.get("size", 1.0)
                if isinstance(size, (list, tuple)):
                    area = float(size[0]) * float(size[1])
                else:
                    area = float(size) * float(size)
            albedo = patch.get("albedo", patch.get("color",
                     patch.get("reflectance", (0.8, 0.8, 0.8))))
            emissive = patch.get("emissive", patch.get("emission",
                       patch.get("emit", (0.0, 0.0, 0.0))))
            return {
                "position": _vec3(position),
                "normal": _normalize(_vec3(normal)),
                "area": float(area),
                "albedo": _vec3(albedo, (0.8, 0.8, 0.8)),
                "emissive": _vec3(emissive, (0.0, 0.0, 0.0)),
            }
        # 对象形式（带属性）
        return {
            "position": _vec3(getattr(patch, "position",
                            getattr(patch, "center", (0.0, 0.0, 0.0)))),
            "normal": _normalize(_vec3(getattr(patch, "normal", (0.0, 1.0, 0.0)))),
            "area": float(getattr(patch, "area", 1.0)),
            "albedo": _vec3(getattr(patch, "albedo",
                          getattr(patch, "color", (0.8, 0.8, 0.8)))),
            "emissive": _vec3(getattr(patch, "emissive", (0.0, 0.0, 0.0))),
        }

    @staticmethod
    def _primitive_as_patch(prim):
        if isinstance(prim, Sphere):
            return {"position": prim.center, "normal": (0.0, 1.0, 0.0),
                    "area": 4.0 * math.pi * prim.radius ** 2,
                    "albedo": prim.material.albedo,
                    "emissive": _vec3(prim.material.emissive)}
        if isinstance(prim, Plane):
            return {"position": prim.point, "normal": prim.normal, "area": 1.0,
                    "albedo": prim.material.albedo,
                    "emissive": _vec3(prim.material.emissive)}
        edge1 = _sub(prim.v1, prim.v0)
        edge2 = _sub(prim.v2, prim.v0)
        area = 0.5 * _length(_cross(edge1, edge2))
        center = ((prim.v0[0] + prim.v1[0] + prim.v2[0]) / 3.0,
                  (prim.v0[1] + prim.v1[1] + prim.v2[1]) / 3.0,
                  (prim.v0[2] + prim.v1[2] + prim.v2[2]) / 3.0)
        return {"position": center, "normal": prim.normal_at(center), "area": area,
                "albedo": prim.material.albedo,
                "emissive": _vec3(prim.material.emissive)}

    # -- 4. 反射 / 折射（菲涅尔 + 斯涅尔） --------------------------------------

    def reflection_refraction(self, direction, normal, n1=1.0, n2=None,
                              material=None, **kwargs):
        """
        菲涅尔效应 + 斯涅尔定律：
        - 反射方向与折射方向（折射率为 n1 -> n2）
        - 菲涅尔反射率随入射角变化（掠射角反射更强）
        - 全内反射检测
        """
        d = _normalize(_vec3(direction))
        n = _normalize(_vec3(normal))
        if material is not None:
            mat = _as_material(material)
            if n2 is None:
                n2 = mat.refractive_index
        if n2 is None:
            n2 = kwargs.get("ior", kwargs.get("refractive_index", 1.5))
        n1 = float(n1)
        n2 = float(n2)
        if _dot(d, n) > 0.0:
            # 从介质内部射出：翻转法线并交换折射率
            n = _scale(n, -1.0)
            n1, n2 = n2, n1
        cos_i = min(1.0, max(0.0, -_dot(d, n)))
        reflectance = _fresnel_exact(cos_i, n1, n2)
        schlick = _fresnel_schlick(cos_i, n1, n2)
        reflected = _reflect(d, n)
        refracted = _refract(d, n, n1, n2)
        tir = refracted is None
        transmittance = 0.0 if tir else 1.0 - reflectance
        return _AttrDict(
            reflected=reflected,
            reflection_dir=reflected,
            reflect_dir=reflected,
            reflection=reflected,
            refracted=refracted,
            refraction_dir=refracted,
            refract_dir=refracted,
            refraction=refracted,
            reflectance=reflectance,
            transmittance=transmittance,
            fresnel=reflectance,
            fresnel_exact=reflectance,
            fresnel_schlick=schlick,
            total_internal_reflection=tir,
            tir=tir,
            n1=n1,
            n2=n2,
            cos_i=cos_i,
            snell=True,
        )

    # -- 5. 焦散（光子映射 + 体积焦散） ------------------------------------------

    def caustics(self, light=None, objects=None, receiver=None, num_photons=None,
                 query_point=None, origin=None, direction=None, view_ray=None,
                 gather_radius=None, gather_k=None, max_depth=None, seed=None,
                 **kwargs):
        """
        光子映射焦散：
        - 从光源向场景发射光子，经透明物体按斯涅尔定律折射聚焦
        - 光子在漫反射接收面上沉积，形成光子图（焦散图案）
        - 支持体积焦散：给定视线时在参与介质中步进累积散射
        返回包含 photon_map / irradiance / volumetric 的字典。
        """
        rng = random.Random(int(seed if seed is not None else self.config.get("seed", 0)))
        if light is not None:
            light_obj = _as_light(light)
        elif self._lights:
            light_obj = self._lights[0]
        else:
            light_obj = Light(position=(0.0, 5.0, 0.0))
        if objects is not None:
            objs = [_as_primitive(o) for o in objects]
        else:
            objs = list(self._objects)
        if receiver is not None:
            receivers = [_as_primitive(r) for r in
                         (receiver if isinstance(receiver, (list, tuple)) else [receiver])]
        else:
            receivers = [p for p in objs
                         if p.material.transparency <= 0.0 and p.material.reflectivity < 1.0]
            if not receivers:
                receivers = [Plane(point=(0.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0),
                                   material=Material(albedo=(0.9, 0.9, 0.9)))]
        # 去重后合并场景
        scene_prims = []
        seen = set()
        for prim in list(objs) + list(receivers):
            if id(prim) not in seen:
                seen.add(id(prim))
                scene_prims.append(prim)
        bvh = BVH(scene_prims) if scene_prims else None

        if num_photons is None:
            num_photons = kwargs.get("photon_count", kwargs.get("photons",
                                   int(self.config["photon_count"])))
        num_photons = max(1, int(num_photons))
        if max_depth is None:
            max_depth = int(kwargs.get("photon_max_depth", 8))
        center, radius = _target_sphere(scene_prims)
        power = _scale(_vec3(light_obj.color),
                       light_obj.intensity / float(num_photons))
        photons = []
        for _ in range(num_photons):
            emit_dir = _emit_direction(rng, light_obj.position, center, radius)
            self.stats["photons_traced"] += 1
            self._trace_photon(light_obj.position, emit_dir, power, scene_prims,
                               bvh, photons, max_depth, rng)
        self.stats["photons_stored"] += len(photons)
        photon_map = PhotonMap(photons)

        radius_gather = gather_radius if gather_radius is not None \
            else float(self.config["photon_gather_radius"])
        k_gather = gather_k if gather_k is not None \
            else int(self.config["photon_gather_k"])
        result = _AttrDict(
            photon_map=photon_map,
            photons=photons,
            num_photons=len(photons),
            photon_count=len(photons),
            light=light_obj,
            irradiance_at=photon_map.irradiance_at,
            volumetric_radiance=photon_map.volumetric_radiance,
            method="photon_mapping",
            volumetric=True,
        )
        if query_point is not None:
            result["irradiance"] = photon_map.irradiance_at(
                _vec3(query_point), radius=radius_gather, k=k_gather)
        if view_ray is not None:
            origin, direction = view_ray
        if origin is not None and direction is not None:
            result["volumetric"] = photon_map.volumetric_radiance(
                _vec3(origin), _normalize(_vec3(direction)), radius=radius_gather)
        return result

    def _trace_photon(self, origin, direction, power, prims, bvh, photons,
                      depth, rng):
        if depth <= 0:
            return
        hit = self._nearest(origin, direction, prims, bvh, True)
        if hit is None:
            return
        t, prim = hit
        point = _add(origin, _scale(direction, t))
        geo_normal = prim.normal_at(point)
        mat = prim.material
        entering = _dot(direction, geo_normal) < 0.0
        normal = geo_normal if entering else _scale(geo_normal, -1.0)
        eps = float(self.config["epsilon"])
        if mat.transparency > 0.0:
            # 透明物体：按菲涅尔概率走反射或折射（斯涅尔定律聚焦光线）
            n1, n2 = (1.0, mat.refractive_index) if entering \
                else (mat.refractive_index, 1.0)
            cos_i = min(1.0, -_dot(direction, normal))
            kr = _fresnel_exact(cos_i, n1, n2)
            refr_dir = _refract(direction, normal, n1, n2)
            if refr_dir is None or rng.random() < kr:
                new_dir = _reflect(direction, normal)
                new_origin = _add(point, _scale(normal, eps))
            else:
                new_dir = refr_dir
                new_origin = _add(point, _scale(normal, -eps))
            self._trace_photon(new_origin, new_dir, power, prims, bvh, photons,
                               depth - 1, rng)
        elif mat.reflectivity >= 1.0 - 1e-6:
            # 纯镜面：反射继续传播
            new_dir = _reflect(direction, normal)
            new_origin = _add(point, _scale(normal, eps))
            self._trace_photon(new_origin, new_dir, _mul(power, mat.albedo),
                               prims, bvh, photons, depth - 1, rng)
        else:
            # 漫反射面：光子沉积到光子图（焦散接收面）
            photons.append((point, _vec3(power), direction))

    def volumetric_caustics(self, origin=None, direction=None, photon_map=None,
                            max_dist=10.0, step=None, radius=None, **kwargs):
        """
        体积焦散：沿视线在参与介质中步进，累积光子图的内散射贡献。
        未提供 photon_map 时自动构建（可传入 light/objects/receiver 等参数）。
        """
        if photon_map is None:
            caustic_kwargs = {k: kwargs[k] for k in
                              ("light", "objects", "receiver", "num_photons",
                               "seed", "max_depth") if k in kwargs}
            photon_map = self.caustics(**caustic_kwargs)["photon_map"]
        elif isinstance(photon_map, dict):
            photon_map = photon_map.get("photon_map", photon_map)
        o = _vec3(origin, (0.0, 5.0, 0.0))
        d = _normalize(_vec3(direction, (0.0, -1.0, 0.0)))
        step = float(step if step is not None else 0.2)
        radius = float(radius if radius is not None
                       else self.config["photon_gather_radius"])
        scattering = float(kwargs.get("scattering", 1.0))
        extinction = float(kwargs.get("extinction", 0.05))
        radiance = photon_map.volumetric_radiance(
            o, d, max_dist=float(max_dist), step=step, radius=radius,
            scattering=scattering, extinction=extinction)
        return TraceResult(radiance, extra={
            "max_dist": float(max_dist),
            "step": step,
            "radius": radius,
            "scattering": scattering,
            "extinction": extinction,
            "photon_count": len(photon_map),
            "method": "volumetric_photon_mapping",
            "volumetric": True,
        })

    # -- 光学工具（静态方法） ----------------------------------------------------

    @staticmethod
    def reflect(direction, normal):
        return reflect(direction, normal)

    @staticmethod
    def refract(direction, normal, n1=1.0, n2=1.5):
        return refract(direction, normal, n1, n2)

    @staticmethod
    def fresnel(cos_theta, n1=1.0, n2=1.5):
        return fresnel(cos_theta, n1, n2)


# ---------------------------------------------------------------------------
# 焦散辅助
# ---------------------------------------------------------------------------

def _target_sphere(prims, clamp=100.0):
    """场景包围球（用于光子发射定向）。"""
    if not prims:
        return (0.0, 0.0, 0.0), 1.0
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for prim in prims:
        b_lo, b_hi = prim.bounds()
        for i in range(3):
            lo[i] = min(lo[i], max(-clamp, b_lo[i]))
            hi[i] = max(hi[i], min(clamp, b_hi[i]))
    center = tuple((lo[i] + hi[i]) * 0.5 for i in range(3))
    radius = max(_length((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])) * 0.5, 1e-3)
    return center, radius


def _emit_direction(rng, light_pos, center, radius):
    """从光源向目标包围球的圆锥内均匀采样方向。"""
    to = _sub(center, light_pos)
    dist = _length(to)
    if dist <= radius:
        return _normalize(_random_in_unit_sphere(rng))
    sin_max = min(1.0, radius / dist)
    cos_max = math.sqrt(max(0.0, 1.0 - sin_max * sin_max))
    cos_t = 1.0 - rng.random() * (1.0 - cos_max)
    sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
    phi = 2.0 * math.pi * rng.random()
    forward = _normalize(to)
    if abs(forward[1]) < 0.999:
        up = (0.0, 1.0, 0.0)
    else:
        up = (1.0, 0.0, 0.0)
    tangent = _normalize(_cross(up, forward))
    bitangent = _cross(forward, tangent)
    x = sin_t * math.cos(phi)
    y = sin_t * math.sin(phi)
    return _normalize((
        tangent[0] * x + bitangent[0] * y + forward[0] * cos_t,
        tangent[1] * x + bitangent[1] * y + forward[1] * cos_t,
        tangent[2] * x + bitangent[2] * y + forward[2] * cos_t,
    ))


__all__ = [
    "RayTracing",
    "Config",
    "Material",
    "Light",
    "Sphere",
    "Plane",
    "Triangle",
    "BVH",
    "PhotonMap",
    "TraceResult",
    "reflect",
    "refract",
    "fresnel",
    "fresnel_schlick",
]
