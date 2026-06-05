from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import geopandas as gpd
import joblib
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.ops import transform

from .config import (
    ARTIFACT_DIR,
    CORE_TOWNS,
    EPSG_MODEL,
    EPSG_WEB,
    POI_LAYERS,
    ROAD_DISTANCE_FEATURES,
    SPATIAL_CACHE_PATH,
)

_TRANSFORMER_TO_MODEL = Transformer.from_crs(EPSG_WEB, EPSG_MODEL, always_xy=True)

def _require_layer(key: str):
    path, layer = POI_LAYERS[key]
    if not path.exists():
        raise FileNotFoundError(f"缺少必要 POI 圖資：{path.name}。請放入 POIs 資料夾後再啟動 app。")
    return path, layer

def _read_layer(key: str) -> gpd.GeoDataFrame:
    path, layer = _require_layer(key)
    gdf = gpd.read_file(path, layer=layer)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(EPSG_MODEL)
    if gdf.crs.to_epsg() != EPSG_MODEL:
        gdf = gdf.to_crs(EPSG_MODEL)
    gdf.geometry = gdf.geometry.map(lambda geom: transform(lambda x, y, z=None: (x, y), geom))
    return gdf

def _node_key(x: float, y: float) -> tuple[float, float]:
    return (round(float(x), 3), round(float(y), 3))

def _build_road_graph() -> tuple[nx.Graph, np.ndarray]:
    roads = _read_layer("roads").explode(index_parts=False).copy()
    roads = roads[roads.geometry.geom_type == "LineString"].copy()
    graph = nx.Graph()
    node_ids: dict[tuple[float, float], int] = {}

    def get_node(coord) -> int:
        key = _node_key(coord[0], coord[1])
        if key not in node_ids:
            node_ids[key] = len(node_ids)
        return node_ids[key]

    for geom in roads.geometry:
        coords = list(geom.coords)
        for start, end in zip(coords[:-1], coords[1:]):
            u = get_node(start)
            v = get_node(end)
            if u == v:
                continue
            weight = math.hypot(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
            if weight > 0:
                graph.add_edge(u, v, weight=weight)

    coords_array = np.zeros((len(node_ids), 2), dtype=float)
    for coord, node_id in node_ids.items():
        coords_array[node_id] = coord
    return graph, coords_array

def _snap_points_to_nodes(gdf: gpd.GeoDataFrame, node_tree: cKDTree) -> list[int]:
    # 🌟 救援行動：強制將所有形狀轉為「中心點 (Centroid)」，避免 Polygon 或 LineString 報錯
    centroids = gdf.geometry.centroid
    coords = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
    _, idx = node_tree.query(coords)
    return [int(i) for i in np.atleast_1d(idx)]

def _point_coords(gdf: gpd.GeoDataFrame) -> np.ndarray:
    # 🌟 救援行動：強制將所有形狀轉為「中心點 (Centroid)」
    centroids = gdf.geometry.centroid
    return np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()]).astype(float)

def build_and_save_spatial_cache(path=SPATIAL_CACHE_PATH) -> dict[str, Any]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    graph, node_coords = _build_road_graph()
    node_tree = cKDTree(node_coords)

    dist_maps: dict[str, dict[int, float]] = {}
    for key in ROAD_DISTANCE_FEATURES:
        poi_gdf = _read_layer(key)
        source_nodes = set(_snap_points_to_nodes(poi_gdf, node_tree))
        if not source_nodes:
            raise ValueError(f"{key} 圖層沒有可用點位，無法計算道路距離。")
        dist_maps[key] = nx.multi_source_dijkstra_path_length(graph, source_nodes, weight="weight")

    # 針對你的專屬模型：快取全聯、公園的座標 (用來算數量)，以及社宅 (用來判定距離)
    cache = {
        "graph": graph,
        "node_coords": node_coords,
        "dist_maps": dist_maps,
        "pxmart_coords": _point_coords(_read_layer("pxmart")),
        "park_coords": _point_coords(_read_layer("park")),
        "socialhouse_gdf": _read_layer("socialhouse"), # 直接存整個 GeoDataFrame 方便字串比對
        "towns": _read_layer("towns"),
    }
    joblib.dump(cache, path)
    return _hydrate_cache(cache)

def _hydrate_cache(cache: dict[str, Any]) -> dict[str, Any]:
    cache = dict(cache)
    cache["node_tree"] = cKDTree(cache["node_coords"])
    cache["pxmart_tree"] = cKDTree(cache["pxmart_coords"])
    cache["park_tree"] = cKDTree(cache["park_coords"])
    return cache

@lru_cache(maxsize=1)
def load_or_build_spatial_cache() -> dict[str, Any]:
    if SPATIAL_CACHE_PATH.exists():
        return _hydrate_cache(joblib.load(SPATIAL_CACHE_PATH))
    return build_and_save_spatial_cache(SPATIAL_CACHE_PATH)

def _count_within(tree: cKDTree, x: float, y: float, radius: float = 800.0) -> int:
    return int(len(tree.query_ball_point([x, y], r=radius)))

def compute_location_features(lon: float, lat: float, cache: dict[str, Any] | None = None) -> dict[str, Any]:
    cache = cache or load_or_build_spatial_cache()
    x, y = _TRANSFORMER_TO_MODEL.transform(float(lon), float(lat))
    point = Point(x, y)
    _, node_idx = cache["node_tree"].query([[x, y]])
    node = int(np.atleast_1d(node_idx)[0])

    features: dict[str, float] = {}
    details: dict[str, float | int] = {"x3826": float(x), "y3826": float(y)}
    
    # 1. 道路距離計算 (路徑規劃)
    for key, feature_name in ROAD_DISTANCE_FEATURES.items():
        dist = cache["dist_maps"][key].get(node)
        if dist is None:
            dist = 15000.0 # 找不到路徑時給一個預設極大值
        dist = max(float(dist), 1.0)
        features[feature_name] = float(np.log(dist))
        details[f"dist_{key}_m"] = round(dist, 2)

    # 2. 800公尺內設施數量計算
    px_count = _count_within(cache["pxmart_tree"], x, y, radius=800.0)
    park_count = _count_within(cache["park_tree"], x, y, radius=800.0)
    features["px_count_800m"] = float(px_count)
    features["park_count_800m"] = float(park_count)
    details["px_count_800m"] = px_count
    details["park_count_800m"] = park_count

    # 3. 社會住宅 2000 公尺涵蓋判定 (暴力字串比對法，超強容錯！)
    sh_dummies = [
        "in_梧棲區三民社會住宅_net_2000", "in_豐原安康一期_net_2000",
        "in_太平長億社會住宅_net_2000", "in_大里區光正一期社會住宅_net_2000",
        "in_烏日區高鐵社會住宅_net_2000", "in_南屯區精密機械科技創新園區社會住宅_net_2000",
        "in_東區恊園_net_2000"
    ]
    for dummy in sh_dummies:
        features[dummy] = 0.0

    sh_gdf = cache["socialhouse_gdf"]
    for idx, row in sh_gdf.iterrows():
        dist = point.distance(row.geometry)
        if dist <= 2000:
            row_str = str(row.to_dict())
            if "安康" in row_str: features["in_豐原安康一期_net_2000"] = 1.0
            elif "長億" in row_str: features["in_太平長億社會住宅_net_2000"] = 1.0
            elif "光正" in row_str: features["in_大里區光正一期社會住宅_net_2000"] = 1.0
            elif "三民" in row_str: features["in_梧棲區三民社會住宅_net_2000"] = 1.0
            elif "高鐵" in row_str: features["in_烏日區高鐵社會住宅_net_2000"] = 1.0
            elif "精密" in row_str or "精科" in row_str: features["in_南屯區精密機械科技創新園區社會住宅_net_2000"] = 1.0
            elif "恊園" in row_str or "協園" in row_str: features["in_東區恊園_net_2000"] = 1.0

    return {"features": features, "details": details}