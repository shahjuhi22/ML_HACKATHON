import numpy as np
from typing import Tuple, Union

# WGS-84 reference constants
R_EARTH = 6378137.0  # Equatorial radius in metres

def project(
    lon: Union[float, np.ndarray], 
    lat: Union[float, np.ndarray], 
    lat0: float
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Project longitude and latitude coordinates into local metric Cartesian coordinates (metres)
    using local equirectangular projection centered on lat0.
    Accurate to within ~0.1% at city scale (50 km).
    """
    rad_lat0 = np.radians(lat0)
    # 1 degree longitude in metres = cos(lat0) * 111320.0
    x = lon * np.cos(rad_lat0) * 111320.0
    # 1 degree latitude in metres = 110574.0 (for mid-latitudes ~50 deg)
    y = lat * 110574.0
    return x, y

def inverse_project(
    x: Union[float, np.ndarray], 
    y: Union[float, np.ndarray], 
    lat0: float
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Convert local metric Cartesian coordinates back to longitude and latitude."""
    rad_lat0 = np.radians(lat0)
    lon = x / (np.cos(rad_lat0) * 111320.0)
    lat = y / 110574.0
    return lon, lat

def seg_dist(
    px: np.ndarray, py: np.ndarray, 
    ax: np.ndarray, ay: np.ndarray, 
    bx: np.ndarray, by: np.ndarray
) -> np.ndarray:
    """
    Perpendicular distance from point P(px, py) to line segment AB(ax, ay) -> (bx, by).
    Clamped to segment endpoints.
    """
    abx = bx - ax
    aby = by - ay
    denom = abx ** 2 + aby ** 2 + 1e-12
    t = np.clip(((px - ax) * abx + (py - ay) * aby) / denom, 0.0, 1.0)
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    return np.hypot(px - proj_x, py - proj_y)

def haversine_distance(
    lon1: np.ndarray, lat1: np.ndarray, 
    lon2: np.ndarray, lat2: np.ndarray
) -> np.ndarray:
    """Exact great-circle distance between coordinates in metres."""
    r = R_EARTH
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return r * c
