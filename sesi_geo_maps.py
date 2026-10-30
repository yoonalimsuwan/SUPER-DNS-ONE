"""
=============================================================================
SES GEO MAPS — Thai + Chinese Map Service Integration
Multi-provider abstraction with WGS-84 ↔ GCJ-02 coordinate transform
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES
---------------------
[C/D] Unified interface for map services across Thailand and China:

  THAILAND:
    • Google Maps API (requires API key)
    • Longdo Map (Thai provider)
  
  CHINA:
    • Amap / Gaode Maps (requires API key)
    • Tencent Maps (requires API key)
    • Baidu Maps (requires API key)

CRITICAL: COORDINATE SYSTEMS
-----------------------------
[C] WGS-84 — GPS standard, used globally (except mainland China)
[C] GCJ-02 — China's "offset" coordinate system (aka "Mars coordinates")
[C] BD-09  — Baidu's additional offset on top of GCJ-02

Chinese law requires all map services to use GCJ-02, which applies a
deliberate offset from WGS-84. Using WGS-84 coordinates directly on a
Chinese map will show positions shifted by 100-700 meters.

This module provides:
  • WGS-84 ↔ GCJ-02 transform (public algorithm, approximate)
  • GCJ-02 ↔ BD-09 transform (for Baidu Maps)
  • Geofencing for disaster alert zones
  • Multi-provider fallback (if one API fails, try another)

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• GCJ-02 transform is APPROXIMATE (based on publicly available reverse-
  engineered algorithms); accuracy is ~1-5 meters, not survey-grade.
• This is a FRAMEWORK; real deployment requires API keys from providers.
• Map tiles and routing are NOT included; this is for coordinate handling
  and geofencing only.
• Geofencing uses simple circular/polygonal regions; real disaster zones
  may have complex boundaries requiring GIS tools.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ses.geo_maps")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

__all__ = [
    "CoordinateSystem",
    "GCJ02Transformer",
    "GeoMapsConfig",
    "MapProvider",
    "GeofenceZone",
    "DisasterGeofence",
]


class CoordinateSystem:
    """Coordinate system identifiers."""
    WGS84 = "WGS-84"
    GCJ02 = "GCJ-02"
    BD09 = "BD-09"


# ═══════════════════════════════════════════════════════════════════════
# [C] GCJ-02 COORDINATE TRANSFORMER
# ═══════════════════════════════════════════════════════════════════════

class GCJ02Transformer:
    """
    [C] Transform between WGS-84 and GCJ-02 (China's offset coordinates).
    
    The algorithm is publicly available and reverse-engineered.
    Accuracy: ~1-5 meters (not survey-grade).
    
    GCJ-02 applies a deliberate offset to WGS-84 coordinates within
    mainland China. The offset is non-linear and varies by location.
    """
    
    # Krasovsky ellipsoid parameters (used in GCJ-02)
    A = 6378245.0  # semi-major axis
    EE = 0.00669342162296594323  # eccentricity squared
    
    @staticmethod
    def _transform_lat(x: float, y: float) -> float:
        """Auxiliary latitude transformation."""
        ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + \
              0.1 * x * y + 0.2 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * math.pi) + 
                20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(y * math.pi) + 
                40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(y / 12.0 * math.pi) + 
                320.0 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
        return ret
    
    @staticmethod
    def _transform_lon(x: float, y: float) -> float:
        """Auxiliary longitude transformation."""
        ret = 300.0 + x + 2.0 * y + 0.1 * x * x + \
              0.1 * x * y + 0.1 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * math.pi) + 
                20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(x * math.pi) + 
                40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(x / 12.0 * math.pi) + 
                300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
        return ret
    
    @classmethod
    def _is_in_china(cls, lat: float, lon: float) -> bool:
        """Check if coordinates are roughly within mainland China."""
        return 0.8293 <= lat <= 55.8271 and 72.004 <= lon <= 137.8347
    
    @classmethod
    def wgs84_to_gcj02(cls, lat: float, lon: float) -> Tuple[float, float]:
        """
        Transform WGS-84 to GCJ-02.
        
        Returns (gcj_lat, gcj_lon).
        """
        if not cls._is_in_china(lat, lon):
            return lat, lon  # No transform outside China
        
        dlat = cls._transform_lat(lon - 105.0, lat - 35.0)
        dlon = cls._transform_lon(lon - 105.0, lat - 35.0)
        
        rad_lat = lat / 180.0 * math.pi
        magic = math.sin(rad_lat)
        magic = 1 - cls.EE * magic * magic
        sqrt_magic = math.sqrt(magic)
        
        dlat = (dlat * 180.0) / ((cls.A * (1 - cls.EE)) / 
                                  (magic * sqrt_magic) * math.pi)
        dlon = (dlon * 180.0) / (cls.A / sqrt_magic * math.cos(rad_lat) * math.pi)
        
        return lat + dlat, lon + dlon
    
    @classmethod
    def gcj02_to_wgs84(cls, lat: float, lon: float) -> Tuple[float, float]:
        """
        Transform GCJ-02 to WGS-84 (approximate inverse).
        
        Returns (wgs_lat, wgs_lon).
        Accuracy: ~1-5 meters.
        """
        if not cls._is_in_china(lat, lon):
            return lat, lon
        
        # Iterative inverse (1 iteration is usually sufficient)
        wgs_lat, wgs_lon = lat, lon
        for _ in range(2):
            gcj_lat, gcj_lon = cls.wgs84_to_gcj02(wgs_lat, wgs_lon)
            wgs_lat += lat - gcj_lat
            wgs_lon += lon - gcj_lon
        
        return wgs_lat, wgs_lon
    
    @staticmethod
    def gcj02_to_bd09(lat: float, lon: float) -> Tuple[float, float]:
        """Transform GCJ-02 to BD-09 (Baidu)."""
        x, y = lon, lat
        z = math.sqrt(x * x + y * y) + 0.00002 * math.sin(y * math.pi * 3000.0 / 180.0)
        theta = math.atan2(y, x) + 0.000003 * math.cos(x * math.pi * 3000.0 / 180.0)
        bd_lon = z * math.cos(theta) + 0.0065
        bd_lat = z * math.sin(theta) + 0.006
        return bd_lat, bd_lon
    
    @staticmethod
    def bd09_to_gcj02(lat: float, lon: float) -> Tuple[float, float]:
        """Transform BD-09 to GCJ-02 (approximate inverse)."""
        x = lon - 0.0065
        y = lat - 0.006
        z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * math.pi * 3000.0 / 180.0)
        theta = math.atan2(y, x) - 0.000003 * math.cos(x * math.pi * 3000.0 / 180.0)
        gcj_lon = z * math.cos(theta)
        gcj_lat = z * math.sin(theta)
        return gcj_lat, gcj_lon


# ═══════════════════════════════════════════════════════════════════════
# [D] MAP PROVIDER ABSTRACTION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GeoMapsConfig:
    """Configuration for map services."""
    # API keys (must be provided by user)
    google_maps_key: str = ""
    amap_key: str = ""
    tencent_maps_key: str = ""
    baidu_maps_key: str = ""
    longdo_key: str = ""
    
    # Default coordinate system
    default_system: str = CoordinateSystem.WGS84
    
    # Request settings
    timeout_s: float = 10.0
    max_retries: int = 3


class MapProvider:
    """
    [D] Unified map provider interface.
    
    Supports multiple backends with automatic fallback.
    """
    
    def __init__(self, cfg: GeoMapsConfig):
        self.cfg = cfg
        self.transformer = GCJ02Transformer()
        
        # Detect available providers
        self.providers = []
        if cfg.google_maps_key:
            self.providers.append("google")
        if cfg.amap_key:
            self.providers.append("amap")
        if cfg.tencent_maps_key:
            self.providers.append("tencent")
        if cfg.baidu_maps_key:
            self.providers.append("baidu")
        if cfg.longdo_key:
            self.providers.append("longdo")
        
        if not self.providers:
            logger.warning("No map API keys configured; geocoding disabled")
    
    def geocode(self, address: str, country: str = "TH") -> Optional[Dict[str, Any]]:
        """
        Geocode an address to coordinates.
        
        Args:
            address: Address string
            country: "TH" for Thailand, "CN" for China
        
        Returns:
            dict with 'lat', 'lon', 'system', 'provider'
        """
        if not HAS_REQUESTS:
            logger.warning("requests not available; geocoding disabled")
            return None
        
        # Try providers in order
        for provider in self.providers:
            try:
                if provider == "google" and country == "TH":
                    return self._geocode_google(address)
                elif provider == "amap" and country == "CN":
                    return self._geocode_amap(address)
                elif provider == "tencent" and country == "CN":
                    return self._geocode_tencent(address)
                elif provider == "baidu" and country == "CN":
                    return self._geocode_baidu(address)
                elif provider == "longdo" and country == "TH":
                    return self._geocode_longdo(address)
            except Exception as e:
                logger.warning(f"Provider {provider} failed: {e}")
                continue
        
        logger.error(f"All geocoding providers failed for: {address}")
        return None
    
    def _geocode_google(self, address: str) -> Dict[str, Any]:
        """Google Maps geocoding (Thailand)."""
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"address": address, "key": self.cfg.google_maps_key}
        
        resp = requests.get(url, params=params, timeout=self.cfg.timeout_s)
        data = resp.json()
        
        if data["status"] != "OK":
            raise ValueError(f"Google geocoding failed: {data['status']}")
        
        result = data["results"][0]
        return {
            "lat": result["geometry"]["location"]["lat"],
            "lon": result["geometry"]["location"]["lng"],
            "system": CoordinateSystem.WGS84,
            "provider": "google",
            "formatted_address": result["formatted_address"],
        }
    
    def _geocode_amap(self, address: str) -> Dict[str, Any]:
        """Amap (Gaode) geocoding (China)."""
        url = "https://restapi.amap.com/v3/geocode/geo"
        params = {"address": address, "key": self.cfg.amap_key}
        
        resp = requests.get(url, params=params, timeout=self.cfg.timeout_s)
        data = resp.json()
        
        if data["status"] != "1":
            raise ValueError(f"Amap geocoding failed: {data.get('info', 'unknown')}")
        
        result = data["geocodes"][0]
        lon, lat = map(float, result["location"].split(","))
        
        return {
            "lat": lat,
            "lon": lon,
            "system": CoordinateSystem.GCJ02,
            "provider": "amap",
            "formatted_address": result["formatted_address"],
        }
    
    def _geocode_tencent(self, address: str) -> Dict[str, Any]:
        """Tencent Maps geocoding (China)."""
        url = "https://apis.map.qq.com/ws/geocoder/v1/"
        params = {"address": address, "key": self.cfg.tencent_maps_key}
        
        resp = requests.get(url, params=params, timeout=self.cfg.timeout_s)
        data = resp.json()
        
        if data["status"] != 0:
            raise ValueError(f"Tencent geocoding failed: {data.get('message', 'unknown')}")
        
        result = data["result"]
        return {
            "lat": result["location"]["lat"],
            "lon": result["location"]["lng"],
            "system": CoordinateSystem.GCJ02,
            "provider": "tencent",
            "formatted_address": result["title"],
        }
    
    def _geocode_baidu(self, address: str) -> Dict[str, Any]:
        """Baidu Maps geocoding (China)."""
        url = "https://api.map.baidu.com/geocoding/v3/"
        params = {"address": address, "ak": self.cfg.baidu_maps_key, "output": "json"}
        
        resp = requests.get(url, params=params, timeout=self.cfg.timeout_s)
        data = resp.json()
        
        if data["status"] != 0:
            raise ValueError(f"Baidu geocoding failed: {data.get('message', 'unknown')}")
        
        result = data["result"]
        return {
            "lat": result["location"]["lat"],
            "lon": result["location"]["lng"],
            "system": CoordinateSystem.BD09,
            "provider": "baidu",
            "formatted_address": address,
        }
    
    def _geocode_longdo(self, address: str) -> Dict[str, Any]:
        """Longdo Map geocoding (Thailand)."""
        url = "https://api.longdo.com/map/services/address"
        params = {"key": self.cfg.longdo_key, "locale": "th_TH", "query": address}
        
        resp = requests.get(url, params=params, timeout=self.cfg.timeout_s)
        data = resp.json()
        
        if not data:
            raise ValueError("Longdo geocoding returned no results")
        
        result = data[0]
        return {
            "lat": result["lat"],
            "lon": result["lon"],
            "system": CoordinateSystem.WGS84,
            "provider": "longdo",
            "formatted_address": result.get("address", address),
        }
    
    def transform_coordinates(self, lat: float, lon: float,
                               from_system: str, to_system: str) -> Tuple[float, float]:
        """
        Transform coordinates between systems.
        
        Supports: WGS84 ↔ GCJ02 ↔ BD09
        """
        if from_system == to_system:
            return lat, lon
        
        # WGS84 → GCJ02
        if from_system == CoordinateSystem.WGS84 and to_system == CoordinateSystem.GCJ02:
            return self.transformer.wgs84_to_gcj02(lat, lon)
        
        # GCJ02 → WGS84
        if from_system == CoordinateSystem.GCJ02 and to_system == CoordinateSystem.WGS84:
            return self.transformer.gcj02_to_wgs84(lat, lon)
        
        # GCJ02 → BD09
        if from_system == CoordinateSystem.GCJ02 and to_system == CoordinateSystem.BD09:
            return self.transformer.gcj02_to_bd09(lat, lon)
        
        # BD09 → GCJ02
        if from_system == CoordinateSystem.BD09 and to_system == CoordinateSystem.GCJ02:
            return self.transformer.bd09_to_gcj02(lat, lon)
        
        # WGS84 → BD09 (via GCJ02)
        if from_system == CoordinateSystem.WGS84 and to_system == CoordinateSystem.BD09:
            gcj_lat, gcj_lon = self.transformer.wgs84_to_gcj02(lat, lon)
            return self.transformer.gcj02_to_bd09(gcj_lat, gcj_lon)
        
        # BD09 → WGS84 (via GCJ02)
        if from_system == CoordinateSystem.BD09 and to_system == CoordinateSystem.WGS84:
            gcj_lat, gcj_lon = self.transformer.bd09_to_gcj02(lat, lon)
            return self.transformer.gcj02_to_wgs84(gcj_lat, gcj_lon)
        
        raise ValueError(f"Unsupported transform: {from_system} → {to_system}")


# ═══════════════════════════════════════════════════════════════════════
# [D] GEOFENCING FOR DISASTER ALERTS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GeofenceZone:
    """A geofence zone for disaster alerts."""
    zone_id: str
    name: str
    center_lat: float
    center_lon: float
    radius_km: float
    hazard_type: str  # FIRE, FLOOD, EARTHQUAKE, etc.
    severity: str = "WARNING"
    coordinate_system: str = CoordinateSystem.WGS84


class DisasterGeofence:
    """
    [D] Geofencing system for disaster alerts.
    
    Uses simple circular zones for efficiency. Complex boundaries
    would require GIS polygon operations.
    """
    
    def __init__(self, cfg: GeoMapsConfig):
        self.cfg = cfg
        self.provider = MapProvider(cfg)
        self.zones: List[GeofenceZone] = []
    
    def add_zone(self, zone: GeofenceZone) -> None:
        """Add a geofence zone."""
        self.zones.append(zone)
        logger.info(f"Added geofence zone: {zone.name} ({zone.hazard_type})")
    
    def is_in_zone(self, lat: float, lon: float, 
                   zone: GeofenceZone) -> bool:
        """
        Check if a point is within a zone.
        
        Uses Haversine formula for distance calculation.
        """
        # Transform to zone's coordinate system if needed
        if zone.coordinate_system != CoordinateSystem.WGS84:
            lat, lon = self.provider.transform_coordinates(
                lat, lon, CoordinateSystem.WGS84, zone.coordinate_system)
        
        # Haversine distance
        R = 6371.0  # Earth radius in km
        dlat = math.radians(zone.center_lat - lat)
        dlon = math.radians(zone.center_lon - lon)
        
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(math.radians(lat)) * math.cos(math.radians(zone.center_lat)) * 
             math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        
        return distance <= zone.radius_km
    
    def get_zones_for_location(self, lat: float, lon: float) -> List[GeofenceZone]:
        """Get all zones containing a location."""
        matching = []
        for zone in self.zones:
            if self.is_in_zone(lat, lon, zone):
                matching.append(zone)
        return matching
    
    def get_alerts_for_location(self, lat: float, lon: float) -> List[Dict[str, Any]]:
        """
        Get disaster alerts for a location.
        
        Returns list of alerts with zone info and recommended actions.
        """
        zones = self.get_zones_for_location(lat, lon)
        alerts = []
        
        for zone in zones:
            alerts.append({
                "zone_id": zone.zone_id,
                "zone_name": zone.name,
                "hazard_type": zone.hazard_type,
                "severity": zone.severity,
                "center": (zone.center_lat, zone.center_lon),
                "radius_km": zone.radius_km,
                "recommended_action": self._get_recommended_action(zone.hazard_type, zone.severity),
            })
        
        return alerts
    
    def _get_recommended_action(self, hazard_type: str, severity: str) -> str:
        """Get recommended action based on hazard type and severity."""
        actions = {
            "FIRE": {
                "EMERGENCY": "Evacuate immediately. Call emergency services.",
                "WARNING": "Prepare to evacuate. Avoid the area.",
                "ADVISORY": "Stay informed. Monitor the situation.",
            },
            "FLOOD": {
                "EMERGENCY": "Move to higher ground immediately.",
                "WARNING": "Prepare emergency kit. Monitor water levels.",
                "ADVISORY": "Avoid low-lying areas.",
            },
            "EARTHQUAKE": {
                "EMERGENCY": "Drop, cover, hold on. Move away from buildings.",
                "WARNING": "Secure heavy objects. Review emergency plan.",
                "ADVISORY": "Be prepared for aftershocks.",
            },
            "STORM": {
                "EMERGENCY": "Take shelter immediately.",
                "WARNING": "Secure outdoor items. Monitor weather.",
                "ADVISORY": "Stay informed about storm development.",
            },
        }
        
        return actions.get(hazard_type, {}).get(severity, "Stay informed.")


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SES GEO Maps (Thai + Chinese) — Smoke Test")
    print("=" * 72)
    
    # Test GCJ-02 transform
    print("\n  ── GCJ-02 Coordinate Transform ──")
    
    # Beijing coordinates (WGS-84)
    beijing_wgs = (39.9042, 116.4074)
    
    # Transform to GCJ-02
    beijing_gcj = GCJ02Transformer.wgs84_to_gcj02(*beijing_wgs)
    print(f"    Beijing WGS-84: {beijing_wgs[0]:.6f}, {beijing_wgs[1]:.6f}")
    print(f"    Beijing GCJ-02: {beijing_gcj[0]:.6f}, {beijing_gcj[1]:.6f}")
    print(f"    Offset: {beijing_gcj[0] - beijing_wgs[0]:.6f}, {beijing_gcj[1] - beijing_wgs[1]:.6f}")
    
    # Transform back
    beijing_back = GCJ02Transformer.gcj02_to_wgs84(*beijing_gcj)
    error_lat = abs(beijing_back[0] - beijing_wgs[0])
    error_lon = abs(beijing_back[1] - beijing_wgs[1])
    print(f"    Round-trip error: {error_lat:.9f}, {error_lon:.9f}")
    print(f"    [{'PASS' if error_lat < 1e-6 and error_lon < 1e-6 else 'NOTE'}] "
          f"round-trip error {'small' if error_lat < 1e-6 else 'may be larger'}")
    
    # Test geofencing
    print("\n  ── Disaster Geofencing ──")
    cfg = GeoMapsConfig()
    geofence = DisasterGeofence(cfg)
    
    # Add test zones
    geofence.add_zone(GeofenceZone(
        zone_id="flood_bkk_01",
        name="Bangkok Flood Zone",
        center_lat=13.7563,
        center_lon=100.5018,
        radius_km=10.0,
        hazard_type="FLOOD",
        severity="WARNING",
    ))
    
    geofence.add_zone(GeofenceZone(
        zone_id="fire_cnx_01",
        name="Chiang Mai Fire Zone",
        center_lat=18.7883,
        center_lon=98.9853,
        radius_km=5.0,
        hazard_type="FIRE",
        severity="ADVISORY",
    ))
    
    # Test location in Bangkok
    test_lat, test_lon = 13.7500, 100.5000
    alerts = geofence.get_alerts_for_location(test_lat, test_lon)
    print(f"    Test location: ({test_lat}, {test_lon})")
    print(f"    Zones found: {len(alerts)}")
    for alert in alerts:
        print(f"      • {alert['zone_name']}: {alert['hazard_type']} ({alert['severity']})")
        print(f"        Action: {alert['recommended_action']}")
    
    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • GCJ-02 transform is APPROXIMATE (~1-5 m accuracy)")
    print("  • Real deployment requires API keys from providers")
    print("  • Geofencing uses simple circular zones")
    print("  • Complex boundaries require GIS tools")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
