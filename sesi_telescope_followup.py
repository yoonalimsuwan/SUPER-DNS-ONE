"""
=============================================================================
SESI TELESCOPE FOLLOW-UP SCHEDULER
EM counterpart observation planning from GW sky maps
=============================================================================
[C] Real telescope catalog (simplified)
[D] Visibility and priority calculations
[O] Real-time coordination (requires GCN/Treasure Map integration)

HONEST SCOPE
------------
• This is a SCHEDULER PROTOTYPE, not GLADE or real observatory software.
• Visibility calculations are simplified (no atmospheric extinction, etc.)
• Real follow-up requires coordination with observatories via GCN.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.telescope_followup")

__all__ = [
    "Telescope",
    "TelescopeCatalog",
    "VisibilityCalculator",
    "FollowupScheduler",
]


# ═══════════════════════════════════════════════════════════════════════
# TELESCOPE CATALOG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Telescope:
    """A ground-based telescope for follow-up."""
    name: str
    latitude_deg: float
    longitude_deg: float
    elevation_m: float
    aperture_m: float
    field_of_view_deg2: float  # square degrees
    limiting_mag: float        # limiting magnitude
    wavelength: str            # "optical", "IR", "radio", etc.
    operational: bool = True


class TelescopeCatalog:
    """
    [C] Simplified catalog of real telescopes.
    
    HONEST: This is a SUBSET for demonstration.
    Real catalogs include hundreds of facilities (GLADE, etc.).
    """

    def __init__(self):
        self.telescopes = self._build_catalog()

    def _build_catalog(self) -> List[Telescope]:
        return [
            # Optical wide-field
            Telescope("ZTF", 33.356, -116.865, 1712, 1.22, 47.0, 20.5, "optical"),
            Telescope("ATLAS-HKO", 20.708, -156.257, 3055, 0.5, 30.0, 19.0, "optical"),
            Telescope("ATLAS-MLO", 19.813, -155.456, 3397, 0.5, 30.0, 19.0, "optical"),
            Telescope("Pan-STARRS", 20.708, -156.257, 3055, 1.8, 7.0, 22.0, "optical"),
            Telescope("DECam", -30.169, -70.806, 2207, 4.0, 3.0, 24.0, "optical"),
            Telescope("VISTA", -24.616, -70.397, 2518, 4.1, 1.0, 21.0, "IR"),
            # Large telescopes
            Telescope("VLT", -24.627, -70.404, 2635, 8.2, 0.003, 26.0, "optical"),
            Telescope("Keck", 19.826, -155.478, 4145, 10.0, 0.001, 27.0, "optical"),
            Telescope("Subaru", 19.826, -155.478, 4160, 8.2, 0.2, 26.0, "optical"),
            # Southern hemisphere
            Telescope("SALT", -32.376, 20.811, 1798, 10.0, 0.1, 22.0, "optical"),
            Telescope("SOAR", -30.238, -70.734, 2738, 4.1, 0.02, 24.0, "optical"),
        ]

    def get_visible(self, ra_deg: float, dec_deg: float, 
                    time_mjd: float) -> List[Telescope]:
        """Get telescopes that can see a given sky position."""
        # Simplified: just return all operational telescopes
        # Real implementation would check altitude > 30°, sun below horizon, etc.
        return [t for t in self.telescopes if t.operational]


# ═══════════════════════════════════════════════════════════════════════
# VISIBILITY CALCULATOR
# ═══════════════════════════════════════════════════════════════════════

class VisibilityCalculator:
    """
    [D] Simplified visibility calculations.
    
    HONEST: Does not include:
      • Atmospheric extinction
      • Moon phase/brightness
      • Weather
      • Scheduling constraints
    """

    @staticmethod
    def altitude(tele_lat: float, tele_lon: float,
                 ra_deg: float, dec_deg: float,
                 time_mjd: float) -> float:
        """
        Compute altitude of a source from a telescope.
        
        Simplified: assumes source is visible if dec is within
        ±(90 - |lat|) of telescope latitude.
        """
        max_dec = 90.0 - abs(tele_lat)
        if abs(dec_deg - tele_lat) < max_dec + 30:  # 30° margin
            return 45.0  # nominal altitude
        return -90.0  # below horizon

    @staticmethod
    def airmass(altitude_deg: float) -> float:
        """Approximate airmass."""
        if altitude_deg <= 0:
            return np.inf
        return 1.0 / math.sin(math.radians(altitude_deg))


# ═══════════════════════════════════════════════════════════════════════
# FOLLOW-UP SCHEDULER
# ═══════════════════════════════════════════════════════════════════════

class FollowupScheduler:
    """
    [D] Schedule telescope follow-up based on GW sky map.
    
    Strategy:
      1. Rank sky regions by probability
      2. For each telescope, compute visible probability
      3. Prioritize by: P(visible) × FOV × sensitivity
    """

    def __init__(self, catalog: TelescopeCatalog):
        self.catalog = catalog
        self.vis = VisibilityCalculator()

    def schedule(self, sky_map: Dict[str, Any], 
                 event_time_mjd: float) -> List[Dict[str, Any]]:
        """
        Generate follow-up schedule.
        
        Args:
            sky_map: dict with 'ra', 'dec', 'prob' arrays (or HEALPix)
            event_time_mjd: event time in MJD
            
        Returns:
            List of observation plans
        """
        plans = []
        
        # Get source position (use peak probability)
        if "ra" in sky_map and "dec" in sky_map:
            peak_idx = np.argmax(sky_map.get("prob", [1.0]))
            ra_deg = float(sky_map["ra"][peak_idx])
            dec_deg = float(sky_map["dec"][peak_idx])
        else:
            # Fallback to center
            ra_deg, dec_deg = 180.0, 0.0
        
        # Get visible telescopes
        visible = self.catalog.get_visible(ra_deg, dec_deg, event_time_mjd)
        
        # Score each telescope
        for tele in visible:
            alt = self.vis.altitude(tele.latitude_deg, tele.longitude_deg,
                                    ra_deg, dec_deg, event_time_mjd)
            if alt < 30:
                continue
            
            # Priority score: FOV × sensitivity × visibility
            # (simplified; real scheduling is more complex)
            score = tele.field_of_view_deg2 * (25.0 - tele.limiting_mag)
            
            plans.append({
                "telescope": tele.name,
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "altitude_deg": alt,
                "airmass": self.vis.airmass(alt),
                "priority_score": score,
                "exposure_s": 300.0,  # default
                "filter": "r",  # default
            })
        
        # Sort by priority
        plans.sort(key=lambda x: x["priority_score"], reverse=True)
        
        return plans

    def report(self, plans: List[Dict[str, Any]]) -> str:
        """Generate human-readable schedule report."""
        lines = ["=" * 60, "TELESCOPE FOLLOW-UP SCHEDULE", "=" * 60]
        
        if not plans:
            lines.append("No telescopes available.")
            return "\n".join(lines)
        
        lines.append(f"\n{len(plans)} telescopes scheduled:")
        for i, p in enumerate(plans[:5]):
            lines.append(f"\n{i+1}. {p['telescope']}")
            lines.append(f"   Target: RA={p['ra_deg']:.2f}°, Dec={p['dec_deg']:.2f}°")
            lines.append(f"   Altitude: {p['altitude_deg']:.1f}° (airmass {p['airmass']:.2f})")
            lines.append(f"   Priority: {p['priority_score']:.2f}")
        
        lines.append("\n" + "=" * 60)
        lines.append("HONEST: Simplified scheduler prototype.")
        lines.append("Real follow-up requires GCN coordination.")
        lines.append("=" * 60)
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Telescope Follow-up Scheduler — Smoke Test")
    print("=" * 72)

    catalog = TelescopeCatalog()
    scheduler = FollowupScheduler(catalog)
    
    # Mock sky map
    sky_map = {
        "ra": np.array([180.0, 181.0, 179.0]),
        "dec": np.array([45.0, 45.5, 44.5]),
        "prob": np.array([0.5, 0.3, 0.2]),
    }
    
    print(f"\n  Telescope catalog: {len(catalog.telescopes)} facilities")
    
    # Generate schedule
    plans = scheduler.schedule(sky_map, event_time_mjd=60000.0)
    
    print(scheduler.report(plans))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
