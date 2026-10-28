"""
=============================================================================
SESI HEALPIX SKY PROBABILITY COINCIDENCE
Bayestar FITS + healpy for proper spatial coincidence
=============================================================================
[C] healpy (if installed)
[C] Bayestar FITS format
[D] Probability overlap calculations

HONEST SCOPE
------------
• Requires healpy for HEALPix operations.
• Bayestar FITS files from GraceDB (may require auth).
• Without real data, uses Gaussian approximation.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.healpix_coincidence")

try:
    import healpy as hp
    HAS_HEALPY = True
except ImportError:
    HAS_HEALPY = False

__all__ = [
    "SkyMapConfig",
    "BayestarSkyMap",
    "HEALPixOverlap",
    "SpatialCoincidenceCalculator",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SkyMapConfig:
    """Configuration for sky map handling."""
    # HEALPix resolution
    nside: int = 64  # HEALPix nside parameter
    
    # Credible regions
    credible_levels: List[float] = None
    
    # Coordinates
    coordinate_system: str = "icrs"  # or "galactic"
    
    def __post_init__(self):
        if self.credible_levels is None:
            self.credible_levels = [0.5, 0.9, 0.95]


# ═══════════════════════════════════════════════════════════════════════
# BAYESTAR SKY MAP
# ═══════════════════════════════════════════════════════════════════════

class BayestarSkyMap:
    """
    [C] Read Bayestar sky map FITS files.
    
    Bayestar FITS format:
      • UNIQ or RING pixelization
      • PROB: probability density per pixel
      • DISTMU, DISTSIGMA: distance parameters (optional)
    """

    def __init__(self, cfg: SkyMapConfig):
        self.cfg = cfg
        self.prob_map = None
        self.nside = cfg.nside

    def load_fits(self, path: str) -> bool:
        """Load Bayestar FITS file."""
        if not HAS_HEALPY:
            logger.warning("healpy not installed; cannot load FITS")
            return False
        
        try:
            # Read probability map
            # Bayestar FITS typically has PROB in column 1
            self.prob_map = hp.read_map(path, field=1, verbose=False)
            self.nside = hp.get_nside(self.prob_map)
            
            # Normalize
            self.prob_map = self.prob_map / np.sum(self.prob_map)
            
            logger.info(f"Loaded sky map: nside={self.nside}, "
                       f"npix={len(self.prob_map)}")
            return True
            
        except Exception as e:
            logger.warning(f"FITS load failed: {e}")
            return False

    def from_gaussian(self, ra_deg: float, dec_deg: float,
                      sigma_deg: float) -> None:
        """
        Create Gaussian sky map (fallback when no FITS available).
        
        HONEST: This is an APPROXIMATION, not a real Bayestar map.
        """
        if not HAS_HEALPY:
            logger.error("healpy required for sky maps")
            return
        
        npix = hp.nside2npix(self.nside)
        self.prob_map = np.zeros(npix)
        
        # Convert center to HEALPix pixel
        theta = np.radians(90.0 - dec_deg)  # colatitude
        phi = np.radians(ra_deg)
        
        # Get all pixel coordinates
        pix = np.arange(npix)
        theta_pix, phi_pix = hp.pix2ang(self.nside, pix)
        
        # Angular separation
        cos_sep = (np.cos(theta) * np.cos(theta_pix) +
                   np.sin(theta) * np.sin(theta_pix) * np.cos(phi - phi_pix))
        cos_sep = np.clip(cos_sep, -1, 1)
        sep = np.degrees(np.arccos(cos_sep))
        
        # Gaussian probability
        sigma_rad = np.radians(sigma_deg)
        self.prob_map = np.exp(-0.5 * (sep / sigma_deg) ** 2)
        self.prob_map = self.prob_map / np.sum(self.prob_map)
        
        logger.info(f"Created Gaussian sky map: center=({ra_deg:.1f}, {dec_deg:.1f}), "
                   f"sigma={sigma_deg:.1f}°")

    def credible_region(self, level: float = 0.9) -> np.ndarray:
        """
        Get pixels within credible region.
        
        Returns:
            Array of pixel indices
        """
        if self.prob_map is None:
            return np.array([])
        
        # Sort by probability (descending)
        sorted_idx = np.argsort(-self.prob_map)
        cumsum = np.cumsum(self.prob_map[sorted_idx])
        
        # Find pixels up to credible level
        n_pixels = np.searchsorted(cumsum, level)
        return sorted_idx[:n_pixels]

    def sky_area(self, level: float = 0.9) -> float:
        """Compute sky area of credible region (square degrees)."""
        pixels = self.credible_region(level)
        if len(pixels) == 0:
            return 0.0
        
        # Pixel area
        pix_area = hp.nside2pixarea(self.nside, degrees=True)
        return len(pixels) * pix_area

    def probability_at(self, ra_deg: float, dec_deg: float) -> float:
        """Get probability density at a specific position."""
        if self.prob_map is None:
            return 0.0
        
        theta = np.radians(90.0 - dec_deg)
        phi = np.radians(ra_deg)
        pix = hp.ang2pix(self.nside, theta, phi)
        
        return float(self.prob_map[pix])


# ═══════════════════════════════════════════════════════════════════════
# HEALPIX OVERLAP
# ═══════════════════════════════════════════════════════════════════════

class HEALPixOverlap:
    """
    [D] Compute overlap between sky probability maps.
    
    For two probability distributions P1, P2 on HEALPix grid:
      overlap = Σ_i P1(i) × P2(i)
    """

    def __init__(self, cfg: SkyMapConfig):
        self.cfg = cfg

    def overlap(self, map1: BayestarSkyMap, map2: BayestarSkyMap) -> float:
        """Compute probability overlap between two sky maps."""
        if map1.prob_map is None or map2.prob_map is None:
            return 0.0
        
        # Ensure same nside
        if map1.nside != map2.nside:
            logger.warning("Different nside; upgrading to common resolution")
            nside = max(map1.nside, map2.nside)
            p1 = hp.ud_grade(map1.prob_map, nside)
            p2 = hp.ud_grade(map2.prob_map, nside)
            p1 = p1 / np.sum(p1)
            p2 = p2 / np.sum(p2)
        else:
            p1, p2 = map1.prob_map, map2.prob_map
        
        # Compute overlap
        return float(np.sum(p1 * p2))

    def angular_separation(self, map1: BayestarSkyMap, 
                           map2: BayestarSkyMap) -> float:
        """Compute angular separation between map peaks."""
        if map1.prob_map is None or map2.prob_map is None:
            return 180.0
        
        # Find peaks
        peak1 = np.argmax(map1.prob_map)
        peak2 = np.argmax(map2.prob_map)
        
        theta1, phi1 = hp.pix2ang(map1.nside, peak1)
        theta2, phi2 = hp.pix2ang(map2.nside, peak2)
        
        # Angular separation
        cos_sep = (np.cos(theta1) * np.cos(theta2) +
                   np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2))
        cos_sep = np.clip(cos_sep, -1, 1)
        
        return float(np.degrees(np.arccos(cos_sep)))

    def coincidence_probability(self, map1: BayestarSkyMap,
                                map2: BayestarSkyMap,
                                background_rate: float = 1e-6) -> Dict[str, float]:
        """
        Compute coincidence probability.
        
        Uses overlap and angular separation to estimate
        probability of chance coincidence.
        """
        overlap = self.overlap(map1, map2)
        sep = self.angular_separation(map1, map2)
        
        # Simplified model: probability of chance coincidence
        # decreases with overlap and increases with sky area
        area1 = map1.sky_area(0.9)
        area2 = map2.sky_area(0.9)
        
        # Chance probability (simplified)
        sky_fraction = (area1 * area2) / (4 * math.pi * 180 / math.pi) ** 2
        p_chance = min(1.0, background_rate * sky_fraction * 1e6)
        
        return {
            "overlap": overlap,
            "angular_separation_deg": sep,
            "p_chance_coincidence": p_chance,
            "sky_area_1_deg2": area1,
            "sky_area_2_deg2": area2,
        }


# ═══════════════════════════════════════════════════════════════════════
# SPATIAL COINCIDENCE CALCULATOR
# ═══════════════════════════════════════════════════════════════════════

class SpatialCoincidenceCalculator:
    """
    [D] Full spatial coincidence calculation for multi-messenger events.
    
    Combines:
      • GW sky map (Bayestar)
      • Neutrino direction (IceCube)
      • EM counterpart position (optional)
    """

    def __init__(self, cfg: SkyMapConfig):
        self.cfg = cfg
        self.overlap_calc = HEALPixOverlap(cfg)

    def evaluate(self, gw_skymap: BayestarSkyMap,
                 neutrino_ra: float, neutrino_dec: float,
                 neutrino_error_deg: float) -> Dict[str, Any]:
        """
        Evaluate spatial coincidence between GW and neutrino.
        
        Args:
            gw_skymap: GW sky probability map
            neutrino_ra/dec: neutrino direction
            neutrino_error_deg: neutrino positional uncertainty
            
        Returns:
            dict with coincidence statistics
        """
        # Create neutrino sky map (Gaussian)
        nu_skymap = BayestarSkyMap(self.cfg)
        nu_skymap.from_gaussian(neutrino_ra, neutrino_dec, neutrino_error_deg)
        
        # Compute overlap
        overlap = self.overlap_calc.overlap(gw_skymap, nu_skymap)
        sep = self.overlap_calc.angular_separation(gw_skymap, nu_skymap)
        
        # Probability of GW at neutrino position
        p_gw_at_nu = gw_skymap.probability_at(neutrino_ra, neutrino_dec)
        
        # Credible region check
        credible_pixels = gw_skymap.credible_region(0.9)
        nu_pixel = hp.ang2pix(gw_skymap.nside,
                              np.radians(90 - neutrino_dec),
                              np.radians(neutrino_ra))
        in_credible = nu_pixel in credible_pixels
        
        return {
            "overlap": overlap,
            "angular_separation_deg": sep,
            "p_gw_at_neutrino": p_gw_at_nu,
            "neutrino_in_90_credible": bool(in_credible),
            "gw_sky_area_90_deg2": gw_skymap.sky_area(0.9),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI HEALPix Coincidence — Smoke Test")
    print("=" * 72)

    cfg = SkyMapConfig(nside=32)
    
    print(f"\n  healpy available: {HAS_HEALPY}")
    
    if not HAS_HEALPY:
        print("  [SKIP] healpy not installed")
        print("    Install with: pip install healpy")
        return
    
    # Create test sky maps
    print("\n  ── Creating Sky Maps ──")
    
    gw_map = BayestarSkyMap(cfg)
    gw_map.from_gaussian(ra_deg=180.0, dec_deg=45.0, sigma_deg=10.0)
    print(f"    GW map: nside={gw_map.nside}, npix={len(gw_map.prob_map)}")
    
    nu_map = BayestarSkyMap(cfg)
    nu_map.from_gaussian(ra_deg=185.0, dec_deg=48.0, sigma_deg=2.0)
    print(f"    Neutrino map: nside={nu_map.nside}")
    
    # Compute overlap
    print("\n  ── Overlap Calculation ──")
    overlap_calc = HEALPixOverlap(cfg)
    
    overlap = overlap_calc.overlap(gw_map, nu_map)
    sep = overlap_calc.angular_separation(gw_map, nu_map)
    
    print(f"    Overlap: {overlap:.6f}")
    print(f"    Angular separation: {sep:.2f}°")
    
    # Credible regions
    print("\n  ── Credible Regions ──")
    area_90 = gw_map.sky_area(0.9)
    print(f"    GW 90% credible area: {area_90:.1f} deg²")
    
    # Full coincidence
    print("\n  ── Spatial Coincidence ──")
    coinc = SpatialCoincidenceCalculator(cfg)
    result = coinc.evaluate(gw_map, 185.0, 48.0, 2.0)
    
    print(f"    Overlap: {result['overlap']:.6f}")
    print(f"    Separation: {result['angular_separation_deg']:.2f}°")
    print(f"    P(GW at neutrino): {result['p_gw_at_neutrino']:.6f}")
    print(f"    In 90% credible: {result['neutrino_in_90_credible']}")
    
    print("\n" + "=" * 72)
    print("  HONEST REMINDERS:")
    print("  • Real Bayestar FITS requires GraceDB access")
    print("  • Gaussian maps are APPROXIMATIONS")
    print("  • Coincidence probability is SIMPLIFIED")
    print("  • Full analysis requires proper background estimation")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
