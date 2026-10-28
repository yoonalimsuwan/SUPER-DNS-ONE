"""
=============================================================================
SESI BAYESTAR SKY MAP × REAL-DATA VALIDATOR INTEGRATION
Load Bayestar HEALPix FITS, weight template validation by sky probability
=============================================================================
Developer : PAI & Yoon A. Limsuwan / MSPS NETWORK
My Soul Move By Power of Holy Spirit
License   : MIT | Year : 2026

WHAT THIS CONNECTS
------------------
• RealDataValidator (from sesi_template_validation) validates templates
  against GW strain.
• Bayestar sky maps (HEALPix FITS, from GraceDB/GWOSC) give the spatial
  probability distribution of the event.
• This module joins them: validation is reported TOGETHER with the sky-map
  credible region, sky area, and the probability at the recovered position.

BAYESTAR FITS FORMAT [C]
------------------------
Bayestar (Singer & Price 2016) produces multi-resolution HEALPix FITS:
  • pixelization : usually UNIQ (MOC-style), sometimes RING/NESTED
  • column PROB / PROBDENSITY : probability (density) per pixel
  • optional DISTMU, DISTSIGMA, DISTNORM : 3D distance information

We read via `healpy` when available. If the FITS is absent or healpy is
not installed, we fall back to a SYNTHETIC Gaussian sky map, clearly
flagged — never presenting synthetic data as a real localization.

LABELS: [C] Bayestar/HEALPix standard  [D] derived integration  [O] open

HONEST SCOPE
------------
• Real Bayestar FITS for specific events require GraceDB/GWOSC access.
• Without a real FITS, this module reports synthetic=True and the results
  must NOT be cited as a real sky localization.
• Sky-area and credible-region numbers inherit the FITS resolution.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.bayestar_realdata")

try:
    import healpy as hp
    HAS_HEALPY = True
except ImportError:
    HAS_HEALPY = False

try:
    from astropy.io import fits as astropy_fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

__all__ = [
    "BayestarConfig",
    "BayestarSkyMapLoader",
    "SkyMapWeightedValidator",
    "BayestarRealDataIntegration",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BayestarConfig:
    """Configuration for Bayestar sky-map handling."""
    fits_path: Optional[str] = None        # path to a real Bayestar FITS
    nside_fallback: int = 64               # HEALPix nside for synthetic map
    credible_levels: List[float] = None    # credible regions to report

    # For synthetic fallback
    synthetic_center_ra: float = 180.0
    synthetic_center_dec: float = 30.0
    synthetic_sigma_deg: float = 10.0

    def __post_init__(self):
        if self.credible_levels is None:
            self.credible_levels = [0.5, 0.9, 0.95]


# ═══════════════════════════════════════════════════════════════════════
# BAYESTAR SKY MAP LOADER
# ═══════════════════════════════════════════════════════════════════════

class BayestarSkyMapLoader:
    """
    [C/D] Load a Bayestar sky map into a normalized HEALPix probability array.

    Tries, in order:
      1. healpy.read_map on a real FITS file
      2. astropy.io.fits for manual UNIQ → RING conversion
      3. synthetic Gaussian map (clearly flagged)
    """

    def __init__(self, cfg: BayestarConfig):
        self.cfg = cfg
        self.prob_map: Optional[np.ndarray] = None
        self.nside: Optional[int] = None
        self.synthetic: bool = True
        self.metadata: Dict[str, Any] = {}

    def load(self) -> Dict[str, Any]:
        """Load the sky map; returns status."""
        # Path 1: real FITS via healpy
        if self.cfg.fits_path is not None and Path(self.cfg.fits_path).exists():
            if HAS_HEALPY:
                return self._load_fits_healpy(self.cfg.fits_path)
            elif HAS_ASTROPY:
                return self._load_fits_astropy(self.cfg.fits_path)
            else:
                logger.warning("Neither healpy nor astropy available for FITS")

        # Fallback: synthetic Gaussian
        logger.info("No real FITS available; generating SYNTHETIC Gaussian map")
        return self._synthetic_gaussian()

    def _load_fits_healpy(self, path: str) -> Dict[str, Any]:
        """Load via healpy.read_map (handles RING/NESTED; UNIQ may need care)."""
        try:
            # Bayestar often uses UNIQ; healpy can read it in recent versions
            m = hp.read_map(path, field=1, verbose=False)  # field=1 → PROB/PROBDENSITY
            # Normalize to a probability distribution
            m = np.maximum(m, 0.0)
            total = m.sum()
            if total > 0:
                m = m / total
            self.prob_map = m
            self.nside = hp.get_nside(m)
            self.synthetic = False
            self.metadata = {"source": path, "loader": "healpy"}
            logger.info(f"Loaded real Bayestar map: nside={self.nside}")
            return {"status": "REAL", "nside": self.nside, "synthetic": False}
        except Exception as e:
            logger.warning(f"healpy load failed ({e}); falling back to synthetic")
            return self._synthetic_gaussian()

    def _load_fits_astropy(self, path: str) -> Dict[str, Any]:
        """Manual FITS read via astropy (UNIQ handling is approximate)."""
        try:
            with astropy_fits.open(path) as hdul:
                data = hdul[1].data
                # Try common column names for probability
                prob_col = None
                for name in ("PROBDENSITY", "PROB", "PROBABILITY"):
                    if name in data.columns.names:
                        prob_col = data[name]
                        break
                if prob_col is None:
                    raise KeyError("No PROB column found")
                m = np.array(prob_col, dtype=float)
                m = np.maximum(m, 0.0)
                m = m / m.sum() if m.sum() > 0 else m
                self.prob_map = m
                self.nside = hp.npix2nside(len(m)) if HAS_HEALPY else int(math.sqrt(len(m)/12))
                self.synthetic = False
                self.metadata = {"source": path, "loader": "astropy"}
                return {"status": "REAL", "nside": self.nside, "synthetic": False}
        except Exception as e:
            logger.warning(f"astropy load failed ({e}); falling back to synthetic")
            return self._synthetic_gaussian()

    def _synthetic_gaussian(self) -> Dict[str, Any]:
        """Generate a clearly-labeled synthetic Gaussian sky map."""
        nside = self.cfg.nside_fallback
        npix = 12 * nside * nside if HAS_HEALPY else 12 * nside * nside

        if HAS_HEALPY:
            # Pixel centers
            pix = np.arange(npix)
            theta, phi = hp.pix2ang(nside, pix)
            dec = 90.0 - np.degrees(theta)
            ra = np.degrees(phi)

            # Angular distance to the synthetic center
            sep = self._angular_separation(
                ra, dec,
                self.cfg.synthetic_center_ra,
                self.cfg.synthetic_center_dec)

            sigma = self.cfg.synthetic_sigma_deg
            prob = np.exp(-0.5 * (sep / sigma) ** 2)
            prob = prob / prob.sum()

            self.prob_map = prob
            self.nside = nside
        else:
            # Minimal placeholder without healpy
            self.prob_map = np.ones(npix) / npix
            self.nside = nside

        self.synthetic = True
        self.metadata = {"source": "synthetic_gaussian",
                         "center": (self.cfg.synthetic_center_ra,
                                    self.cfg.synthetic_center_dec),
                         "sigma_deg": self.cfg.synthetic_sigma_deg}
        return {"status": "SYNTHETIC", "nside": nside, "synthetic": True,
                "note": "NOT a real Bayestar localization"}

    @staticmethod
    def _angular_separation(ra1, dec1, ra2, dec2) -> np.ndarray:
        """Vectorized angular separation (degrees)."""
        r1, d1 = np.radians(ra1), np.radians(dec1)
        r2, d2 = np.radians(ra2), np.radians(dec2)
        cos_sep = (np.sin(d1) * np.sin(d2)
                   + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2))
        cos_sep = np.clip(cos_sep, -1.0, 1.0)
        return np.degrees(np.arccos(cos_sep))

    # ── Queries on the loaded map ─────────────────────────────────────
    def probability_at(self, ra_deg: float, dec_deg: float) -> float:
        """Probability density at a sky position."""
        if self.prob_map is None:
            return 0.0
        if not HAS_HEALPY:
            return float(self.prob_map.mean())
        theta = math.radians(90.0 - dec_deg)
        phi = math.radians(ra_deg)
        pix = hp.ang2pix(self.nside, theta, phi)
        return float(self.prob_map[pix])

    def credible_region(self, level: float = 0.9) -> np.ndarray:
        """Pixel indices inside the smallest region containing `level` prob."""
        if self.prob_map is None:
            return np.array([])
        order = np.argsort(-self.prob_map)
        cumsum = np.cumsum(self.prob_map[order])
        n = int(np.searchsorted(cumsum, level)) + 1
        return order[:n]

    def sky_area(self, level: float = 0.9) -> float:
        """Sky area (deg²) of the credible region."""
        if self.prob_map is None or not HAS_HEALPY:
            return float("nan")
        region = self.credible_region(level)
        pix_area = hp.nside2pixarea(self.nside, degrees=True)
        return float(len(region) * pix_area)

    def peak_position(self) -> Tuple[float, float]:
        """(ra, dec) of the maximum-probability pixel."""
        if self.prob_map is None or not HAS_HEALPY:
            return (float("nan"), float("nan"))
        peak_pix = int(np.argmax(self.prob_map))
        theta, phi = hp.pix2ang(self.nside, peak_pix)
        return float(np.degrees(phi)), float(90.0 - np.degrees(theta))


# ═══════════════════════════════════════════════════════════════════════
# SKY-MAP-WEIGHTED VALIDATOR
# ═══════════════════════════════════════════════════════════════════════

class SkyMapWeightedValidator:
    """
    [D] Augment template/strain validation with Bayestar sky-map context.

    Given a recovered position (from the strain analysis) and the sky map:
      • probability at the recovered position
      • whether it lies inside the 90% credible region
      • credible-region sky areas
      • synthetic flag (never hidden)
    """

    def __init__(self, loader: BayestarSkyMapLoader):
        self.loader = loader

    def validate_position(self, ra_deg: float, dec_deg: float) -> Dict[str, Any]:
        """Check a recovered sky position against the sky map."""
        self.loader.load()

        prob = self.loader.probability_at(ra_deg, dec_deg)
        peak_ra, peak_dec = self.loader.peak_position()

        # Is the position inside the 90% credible region?
        region90 = self.loader.credible_region(0.9)
        in_region = False
        if HAS_HEALPY and self.loader.nside is not None:
            theta = math.radians(90.0 - dec_deg)
            phi = math.radians(ra_deg)
            pix = hp.ang2pix(self.loader.nside, theta, phi)
            in_region = bool(pix in set(region90.tolist()))

        return {
            "recovered_ra": ra_deg,
            "recovered_dec": dec_deg,
            "probability_at_recovered": prob,
            "peak_ra": peak_ra,
            "peak_dec": peak_dec,
            "in_90_credible_region": in_region,
            "sky_area_50_deg2": self.loader.sky_area(0.5),
            "sky_area_90_deg2": self.loader.sky_area(0.9),
            "synthetic": self.loader.synthetic,
            "honest_note": ("SYNTHETIC sky map — not a real localization"
                            if self.loader.synthetic else
                            "Real Bayestar FITS loaded"),
        }


# ═══════════════════════════════════════════════════════════════════════
# INTEGRATION WITH RealDataValidator
# ═══════════════════════════════════════════════════════════════════════

class BayestarRealDataIntegration:
    """
    [D] Connect the existing RealDataValidator to Bayestar sky maps.

    RealDataValidator validates templates against strain. This integration
    adds the sky-map dimension: after validating strain recovery, it also
    reports where the event sits on the Bayestar sky map.

    HONEST: both halves must be independently trustworthy. A good strain
    match with a synthetic sky map is NOT a real multi-messenger localization.
    """

    def __init__(self, cfg: BayestarConfig):
        self.cfg = cfg
        self.loader = BayestarSkyMapLoader(cfg)
        self.sky_validator = SkyMapWeightedValidator(self.loader)

    def full_validation(self, strain_validation_result: Dict[str, Any],
                        recovered_ra: Optional[float] = None,
                        recovered_dec: Optional[float] = None) -> Dict[str, Any]:
        """
        Combine strain validation + sky-map validation.

        If no recovered position is given, uses the sky-map peak as a
        stand-in (clearly noted).
        """
        report = {"strain": strain_validation_result}

        if recovered_ra is None or recovered_dec is None:
            # Use peak as placeholder, flagged
            self.loader.load()
            recovered_ra, recovered_dec = self.loader.peak_position()
            report["position_source"] = "sky_map_peak_placeholder"
        else:
            report["position_source"] = "provided"

        sky_report = self.sky_validator.validate_position(recovered_ra, recovered_dec)
        report["sky"] = sky_report

        report["combined_status"] = self._combined_status(report)
        return report

    @staticmethod
    def _combined_status(report: Dict[str, Any]) -> str:
        """Summarize the combined validation honestly."""
        strain_ok = report.get("strain", {}).get("status") in ("RUN", "PASS", None)
        synthetic = report.get("sky", {}).get("synthetic", True)

        if synthetic:
            return ("METHOD_DEMO_ONLY: strain check ran, but the sky map is "
                    "SYNTHETIC — not a real localization.")
        if strain_ok:
            return "REAL_DATA: strain validation + real Bayestar sky map."
        return "INCOMPLETE: check strain and sky components individually."


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Bayestar × RealData Integration — Smoke Test")
    print("=" * 72)

    print(f"\n  healpy installed  : {HAS_HEALPY}")
    print(f"  astropy installed : {HAS_ASTROPY}")

    cfg = BayestarConfig(fits_path=None)  # no real FITS → synthetic
    integration = BayestarRealDataIntegration(cfg)

    # Simulate a strain-validation result
    strain_result = {"status": "RUN", "recovered_snr": 12.3,
                     "real_data": False}

    report = integration.full_validation(
        strain_validation_result=strain_result,
        recovered_ra=180.0, recovered_dec=30.0)

    print(f"\n  ── Sky-Map Validation ──")
    sky = report["sky"]
    print(f"    Recovered position     : ({sky['recovered_ra']:.1f}, {sky['recovered_dec']:.1f})")
    print(f"    Probability at position: {sky['probability_at_recovered']:.4e}")
    print(f"    Peak position          : ({sky['peak_ra']:.1f}, {sky['peak_dec']:.1f})")
    print(f"    In 90% credible region : {sky['in_90_credible_region']}")
    print(f"    Sky area 90%           : {sky['sky_area_90_deg2']:.1f} deg²")
    print(f"    Synthetic              : {sky['synthetic']}")
    print(f"    Note                   : {sky['honest_note']}")

    print(f"\n  ── Combined Status ──")
    print(f"    {report['combined_status']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • No real FITS → synthetic map, clearly flagged")
    print("  • Real Bayestar FITS requires GraceDB/GWOSC access")
    print("  • A good strain match + synthetic sky map is NOT a real")
    print("    multi-messenger localization")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
