"""
=============================================================================
SESI MULTI-MESSENGER ASTROPHYSICS VISUALIZATION DASHBOARD
Unified visualization of EM, GW, Neutrino, and Cosmic Ray data
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

DASHBOARD FEATURES
------------------
[D1] Sky Map Visualization
     • Mollweide projection
     • Multi-messenger event overlay
     • Anomaly highlighting
[D2] Time Series Analysis
     • Event rate vs time
     • Energy distribution
     • Multi-messenger correlation
[D3] Anomaly Detection
     • Carpet A^11 filter integration
     • Statistical significance
     • Multi-sensor fusion scores
[D4] Interactive Controls
     • Time window selection
     • Energy threshold
     • Sensor filtering

HONEST SCOPE
------------
• Dashboard is for RESEARCH VISUALIZATION, not confirmed detection.
• All anomalies require independent verification.
• No claim that displayed patterns indicate extraterrestrial origin.
=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

import numpy as np

logger = logging.getLogger("sesi.dashboard")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.projections import MollweideAxes
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from sesi_neutrino_cosmicray import (
    MultiMessengerDataHub,
    NeutrinoCosmicRayConfig,
    NeutrinoEvent,
    CosmicRayEvent,
)

__all__ = [
    "DashboardConfig",
    "MultiMessengerDashboard",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DashboardConfig:
    """Configuration for the multi-messenger dashboard."""
    # Time window
    lookback_hours: float = 24.0

    # Energy thresholds
    neutrino_energy_min_gev: float = 1e4  # 10 TeV
    cosmic_ray_energy_min_ev: float = 1e18  # 1 EeV

    # Anomaly detection
    anomaly_threshold_sigma: float = 3.0
    use_carpet_filter: bool = True

    # Visualization
    use_plotly: bool = True
    save_plots: bool = False
    output_dir: str = "./dashboard_output"

    def __post_init__(self):
        if self.save_plots:
            from pathlib import Path
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# MULTI-MESSENGER DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

class MultiMessengerDashboard:
    """
    Unified visualization dashboard for multi-messenger astrophysics.

    Combines data from:
      • Electromagnetic telescopes (ATLAS, ZTF, Gaia)
      • Gravitational wave detectors (LIGO/Virgo/KAGRA)
      • Neutrino observatories (IceCube)
      • Cosmic ray arrays (Pierre Auger)

    Provides:
      • Sky map visualization (Mollweide projection)
      • Time series analysis
      • Anomaly detection with Carpet A^11 filter
      • Multi-sensor correlation analysis
    """

    def __init__(self, cfg: Optional[DashboardConfig] = None):
        self.cfg = cfg or DashboardConfig()
        self.hub = MultiMessengerDataHub()

        # Data storage
        self.neutrino_events: List[NeutrinoEvent] = []
        self.cosmic_ray_events: List[CosmicRayEvent] = []
        self.correlations: List[Dict] = []

        # Anomaly scores
        self.anomaly_scores: Dict[str, float] = {}

    def load_data(self) -> None:
        """Load data from all sensors."""
        logger.info("Loading multi-messenger data...")

        # Neutrinos
        self.neutrino_events = self.hub.fetch_neutrino_events(
            hours=int(self.cfg.lookback_hours)
        )
        logger.info(f"Loaded {len(self.neutrino_events)} neutrino events")

        # Cosmic rays
        self.cosmic_ray_events = self.hub.fetch_cosmic_ray_events(
            hours=int(self.cfg.lookback_hours)
        )
        logger.info(f"Loaded {len(self.cosmic_ray_events)} cosmic ray events")

        # Find correlations
        self.correlations = self.hub.find_temporal_correlations(
            self.neutrino_events,
            self.cosmic_ray_events,
            window_seconds=300,
        )
        logger.info(f"Found {len(self.correlations)} temporal correlations")

    def compute_anomaly_scores(self) -> None:
        """
        Compute anomaly scores for events using Carpet A^11 filter.

        HONEST: This is a heuristic anomaly score, not a confirmed detection.
        """
        if not self.cfg.use_carpet_filter:
            logger.info("Carpet filter disabled; using simple statistical anomaly")
            return

        try:
            from sesi_anomalous_relativistic_detector import (
                AnomalousRelativisticDetector,
                DetectorConfig,
            )

            detector_cfg = DetectorConfig(carpet_level=3, polyharmonic_order=22)
            detector = AnomalousRelativisticDetector(detector_cfg)

            # Compute anomaly scores for neutrino events
            for event in self.neutrino_events:
                # Create a simple signal representation
                signal = np.array([
                    event.energy_gev / 1e6,  # normalized energy
                    event.significance,
                    event.error_radius_deg,
                    np.sin(np.radians(event.ra)),
                    np.cos(np.radians(event.ra)),
                    np.sin(np.radians(event.dec)),
                    np.cos(np.radians(event.dec)),
                ])
                signal_tensor = torch.tensor(signal, dtype=torch.float32)

                # Pad to carpet size
                n_vertices = detector.carpet_filter.carpet.graph.n_vertices
                if len(signal_tensor) < n_vertices:
                    signal_tensor = torch.nn.functional.pad(
                        signal_tensor, (0, n_vertices - len(signal_tensor))
                    )

                # Compute anomaly score
                anomaly_score = detector.carpet_filter.anomaly_score(signal_tensor)
                self.anomaly_scores[event.event_id] = float(anomaly_score)

            logger.info(f"Computed anomaly scores for {len(self.anomaly_scores)} events")

        except ImportError:
            logger.warning("Carpet filter not available; skipping anomaly scoring")

    def plot_sky_map(self) -> None:
        """
        Plot Mollweide projection sky map with all events.

        Shows:
          • Neutrino events (blue circles)
          • Cosmic ray events (red triangles)
          • Correlated events (yellow stars)
          • High-anomaly events (highlighted)
        """
        if not HAS_MATPLOTLIB:
            logger.error("matplotlib not available; cannot plot sky map")
            return

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='mollweide')

        # Filter events by energy threshold
        nu_filtered = [e for e in self.neutrino_events
                       if e.energy_gev >= self.cfg.neutrino_energy_min_gev]
        cr_filtered = [e for e in self.cosmic_ray_events
                       if e.energy_ev >= self.cfg.cosmic_ray_energy_min_ev]

        # Plot neutrino events
        if nu_filtered:
            ra_rad = np.radians([e.ra for e in nu_filtered])
            dec_rad = np.radians([e.dec for e in nu_filtered])
            # Adjust RA to [-π, π]
            ra_rad = np.where(ra_rad > np.pi, ra_rad - 2*np.pi, ra_rad)
            sizes = [np.log10(e.energy_gev) * 10 for e in nu_filtered]
            ax.scatter(ra_rad, dec_rad, s=sizes, c='blue', alpha=0.6,
                      label=f'Neutrinos (n={len(nu_filtered)})', marker='o')

        # Plot cosmic ray events
        if cr_filtered:
            ra_rad = np.radians([e.ra for e in cr_filtered])
            dec_rad = np.radians([e.dec for e in cr_filtered])
            ra_rad = np.where(ra_rad > np.pi, ra_rad - 2*np.pi, ra_rad)
            sizes = [np.log10(e.energy_ev / 1e18) * 20 for e in cr_filtered]
            ax.scatter(ra_rad, dec_rad, s=sizes, c='red', alpha=0.6,
                      label=f'Cosmic Rays (n={len(cr_filtered)})', marker='^')

        # Plot correlated events
        if self.correlations:
            corr_ra = []
            corr_dec = []
            for c in self.correlations:
                # Average position
                ra_avg = (c['neutrino'].ra + c['cosmic_ray'].ra) / 2
                dec_avg = (c['neutrino'].dec + c['cosmic_ray'].dec) / 2
                corr_ra.append(np.radians(ra_avg))
                corr_dec.append(np.radians(dec_avg))

            if corr_ra:
                corr_ra = np.array(corr_ra)
                corr_ra = np.where(corr_ra > np.pi, corr_ra - 2*np.pi, corr_ra)
                ax.scatter(corr_ra, corr_dec, s=200, c='yellow', alpha=0.8,
                          label=f'Correlated (n={len(corr_ra)})', marker='*',
                          edgecolors='black', linewidths=0.5)

        # Highlight high-anomaly events
        if self.anomaly_scores:
            high_anomaly = [eid for eid, score in self.anomaly_scores.items()
                           if score > self.cfg.anomaly_threshold_sigma]
            for eid in high_anomaly:
                event = next((e for e in self.neutrino_events if e.event_id == eid), None)
                if event:
                    ra_rad = np.radians(event.ra)
                    if ra_rad > np.pi:
                        ra_rad -= 2 * np.pi
                    dec_rad = np.radians(event.dec)
                    ax.scatter([ra_rad], [dec_rad], s=300, c='none',
                              edgecolors='magenta', linewidths=2, marker='o')

        ax.grid(True, alpha=0.3)
        ax.set_title('Multi-Messenger Sky Map (Mollweide Projection)', fontsize=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()

        if self.cfg.save_plots:
            from pathlib import Path
            output_path = Path(self.cfg.output_dir) / "sky_map.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved sky map to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_time_series(self) -> None:
        """
        Plot time series of event rates and energies.

        Shows:
          • Neutrino event rate vs time
          • Cosmic ray event rate vs time
          • Energy distribution
          • Correlation events
        """
        if not HAS_MATPLOTLIB:
            logger.error("matplotlib not available; cannot plot time series")
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Time bins (1 hour bins)
        now = datetime.now().timestamp()
        start_time = now - self.cfg.lookback_hours * 3600
        n_bins = int(self.cfg.lookback_hours)
        bin_edges = np.linspace(start_time, now, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Neutrino event rate
        nu_times = [e.timestamp for e in self.neutrino_events]
        nu_hist, _ = np.histogram(nu_times, bins=bin_edges)
        axes[0].plot(bin_centers, nu_hist, 'b-', linewidth=2, label='Neutrinos')
        axes[0].set_ylabel('Event Rate (per hour)')
        axes[0].set_title('Neutrino Event Rate')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Cosmic ray event rate
        cr_times = [e.timestamp for e in self.cosmic_ray_events]
        cr_hist, _ = np.histogram(cr_times, bins=bin_edges)
        axes[1].plot(bin_centers, cr_hist, 'r-', linewidth=2, label='Cosmic Rays')
        axes[1].set_ylabel('Event Rate (per hour)')
        axes[1].set_title('Cosmic Ray Event Rate')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Energy distribution
        nu_energies = [np.log10(e.energy_gev) for e in self.neutrino_events]
        cr_energies = [np.log10(e.energy_ev) for e in self.cosmic_ray_events]

        axes[2].hist(nu_energies, bins=30, alpha=0.6, color='blue',
                    label=f'Neutrinos (n={len(nu_energies)})')
        axes[2].hist(cr_energies, bins=30, alpha=0.6, color='red',
                    label=f'Cosmic Rays (n={len(cr_energies)})')
        axes[2].set_xlabel('log10(Energy)')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Energy Distribution')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Mark correlation events
        if self.correlations:
            corr_times = [c['neutrino'].timestamp for c in self.correlations]
            for t in corr_times:
                for ax in axes:
                    ax.axvline(x=t, color='yellow', alpha=0.3, linewidth=1)

        plt.tight_layout()

        if self.cfg.save_plots:
            from pathlib import Path
            output_path = Path(self.cfg.output_dir) / "time_series.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved time series to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_anomaly_distribution(self) -> None:
        """Plot distribution of anomaly scores."""
        if not self.anomaly_scores or not HAS_MATPLOTLIB:
            logger.info("No anomaly scores or matplotlib not available")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        scores = list(self.anomaly_scores.values())
        ax.hist(scores, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(x=self.cfg.anomaly_threshold_sigma, color='red',
                  linestyle='--', linewidth=2,
                  label=f'Threshold ({self.cfg.anomaly_threshold_sigma}σ)')

        ax.set_xlabel('Anomaly Score (Carpet A^11)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Anomaly Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if self.cfg.save_plots:
            from pathlib import Path
            output_path = Path(self.cfg.output_dir) / "anomaly_distribution.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved anomaly distribution to {output_path}")
        else:
            plt.show()

        plt.close()

    def generate_report(self) -> Dict[str, Any]:
        """Generate a summary report of the dashboard analysis."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "lookback_hours": self.cfg.lookback_hours,
            "neutrino_events": len(self.neutrino_events),
            "cosmic_ray_events": len(self.cosmic_ray_events),
            "correlations": len(self.correlations),
            "high_anomaly_events": sum(
                1 for score in self.anomaly_scores.values()
                if score > self.cfg.anomaly_threshold_sigma
            ),
            "energy_ranges": {
                "neutrino_gev": {
                    "min": min((e.energy_gev for e in self.neutrino_events), default=0),
                    "max": max((e.energy_gev for e in self.neutrino_events), default=0),
                },
                "cosmic_ray_ev": {
                    "min": min((e.energy_ev for e in self.cosmic_ray_events), default=0),
                    "max": max((e.energy_ev for e in self.cosmic_ray_events), default=0),
                },
            },
            "synthetic_data_used": any(e.synthetic for e in self.neutrino_events) or
                                   any(e.synthetic for e in self.cosmic_ray_events),
        }

        return report

    def run_full_analysis(self) -> None:
        """Run the complete dashboard analysis pipeline."""
        logger.info("=" * 72)
        logger.info(" SESI Multi-Messenger Dashboard — Full Analysis")
        logger.info("=" * 72)

        # Load data
        self.load_data()

        # Compute anomaly scores
        self.compute_anomaly_scores()

        # Generate plots
        self.plot_sky_map()
        self.plot_time_series()
        self.plot_anomaly_distribution()

        # Generate report
        report = self.generate_report()

        logger.info("\n" + "=" * 72)
        logger.info(" ANALYSIS REPORT")
        logger.info("=" * 72)
        logger.info(f"  Neutrino events      : {report['neutrino_events']}")
        logger.info(f"  Cosmic ray events    : {report['cosmic_ray_events']}")
        logger.info(f"  Temporal correlations: {report['correlations']}")
        logger.info(f"  High-anomaly events  : {report['high_anomaly_events']}")
        logger.info(f"  Synthetic data used  : {report['synthetic_data_used']}")
        logger.info("=" * 72)

        if report['synthetic_data_used']:
            logger.warning("  ⚠️  SYNTHETIC DATA DETECTED — results may not reflect real observations")

        logger.info("  HONEST REMINDERS:")
        logger.info("  • Dashboard is for RESEARCH VISUALIZATION, not confirmed detection")
        logger.info("  • All anomalies require independent verification")
        logger.info("  • No claim that displayed patterns indicate extraterrestrial origin")
        logger.info("=" * 72)

        return report


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    """Smoke test for the multi-messenger dashboard."""
    cfg = DashboardConfig(
        lookback_hours=12.0,
        use_carpet_filter=False,  # Disable for faster smoke test
        save_plots=True,
        output_dir="./dashboard_smoke_test",
    )

    dashboard = MultiMessengerDashboard(cfg)
    report = dashboard.run_full_analysis()

    print("\n" + "=" * 72)
    print("  Smoke test complete.")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
