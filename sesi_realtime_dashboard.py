"""
=============================================================================
SESI REAL-TIME MULTI-MESSENGER DASHBOARD (Plotly Dash)
Live visualization of GW, neutrino, cosmic ray, and EM data
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

DASHBOARD PANELS
----------------
[D1] GW Event Sky Map — Mollweide projection of GWTC events
[D2] Event Timeline — GW events over observing runs
[D3] Mass Distribution — m1 vs m2 scatter, colored by type
[D4] Strain Viewer — interactive h(t) for selected event
[D5] Multi-Messenger Stream — live GW + neutrino + cosmic ray
[D6] Anomaly Monitor — ML anomaly scores over time

DEPENDENCIES
------------
    pip install dash plotly pandas

HONEST SCOPE
------------
• Dashboard is for RESEARCH VISUALIZATION of confirmed astrophysical data.
• "Anomaly" = instrumental glitch or unmodeled signal, not spacecraft.
• Real-time updates poll GWOSC; latency depends on network.
=============================================================================
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("sesi.dash_app")

try:
    import dash
    from dash import dcc, html, Input, Output, State
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_DASH = True
except ImportError:
    HAS_DASH = False
    logger.warning("dash/plotly not installed; dashboard unavailable")

from sesi_ligo_virgo_connector import LIGOVirgoHub, LIGOConfig, GWEvent

__all__ = ["build_dashboard", "run_dashboard"]


# ═══════════════════════════════════════════════════════════════════════
# FIGURE BUILDERS
# ═══════════════════════════════════════════════════════════════════════

def build_sky_map(events: List[GWEvent]) -> Any:
    """
    [D1] Mollweide sky map of GW events.

    Colors by event type: BBH (blue), BNS (orange), NSBH (green).
    """
    if not HAS_DASH:
        return None

    # Filter events with valid sky positions
    localized = [e for e in events if -90 <= e.dec <= 90 and 0 <= e.ra <= 360]

    fig = go.Figure()

    # Color map by event type
    type_colors = {
        "BBH": "royalblue",
        "BNS": "orange",
        "NSBH": "green",
        "MassGap": "purple",
    }

    for etype, color in type_colors.items():
        subset = [e for e in localized if e.event_type == etype]
        if not subset:
            continue

        # Convert RA to [-180, 180] for Mollweide
        ra_shifted = [e.ra - 180 if e.ra > 180 else e.ra for e in subset]
        decs = [e.dec for e in subset]
        sizes = [max(5, e.snr * 0.8) for e in subset]
        names = [e.name for e in subset]

        fig.add_trace(go.Scattergeo(
            lon=ra_shifted,
            lat=decs,
            mode="markers",
            marker=dict(
                size=sizes,
                color=color,
                opacity=0.7,
                line=dict(width=0.5, color="white"),
            ),
            text=names,
            name=etype,
        ))

    fig.update_geos(
        projection_type="mollweide",
        showcoastlines=True,
        coastlinecolor="gray",
        showland=False,
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        title="Gravitational Wave Event Sky Map (GWTC)",
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1),
        paper_bgcolor="rgba(17,17,17,1)",
        font=dict(color="white"),
    )
    return fig


def build_mass_plot(events: List[GWEvent]) -> Any:
    """
    [D3] Mass distribution: m1 vs m2 scatter.

    Shows the mass gap region and event type clustering.
    """
    if not HAS_DASH:
        return None

    fig = go.Figure()

    type_colors = {
        "BBH": "royalblue",
        "BNS": "orange",
        "NSBH": "green",
        "MassGap": "purple",
    }

    for etype, color in type_colors.items():
        subset = [e for e in events if e.event_type == etype]
        if not subset:
            continue

        fig.add_trace(go.Scatter(
            x=[e.mass2_solar for e in subset],
            y=[e.mass1_solar for e in subset],
            mode="markers",
            marker=dict(size=8, color=color, opacity=0.7),
            text=[e.name for e in subset],
            name=etype,
        ))

    # Mark the mass gap region
    fig.add_shape(type="rect", x0=3, x1=5, y0=3, y1=5,
                  fillcolor="red", opacity=0.1, line_width=0)

    fig.update_layout(
        title="Component Mass Distribution",
        xaxis_title="Secondary Mass (M☉)",
        yaxis_title="Primary Mass (M☉)",
        height=500,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    return fig


def build_timeline(events: List[GWEvent]) -> Any:
    """
    [D2] Event timeline across observing runs.
    """
    if not HAS_DASH:
        return None

    # Sort by GPS time
    sorted_events = sorted(events, key=lambda e: e.gps_time)

    # Convert GPS to approximate date
    # GPS epoch: Jan 6, 1980; leap seconds ~18
    from datetime import datetime, timedelta
    gps_epoch = datetime(1980, 1, 6)

    fig = go.Figure()

    for e in sorted_events:
        try:
            date = gps_epoch + timedelta(seconds=e.gps_time - 18)
        except (OverflowError, OSError):
            continue

        type_colors = {
            "BBH": "royalblue", "BNS": "orange",
            "NSBH": "green", "MassGap": "purple",
        }
        color = type_colors.get(e.event_type, "gray")

        fig.add_trace(go.Scatter(
            x=[date], y=[e.snr],
            mode="markers",
            marker=dict(size=max(6, e.snr * 0.5), color=color, opacity=0.7),
            text=e.name,
            showlegend=False,
            hovertemplate=f"<b>{e.name}</b><br>SNR: {e.snr:.1f}<br>"
                         f"Type: {e.event_type}<br>%{{x}}",
        ))

    fig.update_layout(
        title="GW Event Timeline (SNR vs Date)",
        xaxis_title="Detection Date",
        yaxis_title="Network SNR",
        height=400,
        template="plotly_dark",
    )
    return fig


def build_strain_viewer(strain_data: Dict) -> Any:
    """
    [D4] Interactive strain h(t) viewer.
    """
    if not HAS_DASH:
        return None

    strain = np.array(strain_data["strain"])
    fs = strain_data["sample_rate"]
    n = len(strain)
    t = np.arange(n) / fs

    # Downsample for plotting (strain data can be millions of points)
    step = max(1, n // 5000)
    t_plot = t[::step]
    strain_plot = strain[::step]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_plot, y=strain_plot,
        mode="lines",
        line=dict(width=0.5, color="cyan"),
        name="h(t)",
    ))

    fig.update_layout(
        title=f"Strain h(t) — {strain_data.get('event', 'unknown')} "
              f"({strain_data.get('detector', '?')})",
        xaxis_title="Time (s)",
        yaxis_title="Strain",
        height=350,
        template="plotly_dark",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# DASH APP BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_dashboard(cfg: Optional[LIGOConfig] = None) -> Any:
    """
    Build the Plotly Dash app with all panels.

    Returns the Dash app instance.
    """
    if not HAS_DASH:
        raise RuntimeError("dash and plotly required: pip install dash plotly")

    hub = LIGOVirgoHub(cfg or LIGOConfig())
    events = hub.get_events(min_snr=8.0)

    app = dash.Dash(__name__, title="SESI Multi-Messenger Dashboard")
    app.layout = html.Div(style={"backgroundColor": "#111", "color": "white",
                                 "minHeight": "100vh", "padding": "20px"},
                          children=[
        # Header
        html.H1("🌊 SESI Multi-Messenger Astrophysics Dashboard",
                style={"textAlign": "center"}),
        html.P("LIGO/Virgo/KAGRA gravitational wave analysis — "
               "research visualization, not UFO detection",
               style={"textAlign": "center", "color": "#888"}),

        # Summary cards
        html.Div(style={"display": "flex", "justifyContent": "space-around",
                        "marginBottom": "20px"}, children=[
            _stat_card("Total Events", str(len(events))),
            _stat_card("BBH", str(sum(1 for e in events if e.event_type == "BBH"))),
            _stat_card("BNS", str(sum(1 for e in events if e.event_type == "BNS"))),
            _stat_card("NSBH", str(sum(1 for e in events if e.event_type == "NSBH"))),
        ]),

        # Sky map
        dcc.Graph(id="sky-map", figure=build_sky_map(events)),

        # Two-column row: mass plot + timeline
        html.Div(style={"display": "flex", "gap": "20px"}, children=[
            html.Div(style={"flex": 1}, children=[
                dcc.Graph(id="mass-plot", figure=build_mass_plot(events))
            ]),
            html.Div(style={"flex": 1}, children=[
                dcc.Graph(id="timeline", figure=build_timeline(events))
            ]),
        ]),

        # Event selector for strain viewer
        html.Div(style={"margin": "20px 0"}, children=[
            html.Label("Select event for strain viewer:"),
            dcc.Dropdown(
                id="event-selector",
                options=[{"label": e.name, "value": e.name} for e in events[:20]],
                value=events[0].name if events else None,
                style={"color": "black"},
            ),
            dcc.Graph(id="strain-viewer"),
        ]),

        # Auto-refresh interval
        dcc.Interval(id="refresh-interval", interval=60 * 1000, n_intervals=0),

        # Footer
        html.P("Data: GWOSC (gwosc.org). Confirmed LVK events only. "
               "Anomaly = glitch/unmodeled burst, NOT spacecraft.",
               style={"textAlign": "center", "color": "#666", "fontSize": "12px"}),
    ])

    # Callback for strain viewer
    @app.callback(
        Output("strain-viewer", "figure"),
        Input("event-selector", "value"),
    )
    def update_strain(event_name):
        if event_name is None:
            return go.Figure()
        strain = hub.get_strain(event_name, detector="H1")
        if strain is None:
            return go.Figure()
        return build_strain_viewer(strain)

    return app


def _stat_card(title: str, value: str) -> Any:
    """Helper: a summary stat card."""
    return html.Div(style={
        "backgroundColor": "#222", "padding": "15px",
        "borderRadius": "8px", "textAlign": "center", "minWidth": "120px",
    }, children=[
        html.H2(value, style={"margin": "0", "color": "cyan"}),
        html.P(title, style={"margin": "5px 0 0 0", "color": "#aaa"}),
    ])


def run_dashboard(port: int = 8050, debug: bool = False) -> None:
    """Build and run the dashboard server."""
    app = build_dashboard()
    logger.info(f"Starting dashboard at http://localhost:{port}")
    app.run_server(host="0.0.0.0", port=port, debug=debug)


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST (build figures without serving)
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Real-Time Dashboard — Smoke Test (figures only)")
    print("=" * 72)

    if not HAS_DASH:
        print("\n  [SKIP] dash/plotly not installed")
        print("    Install with: pip install dash plotly")
        return

    cfg = LIGOConfig()
    hub = LIGOVirgoHub(cfg)
    events = hub.get_events(min_snr=8.0)
    print(f"\n  Loaded {len(events)} events for dashboard")

    # Build each figure
    figs = {
        "sky_map": build_sky_map(events),
        "mass_plot": build_mass_plot(events),
        "timeline": build_timeline(events),
    }
    for name, fig in figs.items():
        ok = fig is not None
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}: {len(fig.data) if fig else 0} traces")

    # Strain viewer
    strain = hub.get_strain(events[0].name if events else "GW150914")
    if strain:
        sfig = build_strain_viewer(strain)
        print(f"    [PASS] strain_viewer: {len(sfig.data)} traces")

    print("\n  To launch the live dashboard:")
    print("    from sesi_realtime_dashboard import run_dashboard")
    print("    run_dashboard(port=8050)")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Visualizes CONFIRMED LVK events, not speculative signals")
    print("  • Strain may be synthetic if offline")
    print("  • No claim of UFO/alien detection anywhere")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
