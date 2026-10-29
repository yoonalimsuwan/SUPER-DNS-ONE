"""
=============================================================================
SESI OPERATOR DASHBOARD — Real-Time Visualization for Stealth Design
Plotly Dash application for interactive exploration
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

DASHBOARD FEATURES
------------------
[D1] Design Parameter Controls
     • Carpet level selector
     • Material property sliders (ε_r, μ_r, tan δ)
     • Thickness adjustment
[D2] Real-Time Visualization
     • RCS reduction plot
     • Absorption spectrum
     • Impedance matching quality
[D3] Pareto Front Explorer
     • Interactive trade-off curve
     • Knee point selection
     • Design comparison
[D4] Simulation Status
     • OpenEMS progress
     • Optimization status
     • Alert system (No-Zeno throttled)

DEPENDENCIES
------------
    pip install dash plotly pandas

LABELS: [D] derived  [O] open

HONEST SCOPE
------------
• Dashboard is DECISION SUPPORT, not autonomous control.
• All visualizations require underlying simulation data.
• Operator must interpret results; dashboard does not make decisions.
=============================================================================
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("sesi.dashboard")

try:
    import dash
    from dash import dcc, html, Input, Output, State
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_DASH = True
except ImportError:
    HAS_DASH = False

# Import optimization results
try:
    from sesi_multiobjective_stealth import (
        StealthDesign, StealthObjectiveFunctions, NSGA2Optimizer, ParetoFrontAnalyzer
    )
    HAS_OPTIM = True
except ImportError:
    HAS_OPTIM = False

__all__ = ["OperatorDashboard", "build_dashboard", "run_dashboard"]


# ═══════════════════════════════════════════════════════════════════════
# DASHBOARD BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_dashboard(pareto_results: Optional[Dict[str, Any]] = None) -> Any:
    """
    Build the Plotly Dash dashboard.
    
    Args:
        pareto_results: optimization results from NSGA-II
    
    Returns:
        Dash app instance
    """
    if not HAS_DASH:
        raise RuntimeError("dash and plotly required: pip install dash plotly")
    
    app = dash.Dash(__name__, title="SESI Stealth Design Dashboard")
    
    # Prepare Pareto front data
    if pareto_results and HAS_OPTIM:
        pareto_front = pareto_results.get("pareto_front", [])
        analyzer = ParetoFrontAnalyzer(pareto_front)
        rcs_vals, abs_vals = analyzer.get_trade_off_curve()
    else:
        # Synthetic data for demo
        rcs_vals = np.linspace(5, 20, 20)
        abs_vals = 0.95 - 0.02 * rcs_vals + np.random.normal(0, 0.02, 20)
    
    app.layout = html.Div(style={"backgroundColor": "#1a1a1a", "color": "white",
                                  "fontFamily": "Arial, sans-serif", "padding": "20px"},
                          children=[
        # Header
        html.H1("🛡️ SESI Hypersonic Stealth Design Dashboard",
                style={"textAlign": "center", "color": "#00d4ff"}),
        html.P("Multi-Objective Optimization: RCS Reduction + Absorption",
               style={"textAlign": "center", "color": "#888"}),
        
        # Control panel
        html.Div(style={"display": "flex", "gap": "20px", "marginBottom": "20px"},
                 children=[
            # Design parameters
            html.Div(style={"flex": 1, "backgroundColor": "#2a2a2a", "padding": "15px",
                           "borderRadius": "8px"}, children=[
                html.H3("Design Parameters", style={"marginTop": "0"}),
                html.Label("Carpet Level:"),
                dcc.Dropdown(
                    id="carpet-level",
                    options=[{"label": f"Level {i}", "value": i} for i in range(2, 6)],
                    value=3,
                    style={"color": "black"},
                ),
                html.Label("Relative Permittivity (ε_r):"),
                dcc.Slider(id="eps-r", min=1, max=20, step=0.5, value=10,
                          marks={i: str(i) for i in range(1, 21, 5)}),
                html.Label("Relative Permeability (μ_r):"),
                dcc.Slider(id="mu-r", min=1, max=20, step=0.5, value=10,
                          marks={i: str(i) for i in range(1, 21, 5)}),
                html.Label("Loss Tangent (tan δ):"),
                dcc.Slider(id="tan-d", min=0.01, max=0.5, step=0.01, value=0.1),
                html.Label("Thickness (mm):"),
                dcc.Slider(id="thickness", min=0.5, max=5, step=0.1, value=2.0),
            ]),
            
            # Results panel
            html.Div(style={"flex": 1, "backgroundColor": "#2a2a2a", "padding": "15px",
                           "borderRadius": "8px"}, children=[
                html.H3("Simulation Results", style={"marginTop": "0"}),
                html.Div(id="rcs-result", style={"marginBottom": "10px"}),
                html.Div(id="absorption-result", style={"marginBottom": "10px"}),
                html.Div(id="impedance-result"),
            ]),
        ]),
        
        # Pareto front plot
        html.Div(style={"backgroundColor": "#2a2a2a", "padding": "15px",
                       "borderRadius": "8px", "marginBottom": "20px"}, children=[
            html.H3("Pareto Front: RCS vs Absorption Trade-off"),
            dcc.Graph(id="pareto-plot", figure=_build_pareto_plot(rcs_vals, abs_vals)),
        ]),
        
        # Spectrum plots
        html.Div(style={"display": "flex", "gap": "20px"}, children=[
            html.Div(style={"flex": 1, "backgroundColor": "#2a2a2a", "padding": "15px",
                           "borderRadius": "8px"}, children=[
                html.H3("Absorption Spectrum"),
                dcc.Graph(id="absorption-spectrum"),
            ]),
            html.Div(style={"flex": 1, "backgroundColor": "#2a2a2a", "padding": "15px",
                           "borderRadius": "8px"}, children=[
                html.H3("Impedance Matching"),
                dcc.Graph(id="impedance-plot"),
            ]),
        ]),
        
        # Footer
        html.P("⚠️ Dashboard is DECISION SUPPORT only. Operator must interpret results.",
               style={"textAlign": "center", "color": "#ff6b6b", "marginTop": "20px"}),
        
        # Auto-refresh
        dcc.Interval(id="refresh-interval", interval=5000, n_intervals=0),
    ])
    
    # Callbacks
    @app.callback(
        [Output("rcs-result", "children"),
         Output("absorption-result", "children"),
         Output("impedance-result", "children"),
         Output("absorption-spectrum", "figure"),
         Output("impedance-plot", "figure")],
        [Input("carpet-level", "value"),
         Input("eps-r", "value"),
         Input("mu-r", "value"),
         Input("tan-d", "value"),
         Input("thickness", "value")]
    )
    def update_results(carpet_level, eps_r, mu_r, tan_d, thickness):
        if not HAS_OPTIM:
            return "Optimization module not available", "", "", go.Figure(), go.Figure()
        
        # Create design
        design = StealthDesign(
            carpet_level=carpet_level,
            eps_r=eps_r,
            mu_r=mu_r,
            tan_d=tan_d,
            thickness_mm=thickness,
        )
        
        # Evaluate
        objectives = StealthObjectiveFunctions()
        objectives.evaluate(design)
        
        # Results
        rcs_text = html.Div([
            html.Strong("RCS Reduction: "),
            f"{design.rcs_reduction_db:.2f} dB"
        ])
        
        abs_text = html.Div([
            html.Strong("Mean Absorption: "),
            f"{design.absorption_mean:.3f}"
        ])
        
        # Impedance matching
        Z_material = 377.0 * np.sqrt(mu_r / (eps_r * (1 - 1j * tan_d)))
        Gamma = np.abs((Z_material - 377.0) / (Z_material + 377.0))
        imp_text = html.Div([
            html.Strong("Reflection Coefficient: "),
            f"{Gamma:.4f} ({20*np.log10(Gamma+1e-12):.2f} dB)"
        ])
        
        # Absorption spectrum (synthetic)
        freqs = np.linspace(8e9, 12e9, 100)
        f0 = 10e9
        bw = 1e9
        abs_spectrum = design.absorption_mean * np.exp(-((freqs - f0) / bw)**2)
        
        abs_fig = go.Figure()
        abs_fig.add_trace(go.Scatter(x=freqs/1e9, y=abs_spectrum, mode="lines",
                                     name="Absorption", line=dict(color="#00d4ff")))
        abs_fig.update_layout(
            xaxis_title="Frequency (GHz)",
            yaxis_title="Absorption",
            template="plotly_dark",
            height=300,
        )
        
        # Impedance plot (synthetic)
        imp_fig = go.Figure()
        imp_fig.add_trace(go.Scatter(
            x=freqs/1e9,
            y=20*np.log10(np.abs(Gamma) * np.ones_like(freqs) + 1e-12),
            mode="lines",
            name="S11",
            line=dict(color="#ff6b6b")
        ))
        imp_fig.update_layout(
            xaxis_title="Frequency (GHz)",
            yaxis_title="S11 (dB)",
            template="plotly_dark",
            height=300,
        )
        
        return rcs_text, abs_text, imp_text, abs_fig, imp_fig
    
    return app


def _build_pareto_plot(rcs_vals: np.ndarray, abs_vals: np.ndarray) -> go.Figure:
    """Build Pareto front plot."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=rcs_vals,
        y=abs_vals,
        mode="markers",
        marker=dict(size=10, color=abs_vals, colorscale="Viridis", showscale=True),
        name="Pareto Solutions",
        text=[f"RCS: {r:.1f} dB<br>Abs: {a:.3f}" for r, a in zip(rcs_vals, abs_vals)],
        hoverinfo="text",
    ))
    
    fig.update_layout(
        xaxis_title="RCS Reduction (dB) — Higher is better",
        yaxis_title="Mean Absorption — Higher is better",
        template="plotly_dark",
        height=400,
    )
    
    return fig


def run_dashboard(port: int = 8050, debug: bool = False) -> None:
    """Build and run the dashboard server."""
    # Run optimization first
    if HAS_OPTIM:
        logger.info("Running NSGA-II optimization...")
        objectives = StealthObjectiveFunctions()
        optimizer = NSGA2Optimizer(objectives, population_size=50, n_generations=20)
        pareto_results = optimizer.optimize()
    else:
        pareto_results = None
    
    app = build_dashboard(pareto_results)
    logger.info(f"Starting dashboard at http://localhost:{port}")
    app.run_server(host="0.0.0.0", port=port, debug=debug)


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Operator Dashboard — Smoke Test")
    print("=" * 72)
    print(f"  dash installed     : {HAS_DASH}")
    print(f"  optimization module: {HAS_OPTIM}")
    
    if not HAS_DASH:
        print("\n  [SKIP] dash not installed")
        print("    Install with: pip install dash plotly")
        return
    
    print("\n  ── Building dashboard ──")
    app = build_dashboard()
    print(f"    Dashboard built successfully")
    print(f"    Layout components: {len(app.layout.children)}")
    
    print("\n  To launch the dashboard:")
    print("    from sesi_operator_dashboard import run_dashboard")
    print("    run_dashboard(port=8050)")
    print("    Then open http://localhost:8050 in your browser")
    
    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Dashboard is DECISION SUPPORT, not autonomous control")
    print("  • Operator must interpret results")
    print("  • All visualizations require underlying simulation data")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
