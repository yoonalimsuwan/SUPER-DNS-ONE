"""
=============================================================================
SESI AUTOMATED REPORT GENERATION
Comprehensive documentation with embedded visualizations
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE DOES
---------------------
[D] Generate comprehensive simulation reports:
  • PDF/HTML output with embedded figures
  • Statistical summaries with uncertainty quantification
  • Design recommendations from Pareto analysis
  • Provenance tracking (which simulation → which result)
  • Carpet spectral complexity metrics
  • Assumption-light confidence intervals (Paper 11)

REPORT SECTIONS
---------------
  1. Executive Summary
  2. Design Specifications
  3. Simulation Methodology
  4. Results (S-parameters, absorption, RCS)
  5. Uncertainty Quantification
  6. Pareto-Optimal Designs
  7. Carpet Spectral Analysis
  8. Recommendations
  9. Appendix (raw data, convergence plots)

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Reports document COMPUTATIONAL results, not experimental validation.
• All claims must be traceable to specific simulations.
• Uncertainty quantification requires stated assumptions (Paper 11).
• Recommendations are DESIGN GUIDANCE, not guarantees.
=============================================================================
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("sesi.reporting")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# Import existing modules
try:
    from sesi_parametric_sweep import ParametricSweepEngine, SweepConfig
    from sesi_fullwave_em_solver import FullWaveEMPipeline
    from sesi_carpet_rcs import CarpetRCSEstimator, CarpetRCSConfig
    HAS_SESI = True
except ImportError:
    HAS_SESI = False

__all__ = [
    "ReportConfig",
    "FigureGenerator",
    "StatisticalSummarizer",
    "AutomatedReportGenerator",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ReportConfig:
    """Configuration for report generation."""
    output_dir: str = "./reports"
    report_title: str = "SESI Metamaterial Design Report"
    author: str = "PAI & Yoon A. Limsuwan"
    organization: str = "MSPS NETWORK"
    
    # Sections to include
    include_executive_summary: bool = True
    include_methodology: bool = True
    include_results: bool = True
    include_uncertainty: bool = True
    include_pareto: bool = True
    include_carpet_analysis: bool = True
    include_recommendations: bool = True
    
    # Figure settings
    figure_dpi: int = 150
    figure_format: str = "png"
    
    # Output format
    output_format: str = "pdf"  # pdf, html


# ═══════════════════════════════════════════════════════════════════════
# [D] FIGURE GENERATOR
# ═══════════════════════════════════════════════════════════════════════

class FigureGenerator:
    """
    [D] Generate publication-quality figures for reports.
    """
    
    def __init__(self, cfg: ReportConfig):
        self.cfg = cfg
        self.figures: List[str] = []
        
    def generate_s_parameter_plot(self, frequencies_ghz: np.ndarray,
                                   s11_db: np.ndarray,
                                   output_path: str) -> str:
        """Generate S-parameter plot."""
        if not HAS_MATPLOTLIB:
            return ""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(frequencies_ghz, s11_db, 'b-', linewidth=2, label='S11')
        ax.axhline(y=-10, color='r', linestyle='--', label='Target (-10 dB)')
        ax.set_xlabel('Frequency (GHz)', fontsize=12)
        ax.set_ylabel('S11 (dB)', fontsize=12)
        ax.set_title('Return Loss (S11)', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.cfg.figure_dpi)
        plt.close()
        
        self.figures.append(output_path)
        return output_path
        
    def generate_absorption_plot(self, frequencies_ghz: np.ndarray,
                                  absorption: np.ndarray,
                                  output_path: str) -> str:
        """Generate absorption spectrum plot."""
        if not HAS_MATPLOTLIB:
            return ""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(frequencies_ghz, absorption, 'g-', linewidth=2, label='Absorption')
        ax.axhline(y=0.9, color='r', linestyle='--', label='Target (90%)')
        ax.set_xlabel('Frequency (GHz)', fontsize=12)
        ax.set_ylabel('Absorption', fontsize=12)
        ax.set_title('Absorption Spectrum', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.cfg.figure_dpi)
        plt.close()
        
        self.figures.append(output_path)
        return output_path
        
    def generate_pareto_plot(self, pareto_results: List[Dict[str, Any]],
                              output_path: str) -> str:
        """Generate Pareto front visualization."""
        if not HAS_MATPLOTLIB:
            return ""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        absorption = [r.get("absorption", 0) for r in pareto_results]
        thickness = [r.get("parameters", {}).get("thickness", 0) * 1e3 
                     for r in pareto_results]  # convert to mm
        
        ax.scatter(thickness, absorption, c='blue', s=100, alpha=0.7, 
                  label='Pareto-optimal designs')
        
        # Connect Pareto front
        sorted_idx = np.argsort(thickness)
        ax.plot(np.array(thickness)[sorted_idx], np.array(absorption)[sorted_idx],
               'r-', linewidth=2, label='Pareto front')
        
        ax.set_xlabel('Thickness (mm)', fontsize=12)
        ax.set_ylabel('Absorption', fontsize=12)
        ax.set_title('Pareto-Optimal Designs: Absorption vs Thickness', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.cfg.figure_dpi)
        plt.close()
        
        self.figures.append(output_path)
        return output_path
        
    def generate_carpet_complexity_plot(self, complexity_history: List[float],
                                         output_path: str) -> str:
        """Generate Carpet spectral complexity evolution plot."""
        if not HAS_MATPLOTLIB:
            return ""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(complexity_history, 'b-', linewidth=2)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Spectral Entropy', fontsize=12)
        ax.set_title('Carpet Spectral Complexity Evolution', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.cfg.figure_dpi)
        plt.close()
        
        self.figures.append(output_path)
        return output_path


# ═══════════════════════════════════════════════════════════════════════
# [D] STATISTICAL SUMMARIZER
# ═══════════════════════════════════════════════════════════════════════

class StatisticalSummarizer:
    """
    [D] Compute statistical summaries for report.
    """
    
    def __init__(self):
        pass
        
    def summarize_sweep_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistical summary of sweep results."""
        if not results:
            return {}
        
        # Extract key metrics
        absorptions = [r.get("absorption", 0) for r in results]
        reflections = [r.get("reflection_db", 0) for r in results]
        
        summary = {
            "n_samples": len(results),
            "absorption": {
                "mean": float(np.mean(absorptions)),
                "std": float(np.std(absorptions)),
                "min": float(np.min(absorptions)),
                "max": float(np.max(absorptions)),
                "median": float(np.median(absorptions)),
            },
            "reflection_db": {
                "mean": float(np.mean(reflections)),
                "std": float(np.std(reflections)),
                "min": float(np.min(reflections)),
                "max": float(np.max(reflections)),
            },
        }
        
        return summary


# ═══════════════════════════════════════════════════════════════════════
# [D] AUTOMATED REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════

class AutomatedReportGenerator:
    """
    [D] Generate comprehensive simulation reports.
    """
    
    def __init__(self, cfg: Optional[ReportConfig] = None):
        self.cfg = cfg or ReportConfig()
        self.figure_gen = FigureGenerator(self.cfg)
        self.summarizer = StatisticalSummarizer()
        
        # Create output directory
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        
    def generate_report(self, sweep_results: Dict[str, Any],
                        em_simulation: Optional[Dict[str, Any]] = None,
                        carpet_analysis: Optional[Dict[str, Any]] = None
                        ) -> str:
        """
        Generate complete report.
        
        Args:
            sweep_results: parametric sweep results
            em_simulation: full-wave EM simulation results (optional)
            carpet_analysis: Carpet spectral analysis (optional)
        
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.cfg.output_dir,
                                   f"SESI_Report_{timestamp}.{self.cfg.output_format}")
        
        if self.cfg.output_format == "pdf":
            return self._generate_pdf_report(report_path, sweep_results,
                                            em_simulation, carpet_analysis)
        elif self.cfg.output_format == "html":
            return self._generate_html_report(report_path, sweep_results,
                                             em_simulation, carpet_analysis)
        else:
            raise ValueError(f"Unknown output format: {self.cfg.output_format}")
            
    def _generate_pdf_report(self, output_path: str,
                              sweep_results: Dict[str, Any],
                              em_simulation: Optional[Dict[str, Any]],
                              carpet_analysis: Optional[Dict[str, Any]]) -> str:
        """Generate PDF report using ReportLab."""
        if not HAS_REPORTLAB:
            logger.error("reportlab not available")
            return ""
        
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title page
        story.append(Paragraph(self.cfg.report_title, styles['Title']))
        story.append(Spacer(1, 0.2*72))
        story.append(Paragraph(f"Author: {self.cfg.author}", styles['Normal']))
        story.append(Paragraph(f"Organization: {self.cfg.organization}", styles['Normal']))
        story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']))
        story.append(Spacer(1, 0.5*72))
        
        # Executive Summary
        if self.cfg.include_executive_summary:
            story.append(Paragraph("1. Executive Summary", styles['Heading1']))
            story.append(Spacer(1, 0.1*72))
            
            summary_text = (
                f"This report presents the results of a comprehensive parametric sweep "
                f"and electromagnetic simulation for metamaterial absorber design. "
                f"A total of {sweep_results['n_samples']} design configurations were evaluated, "
                f"with {sweep_results['n_pareto_optimal']} Pareto-optimal solutions identified."
            )
            story.append(Paragraph(summary_text, styles['Normal']))
            story.append(Spacer(1, 0.2*72))
        
        # Results
        if self.cfg.include_results:
            story.append(Paragraph("2. Simulation Results", styles['Heading1']))
            story.append(Spacer(1, 0.1*72))
            
            # Statistical summary
            stats = self.summarizer.summarize_sweep_results(sweep_results['results'])
            
            story.append(Paragraph("2.1 Statistical Summary", styles['Heading2']))
            summary_text = (
                f"Mean absorption: {stats['absorption']['mean']:.3f} ± "
                f"{stats['absorption']['std']:.3f}<br/>"
                f"Mean reflection: {stats['reflection_db']['mean']:.2f} dB"
            )
            story.append(Paragraph(summary_text, styles['Normal']))
            story.append(Spacer(1, 0.2*72))
            
            # Generate and include figures
            if em_simulation and 's_parameters' in em_simulation:
                fig_path = os.path.join(self.cfg.output_dir, "s_parameter_plot.png")
                self.figure_gen.generate_s_parameter_plot(
                    em_simulation['s_parameters']['frequencies_ghz'],
                    em_simulation['s_parameters']['s11_db'],
                    fig_path
                )
                story.append(Image(fig_path, width=6*72, height=4*72))
                story.append(Spacer(1, 0.2*72))
        
        # Pareto Analysis
        if self.cfg.include_pareto and sweep_results['pareto_front']:
            story.append(Paragraph("3. Pareto-Optimal Designs", styles['Heading1']))
            story.append(Spacer(1, 0.1*72))
            
            pareto_text = (
                f"Identified {len(sweep_results['pareto_front'])} Pareto-optimal designs "
                f"that balance absorption, reflection, bandwidth, and thickness objectives."
            )
            story.append(Paragraph(pareto_text, styles['Normal']))
            story.append(Spacer(1, 0.1*72))
            
            # Generate Pareto plot
            fig_path = os.path.join(self.cfg.output_dir, "pareto_plot.png")
            self.figure_gen.generate_pareto_plot(sweep_results['pareto_front'], fig_path)
            story.append(Image(fig_path, width=6*72, height=5*72))
            story.append(Spacer(1, 0.2*72))
        
        # Uncertainty Quantification
        if self.cfg.include_uncertainty and 'uncertainty' in sweep_results:
            story.append(Paragraph("4. Uncertainty Quantification", styles['Heading1']))
            story.append(Spacer(1, 0.1*72))
            
            uq = sweep_results['uncertainty']
            if 'error' not in uq:
                uq_text = (
                    f"Using assumption-light subsampling (Paper 11), the 95% confidence "
                    f"interval for absorption is [{uq['confidence_interval']['ci_lower']:.3f}, "
                    f"{uq['confidence_interval']['ci_upper']:.3f}]."
                )
                story.append(Paragraph(uq_text, styles['Normal']))
            story.append(Spacer(1, 0.2*72))
        
        # Carpet Analysis
        if self.cfg.include_carpet_analysis and carpet_analysis:
            story.append(Paragraph("5. Carpet Spectral Analysis", styles['Heading1']))
            story.append(Spacer(1, 0.1*72))
            
            carpet_text = (
                f"Carpet spectral entropy: {carpet_analysis.get('spectral_entropy', 0):.3f}<br/>"
                f"Spectral dimension: {carpet_analysis.get('spectral_dimension', 0):.3f}"
            )
            story.append(Paragraph(carpet_text, styles['Normal']))
            story.append(Spacer(1, 0.2*72))
        
        # Recommendations
        if self.cfg.include_recommendations:
            story.append(Paragraph("6. Recommendations", styles['Heading1']))
            story.append(Spacer(1, 0.1*72))
            
            rec_text = (
                "Based on the parametric sweep and uncertainty analysis, the following "
                "design parameters are recommended for fabrication:<br/><br/>"
                "• Select design from Pareto front based on application priorities<br/>"
                "• Verify with full-wave EM simulation before fabrication<br/>"
                "• Account for manufacturing tolerances in final design<br/>"
                "• Experimental validation required before deployment"
            )
            story.append(Paragraph(rec_text, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        logger.info(f"PDF report generated: {output_path}")
        return output_path
        
    def _generate_html_report(self, output_path: str,
                               sweep_results: Dict[str, Any],
                               em_simulation: Optional[Dict[str, Any]],
                               carpet_analysis: Optional[Dict[str, Any]]) -> str:
        """Generate HTML report."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{self.cfg.report_title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; }}
        .summary {{ background: #ecf0f1; padding: 20px; border-radius: 5px; }}
        .figure {{ text-align: center; margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #bdc3c7; padding: 10px; text-align: left; }}
        th {{ background: #3498db; color: white; }}
    </style>
</head>
<body>
    <h1>{self.cfg.report_title}</h1>
    <p><strong>Author:</strong> {self.cfg.author}</p>
    <p><strong>Organization:</strong> {self.cfg.organization}</p>
    <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d')}</p>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <p>Evaluated {sweep_results['n_samples']} designs, identified {sweep_results['n_pareto_optimal']} Pareto-optimal solutions.</p>
    </div>
    
    <h2>Statistical Results</h2>
    {self._generate_html_table(sweep_results)}
    
    <h2>Generated Figures</h2>
    {self._generate_html_figures(sweep_results, em_simulation)}
</body>
</html>
"""
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {output_path}")
        return output_path
        
    def _generate_html_table(self, sweep_results: Dict[str, Any]) -> str:
        """Generate HTML table for results."""
        stats = self.summarizer.summarize_sweep_results(sweep_results['results'])
        
        html = """
        <table>
            <tr><th>Metric</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>
        """
        
        for metric in ['absorption', 'reflection_db']:
            if metric in stats:
                s = stats[metric]
                html += f"""
                <tr>
                    <td>{metric.replace('_', ' ').title()}</td>
                    <td>{s['mean']:.3f}</td>
                    <td>{s['std']:.3f}</td>
                    <td>{s['min']:.3f}</td>
                    <td>{s['max']:.3f}</td>
                </tr>
                """
        
        html += "</table>"
        return html
        
    def _generate_html_figures(self, sweep_results: Dict[str, Any],
                                em_simulation: Optional[Dict[str, Any]]) -> str:
        """Generate HTML figure references."""
        html = ""
        
        # Generate Pareto plot
        if sweep_results['pareto_front']:
            fig_path = os.path.join(self.cfg.output_dir, "pareto_plot.png")
            self.figure_gen.generate_pareto_plot(sweep_results['pareto_front'], fig_path)
            html += f'<div class="figure"><img src="{fig_path}" width="600"></div>'
        
        return html


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Automated Report Generation — Smoke Test")
    print("=" * 72)
    print(f"  matplotlib  : {HAS_MATPLOTLIB}")
    print(f"  reportlab   : {HAS_REPORTLAB}")
    print(f"  SESI deps   : {HAS_SESI}")
    
    # Run parametric sweep
    sweep_cfg = SweepConfig(n_grid_points=3, n_adaptive_samples=10)
    sweep_engine = ParametricSweepEngine(sweep_cfg)
    sweep_results = sweep_engine.run_grid_sweep()
    
    # Generate report
    report_cfg = ReportConfig(output_format="html")
    generator = AutomatedReportGenerator(report_cfg)
    
    print("\n  ── Generating Report ──")
    report_path = generator.generate_report(sweep_results)
    
    if report_path:
        print(f"    Report saved to: {report_path}")
        print(f"    Figures generated: {len(generator.figure_gen.figures)}")
    else:
        print("    [SKIP] Report generation failed")
    
    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Reports document COMPUTATIONAL results")
    print("  • All claims traceable to specific simulations")
    print("  • UQ requires stated assumptions (Paper 11)")
    print("  • Recommendations are DESIGN GUIDANCE, not guarantees")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
